"""P5-A v2 artifact publication and the run-wide byte policy (protocol sections 13.2-13.4).

Every P5-A output is serialised to bytes first, so its exact size is known before publication. One publication:

1. checks the file against its fixed cap (section 13.2);
2. checks the reservation ``F + cap(this) + sum(caps(Q)) <= 20,000,000`` (section 13.3), where ``F`` is the measured
   footprint (exact sizes of all regular files under the run root, temporaries included) and ``Q`` the other
   artifacts that may still be required (computed by the caller from the persistent state);
3. writes the bytes to a temporary file owned by this invocation (``O_EXCL``), flushed and fsynced;
4. publishes it with ``os.link(tmp, final)``, which fails with ``FileExistsError`` if ``final`` exists (no-clobber,
   atomic on NTFS and POSIX), then removes the temporary name.

Only this invocation's temporary files are ever removed. A finalized artifact is never deleted or overwritten. If hard
links are unsupported, publication fails; there is no overwrite fallback. Caps are used only in the reservation
check; reported byte counts are always the measured footprint.
"""

from __future__ import annotations

import json
import os
import re
import secrets
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

RETAINED_BYTES_LIMIT = 20_000_000  # decimal bytes, all regular files under the run root (P5v2-D45)
TMP_SUFFIX = ".p5a-tmp"
PROVENANCE_FILES = ("config.resolved.yaml", "config.sha256.json", "run_metadata.json")

# section 13.2: artifact kind -> (cap in bytes, maximum count)
CAPS: dict[str, tuple[int, int]] = {
    "preflight": (300_000, 3),
    "invocation": (50_000, 3),
    "summary": (200_000, 3),
    "start": (20_000, 6),
    "end": (20_000, 6),
    "failure": (100_000, 6),
    "provenance_set": (150_000, 6),  # the three PROVENANCE_FILES of one control together
    "record": (500_000, 6),
    "checkpoint": (2_100_000, 6),
}
PER_CONTROL_KINDS = ("start", "end", "failure", "provenance_set", "record", "checkpoint")
RUN_LEVEL_KINDS = ("invocation", "summary")


def worst_case_bytes() -> int:
    """Section 13.2 estimated worst-case footprint: every cap at its maximum count (a planning bound only)."""
    return sum(cap * count for cap, count in CAPS.values())


def training_worst_case_bytes() -> int:
    """Every training-stage artifact at its maximum count (reserved by a preflight, section 13.3)."""
    return sum(cap * count for kind, (cap, count) in CAPS.items() if kind != "preflight")


_LEDGER = re.compile(r"^ledger/(invocation|summary)_[1-3]\.json$")
_CONTROL_LEDGER = re.compile(r"^ledger/(start|end|failure)_R[2-4]_(single_snapshot|fixed_subset4)\.json$")
_PREFLIGHT = re.compile(r"^preflight/preflight_[1-3]\.json$")


def artifact_kind(rel: str) -> str:
    """The section-13.2 kind of a run-root relative path; any other path is refused."""
    if _PREFLIGHT.match(rel):
        return "preflight"
    if m := _LEDGER.match(rel):
        return m.group(1)
    if m := _CONTROL_LEDGER.match(rel):
        return m.group(1)
    parts = PurePosixPath(rel).parts
    if len(parts) == 2 and re.match(r"^R[2-4]_[a-z0-9_]+_(single_snapshot|fixed_subset4)$", parts[0]):
        if parts[1] in PROVENANCE_FILES:
            return "provenance_set"
        if parts[1] == "p5a_record.json":
            return "record"
        if parts[1] == "autoencoder_p5a_final.safetensors":
            return "checkpoint"
    raise PublicationError(f"{rel!r} is not a P5-A v2 artifact (section 13.2)")


class PublicationError(RuntimeError):
    """A file could not be finalized (filesystem error, collision or cap)."""


class ByteCeilingError(PublicationError):
    """The cap or the reservation check refused the file; nothing was written."""

    def __init__(self, message: str, accounting: dict[str, Any]) -> None:
        super().__init__(message)
        self.accounting = accounting


class ExistingPathError(PublicationError):
    """A target path already exists; it was neither overwritten nor deleted."""


def json_bytes(obj: Any) -> bytes:
    """Strict, sorted, indented JSON (as ``atomic_write_json``); non-finite floats are rejected (section 9.3)."""
    return (json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _write_tmp(path: Path, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    fd = os.open(path, flags, 0o644)
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())


@dataclass
class ArtifactStore:
    """Publishes P5-A files under ``root``; ``writer`` and ``linker`` are injectable for failure tests."""

    root: Path
    limit: int = RETAINED_BYTES_LIMIT
    writer: Callable[[Path, bytes], None] = _write_tmp
    linker: Callable[[Path, Path], None] = os.link

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.published: list[dict[str, Any]] = []
        self.token = secrets.token_hex(6)

    # ---- state ------------------------------------------------------------------------------------------
    @staticmethod
    def _rel(rel: str) -> PurePosixPath:
        p = PurePosixPath(rel)
        if p.is_absolute() or ".." in p.parts or "\\" in rel or not p.parts or p.name.endswith(TMP_SUFFIX):
            raise PublicationError(f"unsafe artifact path {rel!r}")
        return p

    def path(self, rel: str) -> Path:
        return self.root.joinpath(*self._rel(rel).parts)

    def exists(self, rel: str) -> bool:
        return self.path(rel).exists()

    def size(self, rel: str) -> int:
        p = self.path(rel)
        return p.stat().st_size if p.is_file() else 0

    def read_json(self, rel: str) -> dict[str, Any] | None:
        p = self.path(rel)
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None

    def files(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(p.relative_to(self.root).as_posix() for p in self.root.rglob("*") if p.is_file())

    def footprint(self) -> int:
        """Measured footprint: exact sizes of every regular file under the run root (temporaries included)."""
        if not self.root.exists():
            return 0
        return sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())

    # ---- publication ------------------------------------------------------------------------------------
    def check(self, rel: str, data: bytes, reserved_other: int) -> dict[str, Any]:
        """Cap and reservation accounting for one file (section 13.3); ``fits`` is the decision."""
        kind = artifact_kind(str(self._rel(rel)))
        cap = CAPS[kind][0]
        if kind == "provenance_set":  # one cap for the control's three files together: what is left of it
            cap -= provenance_set_bytes(self, PurePosixPath(rel).parts[0])
        footprint = self.footprint()
        acc = {
            "path": rel,
            "kind": kind,
            "bytes": len(data),
            "cap": cap,
            "within_cap": len(data) <= cap,
            "footprint_before": footprint,
            "reserved_other_caps": int(reserved_other),
            "limit": self.limit,
        }
        acc["reservation_total"] = footprint + cap + int(reserved_other)
        acc["fits"] = acc["within_cap"] and acc["reservation_total"] <= self.limit
        acc["resource_stop_cause"] = None if acc["fits"] else refusal_cause(acc)
        return acc

    def publish(self, rel: str, data: bytes, *, reserved_other: int) -> dict[str, Any]:
        """Check, then finalize one file no-clobber. Raises :class:`ByteCeilingError` before writing anything if the
        cap or the reservation is exceeded, :class:`ExistingPathError` if the path exists."""
        final = self.path(rel)
        if final.exists():
            raise ExistingPathError(f"P5-A output path already exists: {rel}")
        acc = self.check(rel, data, reserved_other)
        if not acc["fits"]:
            raise ByteCeilingError(f"{rel}: refused by the byte policy (section 13.3)", acc)
        final.parent.mkdir(parents=True, exist_ok=True)
        tmp = final.with_name(f".{final.name}.{self.token}{TMP_SUFFIX}")
        try:
            self.writer(tmp, data)
            if tmp.stat().st_size != len(data):
                raise OSError(f"{tmp}: short write")
            self.linker(tmp, final)  # FileExistsError if final exists: never overwritten
        except FileExistsError as exc:
            raise ExistingPathError(f"{rel} appeared before publication; left untouched") from exc
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            raise PublicationError(f"{rel}: {type(exc).__name__}: {exc}") from exc
        finally:
            # the temporary name is this invocation's own; the finalized link (if any) is not touched
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass
        entry = {"path": rel, "bytes": len(data), "footprint_before": acc["footprint_before"]}
        self.published.append(entry)
        return entry


def refusal_cause(acc: dict[str, Any]) -> str:
    """v2.1 A4.1: why the byte policy refused a file (reporting only; the v2 section-13.3 check is unchanged)."""
    if not acc["within_cap"]:
        return "per_file_cap"
    if acc["footprint_before"] + acc["bytes"] > acc["limit"]:
        return "aggregate_byte_cap"
    return "evidence_reservation"


def caps_of(kinds: Iterable[str]) -> int:
    return sum(CAPS[k][0] for k in kinds)


def provenance_set_bytes(store: ArtifactStore, control_dir: str) -> int:
    """Bytes already finalized of one control's provenance set (its cap is shared by the three files)."""
    return sum(store.size(f"{control_dir}/{n}") for n in PROVENANCE_FILES)
