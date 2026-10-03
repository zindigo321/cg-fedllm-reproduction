"""Databricks-Dolly-15k source handling (Shepherd's pinned copy).

The paper fine-tunes on Dolly via FedIT/Shepherd, which ships ``new-databricks-dolly-15k.json`` (the
*first* Dolly release, 15,015 records; the current Hugging Face release has 15,011). The file is pinned
by Shepherd commit and SHA-256 in the data config and is downloaded into a cache *outside* the Git
repository. Records are identified by their 0-based position in that file (``source_id``); only these
integer IDs are ever committed to Git (the text is CC BY-SA 3.0 and is not redistributed).
"""

from __future__ import annotations

import json
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from cg_fedllm.config import DataSection, resolve_path
from cg_fedllm.utils.hashing import sha256_file

REQUIRED_FIELDS = ("instruction", "context", "response", "category")


class SourceIntegrityError(RuntimeError):
    """The downloaded/cached source does not match its pinned SHA-256 or schema."""


@dataclass(frozen=True)
class DollyRecord:
    source_id: int
    instruction: str
    context: str
    response: str
    category: str


def source_cache_path(cfg: DataSection) -> Path:
    base = resolve_path(cfg.cache_dir)
    return base / cfg.source_name / f"{cfg.source_sha256[:16]}.json"


def fetch_source(cfg: DataSection, timeout: int = 300) -> Path:
    """Return the verified local path of the pinned source, downloading it if necessary."""
    target = source_cache_path(cfg)
    if target.exists():
        digest = sha256_file(target)
        if digest != cfg.source_sha256:
            raise SourceIntegrityError(
                f"cached source {target} has sha256 {digest}, expected {cfg.source_sha256}"
            )
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False, suffix=".part") as tmp:
        tmp_path = Path(tmp.name)
        with urllib.request.urlopen(cfg.source_url, timeout=timeout) as resp:  # noqa: S310 (pinned https URL)
            while chunk := resp.read(1 << 20):
                tmp.write(chunk)
    digest = sha256_file(tmp_path)
    if digest != cfg.source_sha256:
        tmp_path.unlink(missing_ok=True)
        raise SourceIntegrityError(
            f"downloaded {cfg.source_url} has sha256 {digest}, expected {cfg.source_sha256}"
        )
    tmp_path.replace(target)
    return target


def load_records(path: Path, expected_num_records: int | None = None) -> list[DollyRecord]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise SourceIntegrityError("Dolly source must be a JSON array of records")
    if expected_num_records is not None and len(raw) != expected_num_records:
        raise SourceIntegrityError(f"expected {expected_num_records} records, found {len(raw)}")
    out = []
    for i, rec in enumerate(raw):
        missing = [f for f in REQUIRED_FIELDS if f not in rec]
        if missing:
            raise SourceIntegrityError(f"record {i} is missing fields {missing}")
        out.append(
            DollyRecord(
                source_id=i,
                instruction=str(rec["instruction"]),
                context=str(rec["context"]),
                response=str(rec["response"]),
                category=str(rec["category"]),
            )
        )
    return out


def load_source(cfg: DataSection) -> list[DollyRecord]:
    return load_records(fetch_source(cfg), cfg.source_num_records)
