"""Atomic, checkpoint-safe IO helpers (pathlib only).

Writes go to a temporary sibling file that is then ``os.replace``-d into place, so an interrupted run
never leaves a truncated checkpoint that a later resume would trust.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from cg_fedllm.utils.hashing import canonical_json


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def atomic_write_bytes(path: str | Path, data: bytes) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def atomic_write_text(path: str | Path, text: str) -> Path:
    return atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: str | Path, obj: Any, canonical: bool = False) -> Path:
    """Write JSON atomically. ``canonical=True`` uses the hash-stable canonical encoding."""
    if canonical:
        return atomic_write_bytes(path, canonical_json(obj))
    text = json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return atomic_write_text(path, text + "\n")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def append_jsonl(path: str | Path, record: Mapping[str, Any]) -> None:
    """Append one JSON record (one line). The file is flushed and fsynced after every record."""
    path = Path(path)
    ensure_dir(path.parent)
    line = json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(line)
        fh.flush()
        os.fsync(fh.fileno())


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def save_tensors_atomic(
    path: str | Path, tensors: Mapping[str, torch.Tensor], metadata: Mapping[str, str] | None = None
) -> Path:
    """Save tensors with safetensors via a temporary file + atomic rename."""
    path = Path(path)
    ensure_dir(path.parent)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        contiguous = {k: v.detach().to("cpu").contiguous() for k, v in tensors.items()}
        save_file(contiguous, tmp, metadata=dict(metadata) if metadata else None)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def load_tensors(path: str | Path) -> dict[str, torch.Tensor]:
    return load_file(str(path), device="cpu")
