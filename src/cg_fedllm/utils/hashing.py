"""Deterministic hashing helpers.

All hashes are SHA-256 hex digests. ``canonical_json`` defines the byte representation used to hash
manifests and configs: UTF-8, sorted keys, no insignificant whitespace, trailing newline. Tensor hashes
include dtype and shape so that e.g. an fp32 and a bf16 tensor with coincident bytes never collide.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch

_CHUNK = 1 << 20


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Streaming SHA-256 of a file's raw bytes."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(obj: Any) -> bytes:
    """Canonical JSON bytes (sorted keys, compact separators, UTF-8, trailing newline)."""
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return (text + "\n").encode("utf-8")


def canonical_json_sha256(obj: Any) -> str:
    return sha256_bytes(canonical_json(obj))


def _update_with_tensor(h: hashlib._Hash, name: str, tensor: torch.Tensor) -> None:
    t = tensor.detach().to("cpu").contiguous()
    h.update(name.encode("utf-8"))
    h.update(b"\x00")
    h.update(str(t.dtype).encode("utf-8"))
    h.update(repr(tuple(t.shape)).encode("utf-8"))
    # Reinterpret as raw bytes (works for bf16, which numpy cannot represent, and for 0-dim tensors).
    h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b"")


def tensor_sha256(tensor: torch.Tensor, name: str = "") -> str:
    h = hashlib.sha256()
    _update_with_tensor(h, name, tensor)
    return h.hexdigest()


def tensors_sha256(tensors: Mapping[str, torch.Tensor], order: Iterable[str]) -> str:
    """Hash a collection of named tensors in an *explicit* key order (never dict iteration order)."""
    h = hashlib.sha256()
    order = list(order)
    if sorted(order) != sorted(tensors.keys()):
        raise ValueError("order must be a permutation of the tensor keys")
    for key in order:
        _update_with_tensor(h, key, tensors[key])
    return h.hexdigest()
