"""Explicit seed derivation.

Every stochastic operation in a run draws from a stream derived from ``(base_seed, *keys)``, e.g.
``derive_seed(seed, "fl", round_index, client_id, "data_order")``. Because no stream depends on the
state of a *global* RNG, a run resumed from a round checkpoint reproduces exactly the same streams as
an uninterrupted run, and adding/removing an unrelated random draw cannot silently shift others.
"""

from __future__ import annotations

import hashlib
import os
import random

import numpy as np
import torch

_MASK_63 = (1 << 63) - 1


def derive_seed(base_seed: int, *keys: int | str) -> int:
    """Derive a 63-bit seed from a base seed and an ordered tuple of keys (stable across platforms)."""
    if not isinstance(base_seed, int):
        raise TypeError("base_seed must be an int")
    parts = [f"i:{base_seed}"]
    for k in keys:
        if isinstance(k, bool) or not isinstance(k, (int, str)):
            raise TypeError(f"seed key must be int or str, got {type(k).__name__}")
        parts.append(f"{'i' if isinstance(k, int) else 's'}:{k}")
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & _MASK_63


def numpy_rng(base_seed: int, *keys: int | str) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(derive_seed(base_seed, *keys)))


def torch_generator(base_seed: int, *keys: int | str, device: str | torch.device = "cpu") -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(derive_seed(base_seed, *keys))
    return g


def seed_all(seed: int) -> None:
    """Seed Python, NumPy (legacy global) and torch (CPU + CUDA) global generators."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_determinism(deterministic: bool, num_threads: int | None = None) -> dict[str, object]:
    """Configure torch for reproducible execution and report what was set.

    ``CUBLAS_WORKSPACE_CONFIG`` must be set before the first cuBLAS call; the CLI calls this function
    at process start. With ``deterministic=True`` nondeterministic kernels raise instead of silently
    running, which is how GPU nondeterminism is detected rather than hidden.
    """
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    # TF32 changes fp32 matmul numerics; keep it off so fp32 runs are true fp32.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if num_threads is not None:
        torch.set_num_threads(int(num_threads))
    return {
        "deterministic_algorithms": deterministic,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn_benchmark": False,
        "allow_tf32": False,
        "torch_num_threads": torch.get_num_threads(),
    }
