"""Run provenance capture: Git state, Python/package versions, CUDA/GPU facts.

Every real run writes the dictionary returned by :func:`collect_environment` to ``run_metadata.json``
next to its fully resolved config. Nothing here is required for correctness; everything is required
for auditability.
"""

from __future__ import annotations

import datetime as _dt
import importlib.metadata as md
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

KEY_PACKAGES = (
    "torch",
    "transformers",
    "peft",
    "accelerate",
    "safetensors",
    "numpy",
    "pyarrow",
    "PyYAML",
    "huggingface_hub",
    "tokenizers",
    "datasets",
    "bitsandbytes",
    "lm_eval",
    "pandas",
)

ENV_VARS_OF_INTEREST = (
    "HF_HOME",
    "HF_HUB_OFFLINE",
    "CUBLAS_WORKSPACE_CONFIG",
    "PYTHONHASHSEED",
    "CUDA_VISIBLE_DEVICES",
)


def _run(cmd: list[str], cwd: Path | None = None) -> str | None:
    try:
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def find_repo_root(start: Path | None = None) -> Path | None:
    p = (start or Path(__file__)).resolve()
    for cand in [p, *p.parents]:
        if (cand / ".git").exists():
            return cand
    return None


def git_info(repo_root: Path | None = None) -> dict[str, Any]:
    root = repo_root or find_repo_root()
    if root is None or shutil.which("git") is None:
        return {"available": False}
    status = _run(["git", "status", "--porcelain=v1"], cwd=root)
    return {
        "available": True,
        "commit": _run(["git", "rev-parse", "HEAD"], cwd=root),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root),
        "dirty": bool(status) if status is not None else None,
        "dirty_files": status.splitlines()[:50] if status else [],
    }


def package_versions(names: tuple[str, ...] = KEY_PACKAGES) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for n in names:
        try:
            out[n] = md.version(n)
        except md.PackageNotFoundError:
            out[n] = None
    return out


def cuda_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(
            {
                "device_name": props.name,
                "device_total_memory_bytes": int(props.total_memory),
                "compute_capability": f"{props.major}.{props.minor}",
                "bf16_supported": torch.cuda.is_bf16_supported(),
            }
        )
    smi = _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
    info["nvidia_smi"] = smi
    return info


def collect_environment(repo_root: Path | None = None) -> dict[str, Any]:
    return {
        "captured_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": package_versions(),
        "cuda": cuda_info(),
        "git": git_info(repo_root),
        "env": {k: os.environ.get(k) for k in ENV_VARS_OF_INTEREST},
    }
