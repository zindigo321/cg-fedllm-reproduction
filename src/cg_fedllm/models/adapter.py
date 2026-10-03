"""Canonical LoRA adapter state.

An :class:`AdapterState` maps *canonical* keys such as ``layers.3.self_attn.q_proj.lora_A.weight`` to
CPU float32 tensors (``lora_A``: ``r x in_features``; ``lora_B``: ``out_features x r``). Key order is
always derived explicitly by :func:`canonical_sort_key` -- never from dict insertion order -- so hashes,
aggregation order and the Phi layout are reproducible.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from cg_fedllm.utils.hashing import tensors_sha256
from cg_fedllm.utils.io import load_tensors, save_tensors_atomic

CANONICAL_RE = re.compile(r"^layers\.(\d+)\.(self_attn|mlp)\.([A-Za-z0-9_]+)\.lora_(A|B)\.weight$")
MODULE_RANK = {"q_proj": 0, "k_proj": 1, "v_proj": 2, "o_proj": 3}


class AdapterStateError(ValueError):
    pass


def parse_key(key: str) -> tuple[int, str, str, str]:
    m = CANONICAL_RE.match(key)
    if not m:
        raise AdapterStateError(f"not a canonical LoRA key: {key!r}")
    return int(m.group(1)), m.group(2), m.group(3), m.group(4)


def canonical_sort_key(key: str) -> tuple[int, int, str, int, str]:
    layer, block, module, factor = parse_key(key)
    return (layer, MODULE_RANK.get(module, 100), module, 0 if factor == "A" else 1, block)


def canonical_order(keys: Iterable[str]) -> list[str]:
    return sorted(keys, key=canonical_sort_key)


@dataclass
class AdapterState:
    tensors: dict[str, torch.Tensor]

    def __post_init__(self) -> None:
        for k, v in self.tensors.items():
            parse_key(k)
            if not isinstance(v, torch.Tensor):
                raise AdapterStateError(f"{k}: expected a tensor")

    # ---- structure -------------------------------------------------------------------------------
    def keys(self) -> list[str]:
        return canonical_order(self.tensors)

    def __getitem__(self, key: str) -> torch.Tensor:
        return self.tensors[key]

    def num_elements(self) -> int:
        return int(sum(t.numel() for t in self.tensors.values()))

    def shapes(self) -> dict[str, list[int]]:
        return {k: list(self.tensors[k].shape) for k in self.keys()}

    def same_structure(self, other: AdapterState) -> bool:
        return self.keys() == other.keys() and all(
            self.tensors[k].shape == other.tensors[k].shape for k in self.keys()
        )

    def _check_same(self, other: AdapterState) -> None:
        if not self.same_structure(other):
            raise AdapterStateError("adapter states have different keys/shapes")

    # ---- arithmetic (A and B tensors are always treated independently, key by key) --------------------
    def clone(self) -> AdapterState:
        return AdapterState({k: v.detach().clone() for k, v in self.tensors.items()})

    def sub(self, other: AdapterState) -> AdapterState:
        self._check_same(other)
        return AdapterState({k: self.tensors[k] - other.tensors[k] for k in self.keys()})

    def add(self, other: AdapterState) -> AdapterState:
        self._check_same(other)
        return AdapterState({k: self.tensors[k] + other.tensors[k] for k in self.keys()})

    def is_finite(self) -> bool:
        return all(bool(torch.isfinite(t).all()) for t in self.tensors.values())

    def equal(self, other: AdapterState) -> bool:
        return self.same_structure(other) and all(
            torch.equal(self.tensors[k], other.tensors[k]) for k in self.keys()
        )

    def max_abs_diff(self, other: AdapterState) -> float:
        self._check_same(other)
        return max(float((self.tensors[k] - other.tensors[k]).abs().max()) for k in self.keys())

    def relative_l2_diff(self, other: AdapterState) -> float:
        self._check_same(other)
        num = sum(
            float(((self.tensors[k].double() - other.tensors[k].double()) ** 2).sum()) for k in self.keys()
        )
        den = sum(float((other.tensors[k].double() ** 2).sum()) for k in self.keys())
        return (num / den) ** 0.5 if den > 0 else float(num > 0)

    # ---- statistics / identity ----------------------------------------------------------------------
    def l2_sq(self, factor: str | None = None) -> float:
        total = 0.0
        for k in self.keys():
            if factor is None or parse_key(k)[3] == factor:
                total += float((self.tensors[k].double() ** 2).sum())
        return total

    def sha256(self) -> str:
        return tensors_sha256(self.tensors, self.keys())

    # ---- IO ---------------------------------------------------------------------------------------------
    def save(self, path: str | Path, metadata: Mapping[str, str] | None = None) -> Path:
        meta = {
            "format": "cg_fedllm.adapter_state/v1",
            "key_order": json.dumps(self.keys()),
            "sha256": self.sha256(),
        }
        meta.update(metadata or {})
        return save_tensors_atomic(path, {k: self.tensors[k] for k in self.keys()}, meta)

    @classmethod
    def load(cls, path: str | Path, verify: bool = True) -> AdapterState:
        from safetensors import safe_open

        tensors = load_tensors(path)
        state = cls(tensors)
        if verify:
            with safe_open(str(path), framework="pt") as fh:
                meta = fh.metadata() or {}
            if "sha256" in meta and meta["sha256"] != state.sha256():
                raise AdapterStateError(f"{path}: stored adapter hash does not match its tensors")
        return state


def weighted_sum(states: Sequence[AdapterState], weights: torch.Tensor) -> AdapterState:
    """sum_k weights[k] * states[k], accumulated in list order (A and B keys independently)."""
    if not states:
        raise AdapterStateError("no states to aggregate")
    for s in states[1:]:
        states[0]._check_same(s)
    keys = states[0].keys()
    out = {k: states[0].tensors[k] * weights[0] for k in keys}
    for i in range(1, len(states)):
        for k in keys:
            out[k] = out[k] + states[i].tensors[k] * weights[i]
    return AdapterState(out)


def l1_normalised(values: Sequence[float]) -> torch.Tensor:
    """Shepherd's weight normalisation: ``torch.nn.functional.normalize(tensor(values, f32), p=1, dim=0)``."""
    return F.normalize(torch.tensor(list(values), dtype=torch.float32), p=1, dim=0)
