"""The Phi layout: LoRA factors <-> one 2-D AutoEncoder input, and its inverse.

Reconstruction hypothesis (reviewer decision R4; ordering is an INFERENCE, the shape is from the paper):
the paper stacks the LoRA factors of all Q/K/V/O projections of all layers into one matrix of shape
(8 x 2 x 4 x 32) x 4096 for LLaMA-7B with r=8 (appendix), i.e. ``[1, 4096, 2048]`` (Table 1).

``layer_major_qkvo_AtB`` places, for layer l = 0..L-1 and module m in (q, k, v, o), the column blocks
``A_{l,m}^T`` (d x r) followed by ``B_{l,m}`` (d x r), giving ``X in R^{1 x d x 2*4*L*r}``.
``module_major_qkvo_AtB`` is an alternative ordering (m outer, l inner) that exists so later phases can
test ordering sensitivity without touching TGAP/FAF code.

Phi and Phi^-1 only copy/transpose values, so the round trip is bitwise exact. Geometry is validated and
incompatible adapters (e.g. GQA k/v projections whose ``out_features != d``, missing modules, MLP
targets) raise :class:`LayoutGeometryError` instead of being silently reshaped.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import torch

from cg_fedllm.models.adapter import AdapterState, parse_key
from cg_fedllm.utils.hashing import canonical_json_sha256

QKVO = ("q_proj", "k_proj", "v_proj", "o_proj")


class LayoutGeometryError(ValueError):
    pass


@dataclass(frozen=True)
class LoRAGeometry:
    num_layers: int
    modules: tuple[str, ...]
    rank: int
    hidden: int

    @property
    def phi_shape(self) -> tuple[int, int, int]:
        return (1, self.hidden, 2 * len(self.modules) * self.num_layers * self.rank)

    @property
    def num_elements(self) -> int:
        return 2 * len(self.modules) * self.num_layers * self.rank * self.hidden

    def to_dict(self) -> dict:
        return {
            "num_layers": self.num_layers,
            "modules": list(self.modules),
            "rank": self.rank,
            "hidden": self.hidden,
        }


def infer_geometry(state: AdapterState, modules: tuple[str, ...] = QKVO) -> LoRAGeometry:
    """Validate that ``state`` is a uniform (d x d projections) q/k/v/o LoRA stack and return its geometry."""
    layers: dict[int, set[str]] = {}
    ranks, hiddens = set(), set()
    for key in state.keys():
        layer, block, module, factor = parse_key(key)
        if block != "self_attn" or module not in modules:
            raise LayoutGeometryError(
                f"{key}: only self_attn {modules} LoRA factors are supported by this layout"
            )
        t = state.tensors[key]
        if t.dim() != 2:
            raise LayoutGeometryError(f"{key}: expected a 2-D tensor")
        if factor == "A":
            r, d_in = t.shape
            ranks.add(int(r))
            hiddens.add(int(d_in))
        else:
            d_out, r = t.shape
            ranks.add(int(r))
            hiddens.add(int(d_out))
        layers.setdefault(layer, set()).add(f"{module}.{factor}")
    if len(ranks) != 1:
        raise LayoutGeometryError(f"non-uniform LoRA rank {sorted(ranks)}")
    if len(hiddens) != 1:
        raise LayoutGeometryError(
            f"non-uniform projection dimensions {sorted(hiddens)}: every A must be r x d and every B d x r "
            "(GQA k/v projections are incompatible with this layout)"
        )
    expected = {f"{m}.{f}" for m in modules for f in ("A", "B")}
    n_layers = len(layers)
    if sorted(layers) != list(range(n_layers)):
        raise LayoutGeometryError(f"layers must be contiguous from 0, got {sorted(layers)}")
    for layer, present in layers.items():
        if present != expected:
            raise LayoutGeometryError(
                f"layer {layer}: expected factors {sorted(expected)}, got {sorted(present)}"
            )
    return LoRAGeometry(n_layers, tuple(modules), ranks.pop(), hiddens.pop())


class Layout:
    """Base class: subclasses define the column-block order."""

    layout_id = "abstract"

    def block_order(self, geom: LoRAGeometry) -> list[tuple[int, str, str]]:  # (layer, module, factor)
        raise NotImplementedError

    @staticmethod
    def key(layer: int, module: str, factor: str) -> str:
        return f"layers.{layer}.self_attn.{module}.lora_{factor}.weight"

    def key_order(self, geom: LoRAGeometry) -> list[str]:
        return [self.key(*b) for b in self.block_order(geom)]

    def metadata(self, geom: LoRAGeometry) -> dict:
        keys = self.key_order(geom)
        return {
            "layout_id": self.layout_id,
            "geometry": geom.to_dict(),
            "phi_shape": list(geom.phi_shape),
            "block": "A: transpose(A) (d x r); B: B (d x r)",
            "key_order_sha256": canonical_json_sha256(keys),
            "key_order": keys,
        }

    def forward(self, state: AdapterState, geom: LoRAGeometry | None = None) -> torch.Tensor:
        """Phi: adapter state -> ``[1, d, 2*M*L*r]``."""
        g = geom or infer_geometry(state)
        if geom is not None and infer_geometry(state) != geom:
            raise LayoutGeometryError("adapter state does not match the requested geometry")
        blocks = []
        for layer, module, factor in self.block_order(g):
            t = state.tensors[self.key(layer, module, factor)]
            blocks.append(t.t() if factor == "A" else t)
        return torch.cat(blocks, dim=1).unsqueeze(0).contiguous()

    def inverse(self, x: torch.Tensor, geom: LoRAGeometry) -> AdapterState:
        """Phi^-1: ``[1, d, 2*M*L*r]`` -> adapter state."""
        if tuple(x.shape) != geom.phi_shape:
            raise LayoutGeometryError(f"Phi^-1 expects shape {geom.phi_shape}, got {tuple(x.shape)}")
        mat = x[0]
        out: dict[str, torch.Tensor] = {}
        for i, (layer, module, factor) in enumerate(self.block_order(geom)):
            block = mat[:, i * geom.rank : (i + 1) * geom.rank]
            out[self.key(layer, module, factor)] = (
                (block.t() if factor == "A" else block).contiguous().clone()
            )
        return AdapterState(out)


class LayerMajorQKVOAtB(Layout):
    layout_id = "layer_major_qkvo_AtB"

    def block_order(self, geom: LoRAGeometry) -> list[tuple[int, str, str]]:
        return [(layer, m, f) for layer in range(geom.num_layers) for m in geom.modules for f in ("A", "B")]


class ModuleMajorQKVOAtB(Layout):
    layout_id = "module_major_qkvo_AtB"

    def block_order(self, geom: LoRAGeometry) -> list[tuple[int, str, str]]:
        return [(layer, m, f) for m in geom.modules for layer in range(geom.num_layers) for f in ("A", "B")]


LAYOUTS: dict[str, Layout] = {cls.layout_id: cls() for cls in (LayerMajorQKVOAtB, ModuleMajorQKVOAtB)}


def get_layout(layout_id: str) -> Layout:
    try:
        return LAYOUTS[layout_id]
    except KeyError:
        raise ValueError(f"unknown layout {layout_id!r}; available: {sorted(LAYOUTS)}") from None


def geometry_from_dict(d: dict) -> LoRAGeometry:
    return LoRAGeometry(int(d["num_layers"]), tuple(d["modules"]), int(d["rank"]), int(d["hidden"]))


def geometry_json(geom: LoRAGeometry) -> str:
    return json.dumps(geom.to_dict(), sort_keys=True)
