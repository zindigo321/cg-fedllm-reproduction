"""Fixed, TGAP-fitted AutoEncoder input normalisation (Phase 3, reviewer item A2).

The paper specifies NO input normalisation. These modes are DIAGNOSTIC / STABILISED VARIANTS; they are never
presented as recovered paper behaviour.

* ``none``        -- paper-literal: no arithmetic is applied at all.
* ``global_rms``  -- one scalar ``s`` = RMS over every element of the D1 AE-training split;
                     encode ``x / s``, decode ``s * dec(z)``.
* ``factor_rms``  -- two scalars ``s_A`` / ``s_B`` = RMS over all A (resp. B) elements of the D1 AE-training
                     split; every Phi column block is divided by the scale of the factor it holds before the
                     encoder and multiplied by it after the decoder (layout-aware, so it works for any layout).

The statistics are fitted once on the training split only, frozen, stored in the AE checkpoint metadata and
known to both client and server, so normalisation transmits nothing (zero logical uplink bytes). There is no
clipping: values outside the decoder's Tanh range are measured by :func:`range_report`, not hidden.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch

from cg_fedllm.compression.layout import Layout, LoRAGeometry, geometry_from_dict, get_layout

NORMALIZATION_MODES = ("none", "global_rms", "factor_rms")
QUANTILE_MAX_ELEMENTS = 2**26  # exact quantiles up to 64 Mi elements, a strided subsample above


def factor_column_mask(layout: Layout, geom: LoRAGeometry) -> torch.Tensor:
    """Boolean ``[W]`` mask of the Phi columns that hold A factors (the rest hold B)."""
    cols = []
    for _layer, _module, factor in layout.block_order(geom):
        cols += [factor == "A"] * geom.rank
    return torch.tensor(cols, dtype=torch.bool)


@dataclass(frozen=True)
class Normalizer:
    mode: str = "none"
    scale_global: float | None = None
    scale_A: float | None = None
    scale_B: float | None = None
    layout_id: str | None = None
    geometry: dict[str, Any] | None = None
    fit: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in NORMALIZATION_MODES:
            raise ValueError(f"unknown normalisation mode {self.mode!r}")
        scales = {"global_rms": (self.scale_global,), "factor_rms": (self.scale_A, self.scale_B)}.get(
            self.mode, ()
        )
        for s in scales:
            if s is None or not math.isfinite(s) or s <= 0:
                raise ValueError(f"{self.mode}: scales must be finite and positive, got {scales}")
        if self.mode == "factor_rms" and (self.layout_id is None or self.geometry is None):
            raise ValueError("factor_rms needs the Phi layout and geometry to locate the A/B columns")

    @property
    def is_identity(self) -> bool:
        return self.mode == "none"

    def column_scales(self) -> torch.Tensor:
        """Per-column scale ``[W]`` (float32) for ``factor_rms``."""
        geom = geometry_from_dict(self.geometry)
        mask = factor_column_mask(get_layout(self.layout_id), geom)
        return torch.where(
            mask,
            torch.tensor(self.scale_A, dtype=torch.float32),
            torch.tensor(self.scale_B, dtype=torch.float32),
        )

    def _scale_like(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "global_rms":
            return torch.tensor(self.scale_global, dtype=torch.float32, device=x.device)
        cs = self.column_scales().to(x.device)
        if x.shape[-1] != cs.numel():
            raise ValueError(f"input width {x.shape[-1]} does not match the fitted layout width {cs.numel()}")
        return cs

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` is Phi-shaped (``[..., d, W]``); returns ``x / scale``."""
        return x if self.is_identity else x / self._scale_like(x)

    def denormalize(self, y: torch.Tensor) -> torch.Tensor:
        return y if self.is_identity else y * self._scale_like(y)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> Normalizer:
        return cls() if not d else cls(**d)


def fit_normalizer(
    xs: Sequence[torch.Tensor],
    train_idx: Sequence[int],
    mode: str,
    *,
    layout: Layout | None = None,
    geom: LoRAGeometry | None = None,
) -> Normalizer:
    """Fit the frozen statistics on ``xs[i] for i in train_idx`` ONLY (float64 sums)."""
    if mode not in NORMALIZATION_MODES:
        raise ValueError(f"unknown normalisation mode {mode!r}")
    if not train_idx:
        raise ValueError("cannot fit a normaliser on an empty training split")
    fit = {
        "fitted_on": "train_split_only",
        "num_snapshots": len(train_idx),
        "train_indices": [int(i) for i in train_idx],
    }
    if mode == "none":
        return Normalizer("none", fit=fit)
    if mode == "global_rms":
        ss, n = 0.0, 0
        for i in train_idx:
            v = xs[i].double()
            ss += float((v * v).sum())
            n += v.numel()
        return Normalizer("global_rms", scale_global=math.sqrt(ss / n), fit={**fit, "num_elements": n})
    if layout is None or geom is None:
        raise ValueError("factor_rms needs the Phi layout and geometry")
    mask = factor_column_mask(layout, geom)
    ss_a = ss_b = 0.0
    n_a = n_b = 0
    for i in train_idx:
        v = xs[i].double()
        a, b = v[..., mask], v[..., ~mask]
        ss_a += float((a * a).sum())
        ss_b += float((b * b).sum())
        n_a += a.numel()
        n_b += b.numel()
    return Normalizer(
        "factor_rms",
        scale_A=math.sqrt(ss_a / n_a),
        scale_B=math.sqrt(ss_b / n_b),
        layout_id=layout.layout_id,
        geometry=geom.to_dict(),
        fit={**fit, "num_elements_A": n_a, "num_elements_B": n_b},
    )


def abs_quantiles(
    chunks: Sequence[np.ndarray], qs: Sequence[float], max_elements: int = QUANTILE_MAX_ELEMENTS
) -> dict[str, Any]:
    """Quantiles of the concatenated absolute values (exact unless a strided subsample is needed)."""
    total = int(sum(c.size for c in chunks))
    if total == 0:
        return {"n": 0, "exact": True}
    stride = max(1, math.ceil(total / max_elements))
    flat = np.abs(np.concatenate([c.reshape(-1)[::stride] for c in chunks]).astype(np.float64))
    vals = np.quantile(flat, list(qs))
    out = {f"p{_qname(q)}": float(v) for q, v in zip(qs, vals)}
    out.update(
        {
            "n": total,
            "exact": stride == 1,
            "stride": stride,
            "max": float(max(float(np.abs(c).max()) for c in chunks if c.size)),
        }
    )
    return out


def _qname(q: float) -> str:
    s = f"{100 * q:.6g}"
    return s.replace(".", "_")


def range_report(
    xs: Sequence[torch.Tensor], idx: Sequence[int], norm: Normalizer, layout: Layout, geom: LoRAGeometry
) -> dict[str, Any]:
    """Distribution of ``|normalize(x)|`` per factor and the fraction outside the decoder's Tanh range (|v| >= 1)."""
    mask = factor_column_mask(layout, geom)
    parts: dict[str, list[np.ndarray]] = {"A": [], "B": []}
    beyond = {"A": 0, "B": 0}
    count = {"A": 0, "B": 0}
    for i in idx:
        y = norm.normalize(xs[i].float())
        for name, sel in (("A", mask), ("B", ~mask)):
            v = y[..., sel].reshape(-1)
            parts[name].append(v.numpy())
            beyond[name] += int((v.abs() >= 1.0).sum())
            count[name] += v.numel()
    qs = (0.5, 0.9, 0.99, 0.999)
    out: dict[str, Any] = {"mode": norm.mode, "num_snapshots": len(idx)}
    for name in ("A", "B"):
        out[name] = {
            "fraction_abs_ge_1": beyond[name] / count[name] if count[name] else None,
            **abs_quantiles(parts[name], qs),
        }
    total = beyond["A"] + beyond["B"]
    out["all"] = {
        "fraction_abs_ge_1": total / (count["A"] + count["B"]) if count["A"] + count["B"] else None,
        **abs_quantiles(parts["A"] + parts["B"], qs),
    }
    return out
