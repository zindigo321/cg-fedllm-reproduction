"""Reconstruction and compression metrics.

For a transmitted tensor X (n elements) and its reconstruction X_hat (sums in float64):

* ``mse``                 = ||X - X_hat||^2 / n                         (per-element MSE)
* ``rel_sq_error``        = ||X - X_hat||^2 / ||X||^2
* ``snr_paper``           = ||X||^2 / mse = n * ||X||^2 / ||X - X_hat||^2
                            (the definition implied by the paper's tables: summed signal energy over a
                            per-element MSE -- DERIVED in Phase 1, inflated by the factor n)
* ``snr_standard``        = ||X||^2 / ||X - X_hat||^2 and ``snr_db = 10 log10(snr_standard)``
* ``cosine``              = <X, X_hat> / (||X|| ||X_hat||)
* ``innovation_ratio``    = ||X_hat - X||^2 / ||X - R||^2 for a reference R (e.g. the round-start global
                            state): < 1 means the channel carried information about the local update.
"""

from __future__ import annotations

import math
from typing import Any

import torch


def _f64(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to("cpu", torch.float64)


def reconstruction_metrics(x: torch.Tensor, x_hat: torch.Tensor, reference: torch.Tensor | None = None) -> dict[str, Any]:
    if x.shape != x_hat.shape:
        raise ValueError(f"shape mismatch {tuple(x.shape)} vs {tuple(x_hat.shape)}")
    a, b = _f64(x), _f64(x_hat)
    n = a.numel()
    sse = float(((a - b) ** 2).sum())
    signal = float((a**2).sum())
    mse = sse / n
    out: dict[str, Any] = {
        "numel": n,
        "signal_sq": signal,
        "sse": sse,
        "mse": mse,
        "rel_sq_error": sse / signal if signal > 0 else math.inf,
        "snr_paper": signal / mse if mse > 0 else math.inf,
        "snr_standard": signal / sse if sse > 0 else math.inf,
        "snr_db": 10 * math.log10(signal / sse) if sse > 0 and signal > 0 else math.inf,
        "cosine": float((a * b).sum() / (a.norm() * b.norm())) if a.norm() > 0 and b.norm() > 0 else float("nan"),
        "max_abs_x": float(a.abs().max()) if n else 0.0,
    }
    if reference is not None:
        r = _f64(reference)
        innov = float(((a - r) ** 2).sum())
        out["innovation_sq"] = innov
        out["innovation_ratio"] = sse / innov if innov > 0 else math.inf
        da, db = a - r, b - r
        out["delta_cosine"] = float((da * db).sum() / (da.norm() * db.norm())) if da.norm() > 0 and db.norm() > 0 else float("nan")
    return out


def compression_ratio(input_numel: int, payload_numel: int) -> float:
    return payload_numel / input_numel


def json_safe(obj: Any) -> Any:
    """Recursively replace non-finite floats (in dicts, lists and tuples) with strings for strict JSON."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return "inf" if obj > 0 else ("-inf" if obj < 0 else "nan")
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj
