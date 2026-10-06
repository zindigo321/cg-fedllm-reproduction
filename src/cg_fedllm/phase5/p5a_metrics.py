"""P5-A v2 metrics (protocol section 9), predictors (section 8) and criteria C0-C4 (section 10).

Every quantity is computed in float64 in the original (de-normalised) space over all elements. A metric that is
undefined (zero denominator) or not finite is stored as ``None`` with a reason in the sibling ``undefined`` map
(section 9.3), so records are strict JSON. An undefined or non-finite gated value fails its criterion. Nothing is
filtered before pooling: pooled values are ratios of summed float64 sums (``math.fsum``), never means of ratios.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch

from cg_fedllm.compression.normalization import Normalizer
from cg_fedllm.phase5.p5a import NEAR_THRESHOLD_ULPS, THRESHOLDS

METRIC_FIELDS = ("sse", "sig", "hat", "dot", "mse", "zero_predictor_mse", "rse", "cosine", "norm_ratio")


def snapshot_sums(x: torch.Tensor, p: torch.Tensor) -> dict[str, float]:
    if x.shape != p.shape:
        raise ValueError(f"shape mismatch {tuple(x.shape)} vs {tuple(p.shape)}")
    a = x.detach().to("cpu", torch.float64)
    b = p.detach().to("cpu", torch.float64)
    return {
        "n": a.numel(),
        "sse": float(((b - a) ** 2).sum()),
        "sig": float((a * a).sum()),
        "hat": float((b * b).sum()),
        "dot": float((b * a).sum()),
    }


def _non_finite_reason(v: float) -> str:
    return "non_finite:nan" if math.isnan(v) else ("non_finite:+inf" if v > 0 else "non_finite:-inf")


def metrics_from_sums(sse: float, sig: float, hat: float, dot: float, n_total: int) -> dict[str, Any]:
    """Section 9.1 quantities from float64 sums; ``None`` + ``undefined[field]`` for undefined/non-finite values."""
    undefined: dict[str, str] = {}
    vals: dict[str, float | None] = {"sse": sse, "sig": sig, "hat": hat, "dot": dot}
    vals["mse"] = sse / n_total
    vals["zero_predictor_mse"] = sig / n_total
    if sig == 0:
        undefined.update(rse="sig_zero", cosine="sig_zero", norm_ratio="sig_zero")
    else:
        vals["rse"] = sse / sig
        vals["norm_ratio"] = math.sqrt(hat / sig)  # hat is a sum of squares: >= 0, +inf or NaN
        if hat == 0:
            undefined["cosine"] = "hat_zero"
        else:
            vals["cosine"] = dot / math.sqrt(sig * hat)
    out: dict[str, Any] = {}
    for field in METRIC_FIELDS:
        v = vals.get(field)
        if field in undefined:
            out[field] = None
        elif v is not None and not math.isfinite(v):
            out[field], undefined[field] = None, _non_finite_reason(v)
        else:
            out[field] = v
    out["undefined"] = undefined
    return out


def predictor_metrics(xs: Sequence[torch.Tensor], preds: Sequence[torch.Tensor]) -> dict[str, Any]:
    """Per-snapshot and pooled section-9 metrics for one predictor over the selected snapshots ``xs``."""
    if len(xs) != len(preds) or not xs:
        raise ValueError("need one prediction per selected snapshot")
    sums = [snapshot_sums(x, p) for x, p in zip(xs, preds)]
    per = [metrics_from_sums(s["sse"], s["sig"], s["hat"], s["dot"], s["n"]) for s in sums]
    pooled = metrics_from_sums(
        math.fsum(s["sse"] for s in sums),
        math.fsum(s["sig"] for s in sums),
        math.fsum(s["hat"] for s in sums),
        math.fsum(s["dot"] for s in sums),
        sum(s["n"] for s in sums),
    )
    return {"per_snapshot": per, "pooled": pooled}


def zero_prediction(x: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(x)


def tanh_range_ceiling(x: torch.Tensor, norm: Normalizer) -> torch.Tensor:
    """Inherited predictor ``denorm(clamp(norm(x), -1, 1))`` (``cabc3d7``; F6 ``cmd_forensic_screen``)."""
    return norm.denormalize(norm.normalize(x).clamp(-1.0, 1.0))


def subset_mean(xs: Sequence[torch.Tensor]) -> torch.Tensor:
    """``m = (1/k) sum_j x_j`` in float64 over the control's selected snapshots (section 9.1)."""
    return torch.stack([x.detach().to("cpu", torch.float64) for x in xs]).sum(0) / len(xs)


def _ok(v: Any) -> bool:
    return isinstance(v, float) and math.isfinite(v)


def compare(value: Any, op: str, threshold: float, reason: str | None = None) -> dict[str, Any]:
    """One float64 comparison against a literal threshold (inclusive, no epsilon); undefined fails (section 10)."""
    row: dict[str, Any] = {"value": value if _ok(value) else None, "operator": op, "threshold": threshold}
    if not _ok(value) or not math.isfinite(threshold):
        row.update(pass_=False, fail_reason=reason or "undefined_or_non_finite")
    else:
        row["pass_"] = value <= threshold if op == "<=" else value >= threshold
        row["margin"] = value - threshold
        row["near_threshold"] = abs(value - threshold) <= NEAR_THRESHOLD_ULPS * math.ulp(threshold)
    row["pass"] = row.pop("pass_")
    return row


def _per_snapshot(
    control: str, per: Sequence[dict[str, Any]], field: str, op: str, key: str
) -> dict[str, Any]:
    thr = THRESHOLDS[control][key]
    rows = [compare(m[field], op, thr, m["undefined"].get(field)) for m in per]
    return {
        "field": field,
        "operator": op,
        "threshold": thr,
        "per_snapshot": rows,
        "pass": all(r["pass"] for r in rows),
    }


def criterion_subset_mean(pooled_rse: Any, pooled_mean_rse: Any, reason: str | None = None) -> dict[str, Any]:
    """C4: ``RSE_S(AE) <= 0.8 * RSE_S(m)``, the right side in float64. At a zero subset-mean error the inequality
    is kept as written (passes iff the AE RSE is exactly 0; P5v2-D32). Undefined or non-finite fails."""
    factor = THRESHOLDS["fixed_subset4"]["subset_mean_factor"]
    if not _ok(pooled_mean_rse):
        row = compare(pooled_rse, "<=", math.nan, reason or "subset_mean_rse_undefined_or_non_finite")
        row["threshold"] = None
    else:
        row = compare(pooled_rse, "<=", factor * pooled_mean_rse, reason)
    row.update(factor=factor, pooled_subset_mean_rse=pooled_mean_rse if _ok(pooled_mean_rse) else None)
    return row


def gate(control: str, metrics: dict[str, Any], mean_metrics: dict[str, Any] | None) -> dict[str, Any]:
    """C2, C3 and (``fixed_subset4``) C4 of one predictor's metrics, with that control's frozen thresholds."""
    if control not in THRESHOLDS:
        raise ValueError(f"unknown P5-A control {control!r}")
    per = metrics["per_snapshot"]
    out: dict[str, Any] = {
        "C2": _per_snapshot(control, per, "rse", "<=", "rse_max"),
        "C3": _per_snapshot(control, per, "cosine", ">=", "cosine_min"),
    }
    gated = [m["rse"] for m in per] + [m["cosine"] for m in per]
    if control == "fixed_subset4":
        if mean_metrics is None:
            raise ValueError("fixed_subset4 needs the subset-mean predictor for C4")
        rse, mean_rse = metrics["pooled"]["rse"], mean_metrics["pooled"]["rse"]
        out["C4"] = criterion_subset_mean(rse, mean_rse, metrics["pooled"]["undefined"].get("rse"))
        gated += [rse, mean_rse]
    out["gated_values_defined_and_finite"] = all(_ok(v) for v in gated)
    return out


def required_criteria(control: str, *, with_c1: bool) -> list[str]:
    return (["C1"] if with_c1 else []) + ["C2", "C3"] + (["C4"] if control == "fixed_subset4" else [])


def c0_feasibility(
    control: str, ceiling: dict[str, Any], mean_metrics: dict[str, Any] | None
) -> dict[str, Any]:
    """C0: the Tanh-range ceiling must meet the control's C2, C3 (and C4) before the control is trained."""
    g = gate(control, ceiling, mean_metrics)
    g.pop("gated_values_defined_and_finite")
    required = required_criteria(control, with_c1=False)
    return {"criteria": g, "required": required, "pass": all(g[c]["pass"] for c in required)}


def ae_decision(
    control: str, recon_finite: bool, ae: dict[str, Any], mean_metrics: dict[str, Any] | None
) -> dict[str, Any]:
    """C1-C4 for the evaluated iteration-3,000 checkpoint; ``pass`` is the AND of the required criteria."""
    g = gate(control, ae, mean_metrics)
    defined = g.pop("gated_values_defined_and_finite")
    g["C1"] = {
        "reconstructions_finite": recon_finite,
        "gated_values_defined_and_finite": defined,
        "pass": bool(recon_finite and defined),
    }
    required = required_criteria(control, with_c1=True)
    return {"criteria": g, "required": required, "pass": all(g[c]["pass"] for c in required)}
