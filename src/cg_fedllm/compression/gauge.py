"""Gauge-aware LoRA algebra without dense d x d products (Phase 4, stages F2-F4).

A LoRA module contributes ``M = s * B @ A`` with ``s = lora_alpha / r``, A: r x d_in, B: d_out x r. The factors
are not identifiable: for any invertible r x r Q, ``B' = B Q`` and ``A' = Q^-1 A`` give the same M, so factor
norms (and any statement such as "||G||^2 = 14.29") depend on the gauge, while the singular values of M do not.

Everything here works on thin QR factors and an r x r (or 2r x 2r) core, in float64 on the CPU:

  B = Q_B R_B,  A^T = Q_A R_A      =>  M = Q_B (s R_B R_A^T) Q_A^T = Q_B C Q_A^T,  C = U S V^T
  singular values of M = S;  ||M||_F^2 = sum S^2;  ||M||_* = sum S;  stable rank = ||M||_F^2 / S_max^2

Balanced canonical factors (F3; a PHASE4-FORENSIC representation, not CG-FedLLM's):
  B_c = Q_B U sqrt(S),  A_c = sqrt(S) V^T Q_A^T,  so B_c A_c = s B A and ||A_c||_F^2 + ||B_c||_F^2 = 2 ||M||_*
  (the minimum of ||A||^2 + ||B||^2 over all factorisations of M).
Determinism: component k is sign-normalised so that the largest-magnitude entry of its left vector Q_B U[:, k] is
positive (ties: lowest index). Components with S_k <= tol * S_max are numerically zero and get zero factors.
For repeated singular values (relative gap <= TIE_RTOL) the singular subspace is not unique: the factors are then
LAPACK-deterministic on a given platform but not canonical; such ties are counted and reported.

Effective increment (F4): ``dM = s (B_e A_e - B_s A_s) = s [B_e, -B_s] [A_e; A_s]`` has rank <= 2r. Its exact
spectrum comes from the 2r x 2r core; the best rank-r approximation (Eckart-Young) is truncated from it, and its
retained energy, relative Frobenius truncation error and product cosine (= sqrt(retained energy)) are exact.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from cg_fedllm.models.adapter import AdapterState, parse_key

ZERO_RTOL = 1e-12
TIE_RTOL = 1e-6


def _f64(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to("cpu", torch.float64)


def _core(
    left: torch.Tensor, right_t: torch.Tensor, s: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``M = s * left @ right_t.T`` with left: d_out x k, right_t: d_in x k -> (Q_L, Q_R, C) where C is k x k."""
    q_l, r_l = torch.linalg.qr(left, mode="reduced")
    q_r, r_r = torch.linalg.qr(right_t, mode="reduced")
    return q_l, q_r, s * (r_l @ r_r.T)


def spectrum(A: torch.Tensor, B: torch.Tensor, s: float) -> torch.Tensor:
    """Singular values (descending) of M = s B A."""
    _, _, c = _core(_f64(B), _f64(A).T, s)
    return torch.linalg.svdvals(c)


def _canonical_from_core(
    q_l: torch.Tensor, q_r: torch.Tensor, c: torch.Tensor, rank: int | None = None
) -> dict[str, Any]:
    u, sv, vh = torch.linalg.svd(c)
    k = sv.numel() if rank is None else min(rank, sv.numel())
    smax = float(sv[0]) if sv.numel() else 0.0
    left = q_l @ u[:, :k]  # d_out x k, orthonormal columns
    right = vh[:k] @ q_r.T  # k x d_in, orthonormal rows
    zero = sv[:k] <= ZERO_RTOL * smax if smax > 0 else torch.ones(k, dtype=torch.bool)
    for j in range(k):
        col = left[:, j]
        idx = int(torch.argmax(col.abs()))  # first index of the maximum magnitude
        if col[idx] < 0:
            left[:, j] = -col
            right[j] = -right[j]
    root = torch.sqrt(torch.clamp(sv[:k], min=0.0))
    root = torch.where(zero, torch.zeros_like(root), root)
    b_c = left * root
    a_c = root[:, None] * right
    ties = 0
    if k > 1 and smax > 0:  # adjacent (non-zero) singular values closer than TIE_RTOL * S_max
        gaps = (sv[:k][:-1] - sv[:k][1:]) <= TIE_RTOL * smax
        ties = int(gaps[~zero[1:]].sum())
    return {"A": a_c, "B": b_c, "singular_values": sv, "numerically_zero": int(zero.sum()), "tied": ties}


def balanced_factors(A: torch.Tensor, B: torch.Tensor, s: float) -> dict[str, Any]:
    """Balanced canonical rank-r factors of M = s B A (no d x d allocation)."""
    q_l, q_r, c = _core(_f64(B), _f64(A).T, s)
    return _canonical_from_core(q_l, q_r, c)


def effective_delta(
    A_end: torch.Tensor,
    B_end: torch.Tensor,
    A_start: torch.Tensor,
    B_start: torch.Tensor,
    s: float,
    rank: int,
) -> dict[str, Any]:
    """Exact spectrum of dM = s (B_end A_end - B_start A_start) (rank <= 2r) and its balanced best rank-``rank`` factors."""
    left = torch.cat([_f64(B_end), -_f64(B_start)], dim=1)  # d_out x 2r
    right_t = torch.cat([_f64(A_end), _f64(A_start)], dim=0).T  # d_in x 2r
    q_l, q_r, c = _core(left, right_t, s)
    full = torch.linalg.svdvals(c)
    smax = float(full[0]) if full.numel() else 0.0
    exact_rank = int((full > ZERO_RTOL * smax).sum()) if smax > 0 else 0
    energy = float((full**2).sum())
    kept = float((full[:rank] ** 2).sum())
    retained = kept / energy if energy > 0 else math.nan
    canon = _canonical_from_core(q_l, q_r, c, rank=rank)
    return {
        "A": canon["A"],
        "B": canon["B"],
        "singular_values": full,
        "exact_rank": exact_rank,
        "energy": energy,
        "retained_energy": retained,
        "truncation_rel_fro_error": math.sqrt(max(0.0, 1.0 - retained)) if energy > 0 else math.nan,
        "rank_r_product_cosine": math.sqrt(retained) if energy > 0 else math.nan,
        "tied": canon["tied"],
    }


def module_pairs(state: AdapterState) -> list[tuple[str, str]]:
    out = []
    for k in state.keys():
        if parse_key(k)[3] == "A":
            b = k[: -len("lora_A.weight")] + "lora_B.weight"
            if b not in state.tensors:
                raise ValueError(f"{k}: no matching B factor")
            out.append((k, b))
    return out


def gauge_diagnostics(state: AdapterState, s: float, *, keep_singular_values: bool = False) -> dict[str, Any]:
    """Per-module and total gauge-dependent (factor norms) and gauge-invariant (effective M) statistics."""
    mods = []
    tot = {"A_sq": 0.0, "B_sq": 0.0, "M_fro_sq": 0.0, "M_nuclear": 0.0}
    for ka, kb in module_pairs(state):
        a, b = _f64(state.tensors[ka]), _f64(state.tensors[kb])
        sv = spectrum(a, b, s)
        a_sq, b_sq = float((a * a).sum()), float((b * b).sum())
        fro_sq, nuc = float((sv**2).sum()), float(sv.sum())
        smax = float(sv[0]) if sv.numel() else 0.0
        row = {
            "module": ka[: -len(".lora_A.weight")],
            "A_fro": math.sqrt(a_sq),
            "B_fro": math.sqrt(b_sq),
            "M_fro": math.sqrt(fro_sq),
            "M_nuclear": nuc,
            "stable_rank": fro_sq / smax**2 if smax > 0 else None,
            "sigma_max": smax,
        }
        if keep_singular_values:
            row["singular_values"] = [float(x) for x in sv]
        mods.append(row)
        tot["A_sq"] += a_sq
        tot["B_sq"] += b_sq
        tot["M_fro_sq"] += fro_sq
        tot["M_nuclear"] += nuc
    srs = [m["stable_rank"] for m in mods if m["stable_rank"] is not None]
    return {
        "lora_scaling": s,
        "modules": len(mods),
        "factor_sq_total": tot["A_sq"] + tot["B_sq"],
        "A_sq_total": tot["A_sq"],
        "B_sq_total": tot["B_sq"],
        "M_fro_sq_total": tot["M_fro_sq"],
        "M_nuclear_total": tot["M_nuclear"],
        "balanced_factor_sq_total": 2.0 * tot["M_nuclear"],
        "stable_rank": {"min": min(srs), "median": sorted(srs)[len(srs) // 2], "max": max(srs)}
        if srs
        else None,
        "per_module": mods,
    }


def effective_update_norm(start: AdapterState, end: AdapterState, s: float) -> dict[str, float]:
    """||s (B_e A_e - B_s A_s)||_F summed over modules (exact, low-rank), plus factor-space update norms."""
    sq = 0.0
    for ka, kb in module_pairs(end):
        sv = effective_delta(
            end.tensors[ka], end.tensors[kb], start.tensors[ka], start.tensors[kb], s, rank=1
        )["singular_values"]
        sq += float((sv**2).sum())
    d = end.sub(start)
    return {
        "M_update_fro": math.sqrt(sq),
        "A_update_fro": math.sqrt(d.l2_sq("A")),
        "B_update_fro": math.sqrt(d.l2_sq("B")),
    }
