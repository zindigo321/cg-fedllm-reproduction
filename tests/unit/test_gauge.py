"""Phase-4 F2-F4: gauge diagnostics, balanced canonical factors and the effective increment (no dense d x d)."""

from __future__ import annotations

import math

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from cg_fedllm.compression.diagnostics import lowrank_inner
from cg_fedllm.compression.gauge import (
    balanced_factors,
    effective_delta,
    effective_update_norm,
    gauge_diagnostics,
    spectrum,
)
from cg_fedllm.models.adapter import AdapterState

S = 2.0  # lora_alpha / r = 16 / 8


def _rand(d_out=64, d_in=48, r=8, seed=0, b_scale=1e-2):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(r, d_in, generator=g, dtype=torch.float64) * 0.05
    b = torch.randn(d_out, r, generator=g, dtype=torch.float64) * b_scale
    return a, b


def _gauge(r=8, seed=1):
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(r, r, generator=g, dtype=torch.float64))
    d = torch.diag(torch.linspace(0.5, 2.0, r, dtype=torch.float64))
    return q @ d  # well conditioned (condition number 4)


def test_factor_norms_are_not_gauge_invariant_but_the_product_is():
    a, b = _rand()
    q = _gauge()
    a2, b2 = torch.linalg.solve(q, a), b @ q
    assert torch.allclose(b2 @ a2, b @ a, atol=1e-14, rtol=0)
    assert (
        abs(float(a2.norm()) / float(a.norm()) - 1) > 0.05
        and abs(float(b2.norm()) / float(b.norm()) - 1) > 0.05
    )
    assert torch.allclose(spectrum(a2, b2, S), spectrum(a, b, S), rtol=1e-10, atol=0)
    dense = torch.linalg.svdvals(S * b @ a)[:8]
    assert torch.allclose(spectrum(a, b, S), dense, rtol=1e-10, atol=1e-14)


def test_balanced_factors_preserve_the_product_are_gauge_invariant_and_deterministic():
    a, b = _rand()
    c = balanced_factors(a, b, S)
    assert torch.allclose(c["B"] @ c["A"], S * b @ a, atol=1e-13, rtol=0)
    sv = spectrum(a, b, S)
    lhs = float((c["A"] ** 2).sum() + (c["B"] ** 2).sum())
    assert math.isclose(lhs, 2 * float(sv.sum()), rel_tol=1e-10)  # = 2 * nuclear norm
    for seed in (1, 2, 3):  # any gauge gives the same canonical factors (distinct singular values)
        q = _gauge(seed=seed)
        c2 = balanced_factors(torch.linalg.solve(q, a), b @ q, S)
        assert torch.allclose(c2["A"], c["A"], atol=1e-10, rtol=0) and torch.allclose(
            c2["B"], c["B"], atol=1e-10, rtol=0
        )
        assert (
            lhs <= float((torch.linalg.solve(q, a) ** 2).sum() + ((b @ q) ** 2).sum()) * S + 1e-12
        )  # balanced is minimal
    c3 = balanced_factors(a, b, S)
    assert torch.equal(c3["A"], c["A"]) and torch.equal(c3["B"], c["B"])
    for j in range(8):  # sign convention: largest-magnitude entry of every left vector is positive
        col = c["B"][:, j]
        assert col[int(torch.argmax(col.abs()))] > 0
    assert c["tied"] == 0 and c["numerically_zero"] == 0


def test_rank_deficient_and_near_zero_components():
    a, _ = _rand()
    zero = balanced_factors(a, torch.zeros(64, 8, dtype=torch.float64), S)
    assert (
        zero["numerically_zero"] == 8
        and float(zero["A"].abs().max()) == 0.0
        and float(zero["B"].abs().max()) == 0.0
    )
    g = torch.Generator().manual_seed(5)
    u = torch.randn(64, 1, generator=g, dtype=torch.float64)
    b1 = u @ torch.randn(1, 8, generator=g, dtype=torch.float64)  # rank-1 B
    c = balanced_factors(a, b1, S)
    assert c["numerically_zero"] == 7
    assert torch.allclose(c["B"] @ c["A"], S * b1 @ a, atol=1e-12, rtol=0)
    b_tiny = b1 + 1e-15 * torch.randn(
        64, 8, generator=g, dtype=torch.float64
    )  # near-zero extra singular values
    ct = balanced_factors(a, b_tiny, S)
    assert ct["numerically_zero"] == 7 and torch.allclose(
        ct["B"] @ ct["A"], S * b_tiny @ a, atol=1e-12, rtol=0
    )


class _MaxNumel(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.max_numel = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for t in out if isinstance(out, (tuple, list)) else (out,):
            if isinstance(t, torch.Tensor):
                self.max_numel = max(self.max_numel, t.numel())
        return out


def test_qwen_geometry_without_dense_allocation():
    d, r = 2048, 8
    a_e, b_e = _rand(d, d, r, seed=7, b_scale=1e-3)
    a_s, b_s = _rand(d, d, r, seed=8, b_scale=1e-3)
    with _MaxNumel() as m:
        c = balanced_factors(a_e, b_e, S)
        e = effective_delta(a_e, b_e, a_s, b_s, S, rank=r)
    assert m.max_numel <= d * 2 * r  # never anything like d x d (4,194,304)
    assert (
        c["A"].shape == (r, d)
        and c["B"].shape == (d, r)
        and e["A"].shape == (r, d)
        and e["B"].shape == (d, r)
    )
    # B_c A_c = s B A, checked through r x r Gram matrices only
    err = (
        lowrank_inner(c["B"], c["A"], c["B"], c["A"])
        - 2 * lowrank_inner(c["B"], c["A"], S * b_e, a_e)
        + lowrank_inner(S * b_e, a_e, S * b_e, a_e)
    )
    assert abs(err) / lowrank_inner(S * b_e, a_e, S * b_e, a_e) < 1e-12  # float64 cancellation floor ~1e-16


def test_effective_delta_matches_a_dense_reference():
    a_e, b_e = _rand(seed=11)
    a_s, b_s = _rand(seed=12)
    e = effective_delta(a_e, b_e, a_s, b_s, S, rank=8)
    dm = S * (b_e @ a_e - b_s @ a_s)
    sv = torch.linalg.svdvals(dm)
    assert e["exact_rank"] == 16 and torch.allclose(e["singular_values"], sv[:16], rtol=1e-9, atol=1e-15)
    retained = float((sv[:8] ** 2).sum() / (sv**2).sum())
    assert math.isclose(e["retained_energy"], retained, rel_tol=1e-10)
    u, s_, vh = torch.linalg.svd(dm)
    best = (u[:, :8] * s_[:8]) @ vh[:8]
    assert torch.allclose(
        e["B"] @ e["A"], best, atol=1e-12, rtol=0
    )  # balanced factors of the best rank-8 approximation
    cos = float((best * dm).sum() / (best.norm() * dm.norm()))
    assert math.isclose(e["rank_r_product_cosine"], cos, rel_tol=1e-10)
    assert math.isclose(e["truncation_rel_fro_error"], float((dm - best).norm() / dm.norm()), rel_tol=1e-9)
    # from a zero-B start (round 0) the increment is exactly rank <= r and the truncation is lossless
    e0 = effective_delta(a_e, b_e, a_s, torch.zeros_like(b_s), S, rank=8)
    assert e0["exact_rank"] == 8 and math.isclose(e0["retained_energy"], 1.0, rel_tol=1e-12)


def test_gauge_diagnostics_on_an_adapter_state():
    g = torch.Generator().manual_seed(3)
    t = {}
    for layer in range(2):
        for m in ("q_proj", "k_proj", "v_proj", "o_proj"):
            t[f"layers.{layer}.self_attn.{m}.lora_A.weight"] = torch.randn(8, 32, generator=g) * 0.05
            t[f"layers.{layer}.self_attn.{m}.lora_B.weight"] = torch.randn(32, 8, generator=g) * 0.01
    st = AdapterState(t)
    gd = gauge_diagnostics(st, S, keep_singular_values=True)
    dense_fro = sum(
        float(((S * t[k[: -len("lora_A.weight")] + "lora_B.weight"].double() @ t[k].double()) ** 2).sum())
        for k in t
        if "lora_A" in k
    )
    assert gd["modules"] == 8 and math.isclose(gd["M_fro_sq_total"], dense_fro, rel_tol=1e-10)
    assert math.isclose(gd["balanced_factor_sq_total"], 2 * gd["M_nuclear_total"], rel_tol=1e-15)
    assert math.isclose(gd["factor_sq_total"], st.l2_sq(), rel_tol=1e-12)
    assert all(1.0 <= m["stable_rank"] <= 8.0 + 1e-9 for m in gd["per_module"])
    upd = effective_update_norm(st, st, S)
    assert upd["M_update_fro"] == pytest.approx(0.0, abs=1e-12) and upd["A_update_fro"] == 0.0
