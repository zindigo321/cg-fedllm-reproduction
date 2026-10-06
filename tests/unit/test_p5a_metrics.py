"""P5-A v2 section-9 metrics and C0-C4 criteria on constructed tensors (expected values worked out by hand)."""

from __future__ import annotations

import json
import math

import pytest
import torch

from cg_fedllm.compression.normalization import Normalizer
from cg_fedllm.phase5.p5a_artifacts import json_bytes
from cg_fedllm.phase5.p5a_metrics import (
    ae_decision,
    c0_feasibility,
    compare,
    criterion_subset_mean,
    gate,
    predictor_metrics,
    subset_mean,
    tanh_range_ceiling,
)


def t(*v):
    return torch.tensor([[list(v)]], dtype=torch.float32)  # [1, 1, n]


def _m(rse, cos, undefined=None):
    """A per-snapshot metric row with given gated values (other fields irrelevant to C2/C3)."""
    return {"rse": rse, "cosine": cos, "undefined": undefined or {}}


def _metrics(rows, pooled_rse=0.0):
    return {"per_snapshot": rows, "pooled": {"rse": pooled_rse, "undefined": {}}}


def test_per_snapshot_and_pooled_values_by_hand():
    x1, p1 = t(3.0, 4.0), t(3.0, 0.0)  # sse 16, sig 25, hat 9, dot 9
    x2, p2 = t(1.0, 0.0), t(2.0, 0.0)  # sse 1, sig 1, hat 4, dot 2
    m = predictor_metrics([x1, x2], [p1, p2])
    a, b = m["per_snapshot"]
    assert (a["sse"], a["sig"], a["hat"], a["dot"]) == (16.0, 25.0, 9.0, 9.0)
    assert a["mse"] == 8.0 and a["zero_predictor_mse"] == 12.5
    assert a["rse"] == 16 / 25 and a["cosine"] == 9 / math.sqrt(25 * 9) and a["norm_ratio"] == 3 / 5
    assert b["rse"] == 1.0 and b["cosine"] == 2 / math.sqrt(1 * 4) and b["norm_ratio"] == 2.0
    pooled = m["pooled"]
    # pooled RSE is SSE / SIG = 17 / 26, NOT the mean of 0.64 and 1.0
    assert pooled["rse"] == 17 / 26 and pooled["rse"] != (16 / 25 + 1.0) / 2
    assert pooled["cosine"] == 11 / math.sqrt(26 * 13)
    assert pooled["mse"] == 17 / 4 and pooled["zero_predictor_mse"] == 26 / 4
    assert pooled["norm_ratio"] == math.sqrt(13 / 26) and pooled["undefined"] == {}


def test_float64_accumulation_is_used():
    x = torch.full((1, 1, 4), 1e-4, dtype=torch.float32)
    p = x * (1 + 1e-7)  # fp32 rounding of the difference would be lost in an fp32 sum of squares
    m = predictor_metrics([x], [p])["per_snapshot"][0]
    a, b = x.reshape(-1).double().tolist(), p.reshape(-1).double().tolist()
    assert m["sse"] == math.fsum((q - r) ** 2 for q, r in zip(b, a)) and m["sse"] > 0


def test_undefined_and_non_finite_values_are_null_with_reasons_and_json_strict():
    zero_sig = predictor_metrics([t(0.0, 0.0)], [t(1.0, 0.0)])["per_snapshot"][0]
    assert zero_sig["rse"] is None and zero_sig["cosine"] is None and zero_sig["norm_ratio"] is None
    assert zero_sig["undefined"] == {"rse": "sig_zero", "cosine": "sig_zero", "norm_ratio": "sig_zero"}
    zero_hat = predictor_metrics([t(1.0, 2.0)], [t(0.0, 0.0)])["per_snapshot"][0]
    assert (
        zero_hat["rse"] == 1.0
        and zero_hat["cosine"] is None
        and zero_hat["undefined"] == {"cosine": "hat_zero"}
    )
    for bad, reason in ((float("nan"), "non_finite:nan"), (float("inf"), "non_finite:+inf")):
        m = predictor_metrics([t(1.0, 2.0)], [t(bad, 0.0)])
        assert m["per_snapshot"][0]["rse"] is None and m["per_snapshot"][0]["undefined"]["rse"] == reason
        assert m["pooled"]["rse"] is None and m["pooled"]["undefined"]["rse"] == reason  # kept in the pool
    neg = predictor_metrics([t(1.0, 0.0)], [t(-math.inf, 0.0)])["per_snapshot"][0]
    assert neg["undefined"]["dot"] == "non_finite:-inf"
    text = json_bytes(
        m
    ).decode()  # strict JSON: no NaN/Infinity tokens, no "nan"/"inf" strings in metric fields
    assert "NaN" not in text and "Infinity" not in text and '"nan"' not in text
    assert json.loads(text)["pooled"]["rse"] is None


@pytest.mark.parametrize(
    ("control", "rse_max", "cos_min"), [("single_snapshot", 0.01, 0.99), ("fixed_subset4", 0.50, 0.90)]
)
def test_c2_c3_inclusive_equality_and_adjacent_float64_values(control, rse_max, cos_min):
    def g(rse, cos):
        rows = [_m(rse, cos)] * (1 if control == "single_snapshot" else 4)
        return gate(control, _metrics(rows), _metrics(rows, 1.0) if control == "fixed_subset4" else None)

    assert g(rse_max, 1.0)["C2"]["pass"] and g(math.nextafter(rse_max, 0), 1.0)["C2"]["pass"]
    assert not g(math.nextafter(rse_max, 1), 1.0)["C2"]["pass"]
    assert g(0.0, cos_min)["C3"]["pass"] and g(0.0, math.nextafter(cos_min, 1))["C3"]["pass"]
    assert not g(0.0, math.nextafter(cos_min, 0))["C3"]["pass"]
    row = g(rse_max, 1.0)["C2"]["per_snapshot"][0]
    assert row == {"value": rse_max, "operator": "<=", "threshold": rse_max, "margin": 0.0, "near_threshold": True,
                   "pass": True}  # fmt: skip
    assert not g(None, 1.0)["C2"]["pass"] and g(None, 1.0)["C2"]["per_snapshot"][0]["fail_reason"]


def test_strict_single_snapshot_standard_rejects_what_the_subset_standard_accepts():
    rows = [_m(0.02, 0.995)]  # passes 0.50 / 0.90 but not 0.01 / 0.99
    single = ae_decision("single_snapshot", True, _metrics(rows), None)
    assert not single["pass"] and not single["criteria"]["C2"]["pass"] and single["criteria"]["C3"]["pass"]
    assert single["required"] == ["C1", "C2", "C3"] and "C4" not in single["criteria"]
    assert gate("fixed_subset4", _metrics(rows * 4), _metrics(rows * 4, 1.0))["C2"]["pass"]


def test_c4_boundary_zero_subset_mean_error_and_undefined():
    assert criterion_subset_mean(0.4, 0.5)["pass"]  # 0.8 * 0.5 = 0.4: equality passes
    assert not criterion_subset_mean(math.nextafter(0.4, 1), 0.5)["pass"]
    assert criterion_subset_mean(math.nextafter(0.4, 0), 0.5)["pass"]
    # v2 (P5v2-D32): at a zero subset-mean error the inequality is kept: 0 <= 0.8 * 0 holds only for AE RSE 0
    assert criterion_subset_mean(0.0, 0.0)["pass"]
    assert not criterion_subset_mean(5e-324, 0.0)["pass"]
    assert not criterion_subset_mean(None, 0.5)["pass"] and not criterion_subset_mean(0.1, None)["pass"]
    assert criterion_subset_mean(0.1, None)["threshold"] is None


def test_every_snapshot_must_pass_even_when_pooled_values_pass():
    xs = [t(10.0, 0.0), t(10.0, 0.0), t(10.0, 0.0), t(1.0, 0.0)]
    preds = [t(10.0, 0.0), t(10.0, 0.0), t(10.0, 0.0), t(0.0, 1.0)]  # last: rse 2, cosine 0
    ae = predictor_metrics(xs, preds)
    assert ae["pooled"]["rse"] == 2 / 301 and ae["pooled"]["cosine"] > 0.99  # pooled would pass
    mean = predictor_metrics(xs, [subset_mean(xs).float()] * 4)
    d = ae_decision("fixed_subset4", True, ae, mean)
    assert not d["criteria"]["C2"]["pass"] and not d["criteria"]["C3"]["pass"] and not d["pass"]
    assert d["criteria"]["C4"]["pass"]  # pooled RSE is far below 0.8 x subset-mean RSE


def test_subset_mean_is_float64_and_from_the_selected_subset_only():
    m = subset_mean([t(1.0), t(2.0), t(4.0), t(5.0)])
    assert m.dtype == torch.float64 and float(m.flatten()[0]) == 3.0


def test_c1_requires_finite_reconstructions_and_defined_gated_values():
    x = [t(1.0, 2.0)]
    ae = predictor_metrics(x, [t(1.0, 2.0)])
    assert ae_decision("single_snapshot", True, ae, None)["pass"]
    assert not ae_decision("single_snapshot", False, ae, None)["pass"]
    zero = predictor_metrics(x, [t(0.0, 0.0)])  # cosine undefined: C1 and C3 fail, never a pass
    d = ae_decision("single_snapshot", True, zero, None)
    assert not d["criteria"]["C1"]["pass"] and not d["criteria"]["C3"]["pass"] and not d["pass"]
    with pytest.raises(ValueError):
        ae_decision("fixed_subset4", True, ae, None)


def test_tanh_range_ceiling_and_c0_use_the_control_thresholds():
    norm = Normalizer("global_exact_maxabs_train", scale_global=2.0)
    x = t(1.0, -3.0, 5.0)  # normalised 0.5, -1.5, 2.5 -> clamp 0.5, -1, 1 -> 1, -2, 2
    assert torch.equal(tanh_range_ceiling(x, norm), t(1.0, -2.0, 2.0))
    inside = [t(0.5, -1.0, 1.5)]  # |x / 2| <= 1: the ceiling is exact
    ok = predictor_metrics(inside, [tanh_range_ceiling(v, norm) for v in inside])
    assert c0_feasibility("single_snapshot", ok, None)["pass"]
    # ceiling (2, 0) vs truth (2.2, 0): rse 0.04 / 4.84 = 0.00826 -> passes 0.01; (2.3, 0): 0.09 / 5.29 = 0.017 -> fails
    near = [t(2.2, 0.0)]
    res = c0_feasibility(
        "single_snapshot", predictor_metrics(near, [tanh_range_ceiling(v, norm) for v in near]), None
    )
    assert res["pass"] and res["required"] == ["C2", "C3"]
    far = [t(2.3, 0.0)]
    res = c0_feasibility(
        "single_snapshot", predictor_metrics(far, [tanh_range_ceiling(v, norm) for v in far]), None
    )
    assert not res["pass"] and res["criteria"]["C2"]["per_snapshot"][0]["value"] == pytest.approx(0.09 / 5.29)


def test_near_threshold_flag_uses_four_ulps():
    assert compare(0.01, "<=", 0.01)["near_threshold"]
    v = 0.01
    for _ in range(4):
        v = math.nextafter(v, 1)
    assert (
        compare(v, "<=", 0.01)["near_threshold"]
        and not compare(math.nextafter(v, 1), "<=", 0.01)["near_threshold"]
    )
