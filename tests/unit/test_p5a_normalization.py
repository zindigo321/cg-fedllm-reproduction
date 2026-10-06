"""P5-A v2 section-5 normalisation: exact max-abs, frozen scale, tolerance and range check (synthetic tensors only).

Oracles are written independently of the code under test: maxima from Python loops over element lists, scales from
literal float64 arithmetic, fp32 rounding through ``numpy.float32``.
"""

from __future__ import annotations

import dataclasses
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from cg_fedllm.compression.normalization import Normalizer, fit_normalizer, train_abs_max
from cg_fedllm.phase5 import p5a
from cg_fedllm.phase5.p5a import P5AProtocolError
from cg_fedllm.phase5.p5a_inputs import (
    F6_EVIDENCE_COMMIT,
    F6_TRAIN_RMS,
    FROZEN,
    FROZEN_MAX,
    FROZEN_SCALE,
    frozen_normalizer,
    range_check,
    rel_close,
)

REPO = Path(__file__).resolve().parents[2]
F6_FILES = {
    "R2": "results/phase4/screen/f6_r2_balanced_effective_state.json",
    "R3": "results/phase4/screen/f6_r3_balanced_effective_delta_r8.json",
    "R4": "results/phase4/screen/f6_r4_mean_step_gradient.json",
}


def _py_max_abs(xs) -> float:
    """Independent oracle: the maximum over a plain Python list of every element's absolute value."""
    return max(abs(v) for x in xs for v in x.reshape(-1).tolist())


def _frozen_for(xs, **kw):
    m = _py_max_abs(xs)
    base = dataclasses.replace(FROZEN["R4"], n_train=len(xs), max_abs=m, scale=m / 0.95)
    return dataclasses.replace(base, **kw)


def test_frozen_values_match_the_protocol_table_and_committed_f6_evidence():
    assert FROZEN_MAX == {"R2": 0.032825905829668045, "R3": 0.014964050613343716, "R4": 0.0643031895160675}
    assert FROZEN_SCALE == {"R2": 0.034553585083861103, "R3": 0.015751632224572334, "R4": 0.06768756791165001}
    assert F6_EVIDENCE_COMMIT == "d3c4eb3d631e0280533f35eb652fb24f68870079"
    for rid, path in F6_FILES.items():
        try:
            raw = subprocess.run(["git", "show", f"{F6_EVIDENCE_COMMIT}:{path}"], cwd=REPO, capture_output=True,
                                 check=True).stdout  # fmt: skip
        except (OSError, subprocess.CalledProcessError):
            raw = (REPO / path).read_bytes()  # e.g. a shallow CI checkout: the committed file itself
        stats = json.loads(raw)["input_stats"]
        assert stats["train_max_abs"] == FROZEN_MAX[rid] and stats["train_rms"] == F6_TRAIN_RMS[rid]
        assert FROZEN_MAX[rid] / 0.95 == FROZEN_SCALE[rid]  # float64, bitwise
        assert float(np.float32(FROZEN_MAX[rid])) == FROZEN_MAX[rid]  # an fp32 maximum
        assert FROZEN[rid].max_abs == FROZEN_MAX[rid] and FROZEN[rid].scale == FROZEN_SCALE[rid]
    assert float("0.0345535850838611") == FROZEN_SCALE["R2"]  # the two R2 literals are one float64


def test_exact_max_covers_every_element_including_last_and_negative():
    xs = [torch.tensor([[[0.1, -0.2]]]), torch.tensor([[[0.3, 0.05]]]), torch.tensor([[[0.0, -0.7]]])]
    assert train_abs_max(xs, range(3)) == float(np.float32(0.7)) == _py_max_abs(xs)  # last element, negative
    big = torch.zeros(1, 4, 8)
    big[0, 3, 7] = 2.5  # last element of the last tensor
    assert train_abs_max([torch.ones(1, 4, 8), big], range(2)) == 2.5
    with pytest.raises(ValueError):
        train_abs_max(xs, [])
    nan = [torch.tensor([[[0.1, float("nan")]]])]
    assert math.isnan(train_abs_max(nan, [0]))  # a NaN is never skipped


def test_new_mode_is_distinct_and_historical_p99_9_mode_is_unchanged():
    g = torch.Generator().manual_seed(0)
    xs = [torch.randn(1, 8, 16, generator=g) for _ in range(3)]
    exact = fit_normalizer(xs, [0, 1, 2], "global_exact_maxabs_train")
    assert exact.mode == "global_exact_maxabs_train" and exact.scale_global == _py_max_abs(xs) / 0.95
    hist = fit_normalizer(xs, [0, 1, 2], "global_maxabs_train")
    vals = sorted(abs(v) for x in xs for v in x.reshape(-1).double().tolist())
    h = 0.999 * (len(vals) - 1)  # numpy "linear" written out
    q = vals[math.floor(h)] + (h - math.floor(h)) * (vals[math.floor(h) + 1] - vals[math.floor(h)])
    assert hist.scale_global == pytest.approx(q / 0.95, rel=1e-15) and hist.scale_global < exact.scale_global
    assert hist.fit["quantile"] == 0.999 and hist.fit["exact"] is True


def test_frozen_scalar_is_applied_and_the_tolerance_is_inclusive_on_both_sides():
    xs = [torch.tensor([[[0.001, -0.002, 0.004, 0.0005]]]), torch.tensor([[[0.003, 0.0, 0.0, 0.0]]])]
    m = _py_max_abs(xs)
    norm, chk = frozen_normalizer(_frozen_for(xs), xs)
    assert norm.mode == p5a.NORMALIZATION_MODE and norm.scale_global == m / 0.95 and chk["bitwise_equal"]
    # the frozen value is applied even when it differs (within tolerance) from the recomputed maximum
    for frozen_m in (m / (1 + 0.9e-6), m / (1 - 0.9e-6)):
        fz = _frozen_for(xs, max_abs=frozen_m, scale=frozen_m / 0.95)
        n2, c2 = frozen_normalizer(fz, xs)
        assert (
            n2.scale_global == frozen_m / 0.95 and not c2["bitwise_equal"] and c2["recomputed_max_abs"] == m
        )
    for frozen_m in (m / (1 + 1.1e-6), m / (1 - 1.1e-6)):
        with pytest.raises(P5AProtocolError, match="not replaced"):
            frozen_normalizer(_frozen_for(xs, max_abs=frozen_m, scale=frozen_m / 0.95), xs)
    # rel_close is inclusive at exactly tol * |b| and fails beyond it
    b = 1.0
    assert rel_close(b + 1e-6 * b, b, 1e-6) and not rel_close(math.nextafter(b + 1e-6, 2), b, 1e-6)
    assert not rel_close(math.nan, b, 1e-6) and not rel_close(b, math.inf, 1e-6)


def test_application_is_fp32_division_by_float32_scale_without_clipping():
    x = torch.tensor([[[1e-3, -2e-3, 5e-2, 0.0]]])
    norm, _ = frozen_normalizer(_frozen_for([x]), [x])
    s32 = np.float32(_py_max_abs([x]) / 0.95)
    expected = (x.numpy() / s32).astype(np.float32)
    assert np.array_equal(norm.normalize(x).numpy(), expected)
    assert float(norm.normalize(x).abs().max()) <= 0.95 * (1 + 2e-6)
    assert np.array_equal(norm.denormalize(norm.normalize(x)).numpy(), (expected * s32).astype(np.float32))
    assert np.allclose(norm.denormalize(norm.normalize(x)).numpy(), x.numpy(), rtol=2e-7, atol=0)


def _scaled_norm(scale: float) -> Normalizer:
    return Normalizer("global_exact_maxabs_train", scale_global=scale)


def test_range_check_boundaries_on_both_sides():
    upper, lower = 0.95 * (1 + 2e-6), 0.95 * (1 - 2e-6)
    # with scale 1.0 the normalised values are the fp32 values themselves
    at = [torch.tensor([[[np.float32(0.95)]]])]
    assert range_check(_scaled_norm(1.0), at)["pass"]
    over = float(np.nextafter(np.float32(upper), np.float32(2)))  # first fp32 above the upper bound
    assert not range_check(_scaled_norm(1.0), [torch.tensor([[[over]]])])["pass"]
    below = float(np.nextafter(np.float32(lower), np.float32(0)))
    assert below < lower and not range_check(_scaled_norm(1.0), [torch.tensor([[[below]]])])["pass"]
    lo_ok = float(np.nextafter(np.float32(lower), np.float32(2)))
    assert lo_ok >= lower and range_check(_scaled_norm(1.0), [torch.tensor([[[lo_ok]]])])["pass"]
    assert over > upper and float(np.float32(0.95)) <= upper
    # every snapshot is bounded above; only the population maximum is bounded below
    two = [torch.tensor([[[0.95]]]), torch.tensor([[[0.1]]])]
    assert range_check(_scaled_norm(1.0), two)["pass"]
    assert range_check(_scaled_norm(1.0), two)["snapshots_over_upper"] == 0


def test_degenerate_and_non_finite_populations_stop():
    zeros = [torch.zeros(1, 2, 4)]
    with pytest.raises(P5AProtocolError, match="degenerate"):
        frozen_normalizer(dataclasses.replace(FROZEN["R4"], n_train=1), zeros)
    nan = [torch.tensor([[[0.1, float("nan")]]])]
    with pytest.raises(P5AProtocolError, match="non-finite"):
        frozen_normalizer(_frozen_for([torch.tensor([[[0.1, 0.2]]])], n_train=1), nan)
    with pytest.raises(P5AProtocolError, match="expected"):
        frozen_normalizer(_frozen_for(zeros), zeros + zeros)  # wrong population size
    with pytest.raises(ValueError):
        Normalizer("global_exact_maxabs_train", scale_global=0.0)
