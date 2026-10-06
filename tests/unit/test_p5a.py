"""P5-A v2 protocol tests on synthetic data only (docs/phase5_preregistration_v2.md section 15).

No test reads a real snapshot or gradient payload, loads a pretrained model, needs CUDA, or writes outside pytest's
temporary directory. Expected values are written out by hand from the preregistration, not produced by the code
under test.
"""

from __future__ import annotations

import itertools
import random

import pytest

from cg_fedllm.config import PHASE3_RESULT_LABELS, RESULT_LABELS, config_from_dict
from cg_fedllm.phase5 import p5a
from cg_fedllm.phase5.p5a import (
    DEMONSTRATED,
    FIT_FAIL,
    FIT_PASS,
    INCOMPLETE_INTERRUPTED,
    INCOMPLETE_NON_FINITE,
    INCOMPLETE_NOT_STARTED,
    INCOMPLETE_PROVENANCE,
    INCOMPLETE_PUBLICATION,
    INCOMPLETE_RESOURCE,
    INCONCLUSIVE,
    NOT_DEMONSTRATED,
    NOT_INFORMATIVE,
    P5AProtocolError,
)

# Phase-3 federated set: 5 clients per time index (client ids are the frozen R2/R3 training records of the
# preregistration only at the selected positions; the rest are filler with the same ordering properties)
SELECTED_R23_SINGLE = (7, 91)
SELECTED_R23_SUBSET = [(0, 2), (5, 32), (10, 43), (15, 84)]
SELECTED_R4_SINGLE = (0, 55)
SELECTED_R4_SUBSET = [(0, 2), (0, 26), (0, 75), (0, 86)]
R4_CLIENTS = [2, 26, 55, 75, 86]


def _r23_keys() -> list[tuple[int, int]]:
    """80 keys, t = 0..15, five ascending clients each. Position p = 5 t + j, so the preregistered keys sit at
    0 -> (0, 2), 26 -> (5, 32) [j=1], 39 -> (7, 91) [j=4], 53 -> (10, 43) [j=3], 79 -> (15, 84) [j=4]."""
    clients = {t: [10, 20, 30, 40, 50] for t in range(16)}
    clients[0] = [2, 20, 30, 40, 50]
    clients[5] = [10, 32, 40, 50, 60]
    clients[7] = [10, 20, 30, 40, 91]
    clients[10] = [10, 20, 30, 43, 60]
    clients[15] = [10, 20, 30, 40, 84]
    return [(t, c) for t in range(16) for c in clients[t]]


def _records(keys):
    return [{"time_index": t, "client_id": c} for t, c in keys]


def test_selection_positions_follow_the_written_rule():
    # section 6, worked by hand: N = 80 -> floor(79/2) = 39; round_half_up(j*79/3) = 0, 26, 53, 79
    assert p5a.control_positions("single_snapshot", 80) == [39]
    assert p5a.control_positions("fixed_subset4", 80) == [0, 26, 53, 79]
    # N = 5 -> floor(4/2) = 2; j*4/3 = 0, 1.333, 2.667, 4 -> 0, 1, 3, 4
    assert p5a.control_positions("single_snapshot", 5) == [2]
    assert p5a.control_positions("fixed_subset4", 5) == [0, 1, 3, 4]
    # rounding convention: half values round up (j*(N-1)/3 = 1.5 for N = ... none); check the helper directly
    assert p5a.round_half_up_ratio(3, 2) == 2 and p5a.round_half_up_ratio(5, 2) == 3
    assert p5a.round_half_up_ratio(1, 3) == 0 and p5a.round_half_up_ratio(2, 3) == 1
    with pytest.raises(P5AProtocolError):
        p5a.control_positions("fixed_subset4", 3)  # positions 0, 1, 1, 2 are not distinct
    with pytest.raises(ValueError):
        p5a.control_positions("first_k", 80)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_selection_identities_are_input_order_invariant(seed):
    r23 = _records(_r23_keys())
    r4 = _records([(0, c) for c in R4_CLIENTS])
    random.Random(seed).shuffle(r23)
    random.Random(seed).shuffle(r4)
    e23, e4 = p5a.sort_records(r23), p5a.sort_records(r4)
    key = lambda rows: [(r["time_index"], r["client_id"]) for r in rows]  # noqa: E731
    assert key(p5a.select_control(e23, "single_snapshot")) == [SELECTED_R23_SINGLE]
    assert key(p5a.select_control(e23, "fixed_subset4")) == SELECTED_R23_SUBSET
    assert key(p5a.select_control(e4, "single_snapshot")) == [SELECTED_R4_SINGLE]
    assert key(p5a.select_control(e4, "fixed_subset4")) == SELECTED_R4_SUBSET
    assert [r["position"] for r in p5a.select_control(e4, "fixed_subset4")] == [0, 1, 3, 4]


def test_sorting_is_numeric_and_duplicate_keys_abort():
    rows = _records([(10, 1), (2, 30), (2, 4), (9, 100)])
    assert [(r["time_index"], r["client_id"]) for r in p5a.sort_records(rows)] == [
        (2, 4),
        (2, 30),
        (9, 100),
        (10, 1),
    ]
    with pytest.raises(P5AProtocolError, match="not unique"):
        p5a.sort_records(_records([(0, 2), (0, 26), (0, 2)]))
    # string-typed metadata sorts numerically too
    assert p5a.record_key({"time_index": "10", "client_id": "2"}) == (10, 2)


# section 11.1, written out independently of the implementation's tuple
STATUSES = (
    "FIT PASS",
    "FIT FAIL",
    "NOT INFORMATIVE (TANH RANGE)",
    "INCOMPLETE (PROVENANCE)",
    "INCOMPLETE (RESOURCE STOP)",
    "INCOMPLETE (NON-FINITE LOSS)",
    "INCOMPLETE (INTERRUPTED)",
    "INCOMPLETE (PUBLICATION)",
    "INCOMPLETE (NOT STARTED)",
)


def _expected_outcome(single: str, subset: str) -> str:
    """Section 11.2 table, written out independently of the implementation."""
    if single == "FIT PASS" and subset == "FIT PASS":
        return "CAPACITY FIT DEMONSTRATED"
    if "FIT FAIL" in (single, subset):
        return "CAPACITY FIT NOT DEMONSTRATED"
    return "INCONCLUSIVE"


def test_status_set_is_the_nine_v2_statuses():
    assert set(p5a.CONTROL_STATUSES) == set(STATUSES) and len(p5a.CONTROL_STATUSES) == 9


@pytest.mark.parametrize(("single", "subset"), list(itertools.product(STATUSES, STATUSES)))
def test_section11_aggregation_is_exhaustive(single, subset):
    assert p5a.representation_outcome(single, subset) == _expected_outcome(single, subset)


def test_aggregation_examples_and_structured_outcome_records():
    assert p5a.representation_outcome(FIT_PASS, FIT_PASS) == DEMONSTRATED
    assert p5a.representation_outcome(FIT_PASS, INCOMPLETE_PUBLICATION) == INCONCLUSIVE
    assert p5a.representation_outcome(INCOMPLETE_PROVENANCE, FIT_FAIL) == NOT_DEMONSTRATED
    assert p5a.representation_outcome(NOT_INFORMATIVE, NOT_INFORMATIVE) == INCONCLUSIVE
    for other in (NOT_INFORMATIVE, INCOMPLETE_PROVENANCE, INCOMPLETE_NON_FINITE, INCOMPLETE_INTERRUPTED,
                  INCOMPLETE_RESOURCE, INCOMPLETE_PUBLICATION, INCOMPLETE_NOT_STARTED):  # fmt: skip
        assert p5a.representation_outcome(FIT_PASS, other) == INCONCLUSIVE
        assert p5a.representation_outcome(other, FIT_FAIL) == NOT_DEMONSTRATED
    with pytest.raises(ValueError):
        p5a.representation_outcome("AE TRAINING-FIT FAILURE", FIT_PASS)
    rec = p5a.outcome_record("R4", {"single_snapshot": FIT_PASS, "fixed_subset4": FIT_PASS})
    assert rec["designation"] == "secondary" and rec["outcome"] == DEMONSTRATED
    assert rec["qualifier"].startswith(
        "under the P5-A v2 protocol (fixed ResNet-3, exact training-population max-abs"
    )
    assert rec["standards"]["single_snapshot"] == "memorisation (RSE <= 0.01, cos >= 0.99)"
    assert "stability across seeds" in rec["does_not_establish"]
    assert rec["authorises"].startswith("only drafting a separate")
    assert (
        p5a.outcome_record("R2", {"single_snapshot": FIT_FAIL, "fixed_subset4": FIT_PASS})["authorises"]
        == "nothing"
    )
    assert (
        p5a.outcome_record("R2", {"single_snapshot": FIT_FAIL, "fixed_subset4": FIT_PASS})["designation"]
        == "primary"
    )
    # limitation statements may name the words a v1 filter refused: no blanket word filter (P5v2-D37)
    assert "incompressible" in p5a.outcome_record("R3", {"single_snapshot": FIT_FAIL,
                                                          "fixed_subset4": FIT_FAIL})["meaning"]  # fmt: skip
    assert not hasattr(p5a, "check_wording") and not hasattr(p5a, "FORBIDDEN_WORDS")


def test_plan_order_designation_and_frozen_values():
    assert p5a.PLAN == (("R2", "single_snapshot"), ("R2", "fixed_subset4"), ("R3", "single_snapshot"),
                        ("R3", "fixed_subset4"), ("R4", "single_snapshot"), ("R4", "fixed_subset4"))  # fmt: skip
    assert p5a.DESIGNATION == {"R2": "primary", "R3": "primary", "R4": "secondary"}
    assert p5a.THRESHOLDS == {"single_snapshot": {"rse_max": 0.01, "cosine_min": 0.99},
                              "fixed_subset4": {"rse_max": 0.5, "cosine_min": 0.9, "subset_mean_factor": 0.8}}  # fmt: skip
    assert p5a.P5A_RUN_NAME == "phase5_p5a_v2_seed1" and p5a.P5A_V1_RUN_NAME == "phase5_p5a_seed1"
    assert p5a.NORMALIZATION_MODE == "global_exact_maxabs_train"
    assert (p5a.MAX_TRAININGS, p5a.TRAINING_LIMIT_S, p5a.AGGREGATE_LIMIT_S) == (6, 1_800.0, 10_800.0)
    assert (p5a.MAX_INVOCATIONS, p5a.MAX_PREFLIGHTS, p5a.SCALE_REL_TOL, p5a.RANGE_REL_ALLOWANCE) == (
        3,
        3,
        1e-6,
        2e-6,
    )


def test_protocol_provenance_is_the_full_v2_commit():
    import subprocess
    from pathlib import Path

    assert p5a.P5A_PROTOCOL_COMMIT == "dd4bf3f4348fd3b2b1b2c4ad23e7204fbef8369c"
    assert p5a.P5A_PROTOCOL_PATH == "docs/phase5_preregistration_v2.md"
    assert p5a.P5A_PROTOCOL_BLOB == "e80d21aeda653ab6d1339d34f7b88118588855df"
    assert p5a.P5A_V1_COMMIT == "52a0dd4e7bb8521752ac80ff8f77072dc0e0761c"
    repo = Path(__file__).resolve().parents[2]
    assert p5a.P5A_ADDENDUM_COMMIT == "33e51fd63c6cd0679e76e12380ae5d418158d698"
    assert p5a.P5A_ADDENDUM_PATH == "docs/phase5_preregistration_v2_1.md"
    assert p5a.P5A_ADDENDUM_BLOB == "28e05f9ee33ed13386d346fce6e47e9108c937d7"
    ident = p5a.protocol_identity()
    assert (
        ident["commit"] == p5a.P5A_PROTOCOL_COMMIT and ident["addendum"]["commit"] == p5a.P5A_ADDENDUM_COMMIT
    )
    try:
        blobs = [subprocess.run(["git", "hash-object", path], cwd=repo, capture_output=True, check=True,
                                text=True).stdout.strip() for path in (p5a.P5A_PROTOCOL_PATH, p5a.P5A_ADDENDUM_PATH)]  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git unavailable")
    assert blobs == [p5a.P5A_PROTOCOL_BLOB, p5a.P5A_ADDENDUM_BLOB]  # the working copies are the frozen texts


def test_phase5_label_extends_the_schema_without_dropping_history():
    assert RESULT_LABELS[-1] == "PHASE5-DIAGNOSTIC" and p5a.P5A_LABEL == "PHASE5-DIAGNOSTIC"
    assert set(PHASE3_RESULT_LABELS) <= set(RESULT_LABELS)
    for old in ("PHASE4-FORENSIC", "PHASE4-BASELINE", "PHASE4-DIAGNOSTIC", "PHASE3-DIAGNOSTIC", "DERIVED"):
        assert old in RESULT_LABELS
    assert (
        config_from_dict({"run": {"name": "x", "result_label": "PHASE5-DIAGNOSTIC"}}).run.result_label
        == p5a.P5A_LABEL
    )
    assert "EXPLORATORY-DIAGNOSTIC" not in RESULT_LABELS
