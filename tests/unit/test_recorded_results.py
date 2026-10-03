"""The committed Phase-2 evidence is complete, labelled and passes its gates (testing standard item 24 and the
recorded exit-gate evidence). Pure JSON checks: no model, no GPU, nothing downloaded."""

from __future__ import annotations

import json

import pytest

from cg_fedllm.config import PHASE2_RESULT_LABELS, PHASE3_RESULT_LABELS, RESULT_LABELS, canonical_result_label
from tests.conftest import REPO

RESULTS = REPO / "results" / "phase2"


def _load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def _labels(obj) -> list[str]:
    """Every value stored under a ``label`` / ``result_label`` key anywhere in a JSON document."""
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("label", "result_label") and isinstance(v, str):
                out.append(v)
            out += _labels(v)
    elif isinstance(obj, list):
        for v in obj:
            out += _labels(v)
    return out


def test_every_committed_result_stays_valid_under_the_current_label_schema():
    """Label-schema migration regression: committed Phase-2 records (and every later record) carry only labels
    that the current schema accepts; Phase-2 records carry only Phase-2-era labels."""
    files = sorted((REPO / "results").rglob("*.json"))
    assert files
    annotated = []
    for path in files:
        labels = _labels(json.loads(path.read_text(encoding="utf-8")))
        assert labels, f"{path} carries no result label"
        canonical = {canonical_result_label(v) for v in labels}  # raises on anything outside the schema
        if path.is_relative_to(RESULTS):
            assert canonical <= set(PHASE2_RESULT_LABELS), path
        else:
            assert set(labels) <= set(RESULT_LABELS), path  # post-Phase-2 writers emit exact labels only
        if path.is_relative_to(REPO / "results" / "phase3"):
            assert set(labels) <= set(PHASE3_RESULT_LABELS), (
                path
            )  # Phase-3 evidence stays valid under later schemas
        annotated += [(path.name, v) for v in labels if v not in RESULT_LABELS]
    # the one historical annotated label (Phase 2), accepted by the read-side migration rule
    assert annotated == [
        ("tgap_snapshot_stats.json", "DERIVED (from PHASE2-SMOKE snapshots, smoke_final @ 25a3f7f)")
    ]
    with pytest.raises(ValueError):
        canonical_result_label("LOCAL-EVALUATOR-VALIDATION")
    with pytest.raises(ValueError):
        canonical_result_label("PHASE2-SMOKE-ISH (annotated)")


def test_mmlu_scorer_agrees_with_lm_eval_within_half_a_point():
    x = _load("mmlu_lmeval_crosscheck.json")
    assert x["label"] in RESULT_LABELS
    assert x["questions_compared"] == 14042 and x["all_questions_covered_once"] is True
    assert sum(part["num_questions"] for part in x["lm_eval_parts"]) == 14042
    assert all(part["limit"] is None for part in x["lm_eval_parts"])
    assert abs(100 * (x["ours_overall"] - x["lm_eval_overall"]) - x["delta_pp"]) < 1e-9
    assert abs(x["delta_pp"]) <= 0.5 and x["pass"] is True


def test_evaluator_validation_is_complete():
    s = _load("eval_validation_qwen15_0p5b.json")
    assert s["label"] == "PHASE2-SMOKE"
    sizes = {"mmlu/test": 14042, "ceval/val": 1346, "ceval/test": 12342}
    assert set(s["benchmarks"]) == set(sizes)
    for key, n in sizes.items():
        b = s["benchmarks"][key]
        assert b["aggregates"]["complete"] is True and b["aggregates"]["num_questions"] == n
        assert b["scoring"]["truncated_requests"] == 0 and b["scoring"]["reduced_shot_requests"] == 0


def test_gpu_smoke_summary_gates():
    s = _load("smoke/smoke_summary.json")
    assert s["label"] == "PHASE2-SMOKE"
    assert all(c["pass"] for c in s["checks"].values())
    assert s["checks"]["identity_state_vs_lora_ft"]["bitwise_equal"] is True
    assert s["checks"]["resume_vs_uninterrupted"]["bitwise_equal"] is True
    assert (
        s["faf_autoencoder"]["uplink_logical_bytes_total"] * 64
        == s["faf_identity"]["uplink_logical_bytes_total"]
    )
    assert s["total_time_s"] <= 20 * 60


def test_gpu_microbenchmarks_are_labelled_and_cover_the_candidates():
    b = _load("gpu_microbench.json")
    assert b["label"] == "LOCAL-MICROBENCH" and all(r["label"] == "LOCAL-MICROBENCH" for r in b["results"])
    assert b["vram_guard"]["allocator_cap_bytes"] < b["vram_guard"]["device_total_bytes"]
    assert {"Qwen/Qwen1.5-0.5B", "Qwen/Qwen1.5-1.8B"} <= {r["model"] for r in b["results"]}
    assert all(r["status"] in ("ok", "OOM") for r in b["results"])
