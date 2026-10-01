"""The committed Phase-3A evidence is complete, labelled and internally consistent (pure JSON; no model, no GPU)."""

from __future__ import annotations

import json

import pytest

from cg_fedllm.tgap.viability import select_primary
from tests.conftest import REPO

R = REPO / "results" / "phase3"
A5 = ("none", "global_rms", "factor_rms")


def _load(rel: str) -> dict:
    return json.loads((R / rel).read_text(encoding="utf-8"))


def test_a3_calibration_records_the_rule_outcome():
    mb2, mb1 = _load("calibration/a3_calibration_mb2.json"), _load("calibration/a3_calibration_mb1.json")
    for c, mb in ((mb2, 2), (mb1, 1)):
        assert c["label"] == "PHASE3-DIAGNOSTIC" and c["config"]["micro_batch_size"] == mb
        assert c["config"]["dtype"] == "bfloat16" and c["config"]["quantization"] == "none" and c["config"]["gradient_checkpointing"] is True
        assert all(r["finite"] for r in c["timing"]["clients"]) and c["timing"]["worst_case_micro_batch"]["finite"]
        assert c["vram_guard"]["allocator_cap_bytes"] < c["vram_guard"]["device_total_bytes"]
    assert mb2["timing"]["decision"]["switch_to_micro_batch_1"] is True  # the pre-registered A3 rule triggered ...
    assert mb1["timing"]["decision"]["switch_to_micro_batch_1"] is False  # ... and the selected configuration passes it


def test_tgap_sets_are_complete():
    fed, loc = _load("tgap/federated_state_seed1_stats.json"), _load("tgap/local_state_seed1_stats.json")
    for s, mode in ((fed, "federated_pretrain"), (loc, "local_pretrain")):
        assert s["label"] == "DERIVED" and s["source_mode"] == mode
        assert s["num_snapshots"] == 100 and s["num_time_indices"] == 20 and len(s["manifest"]) == 100
        assert len(s["split"]["train"]) == 80 and s["split"]["val_time_indices"] == [16, 17, 18, 19]
        assert s["phi_shape"] == [1, 2048, 1536] and s["layout_id"] == "layer_major_qkvo_AtB"
    assert fed["unique_clients"] == 67 and fed["distinct_start_states"] == 20
    assert loc["unique_clients"] == 5 and loc["distinct_start_states"] == 96
    assert fed["collection"]["label"] == "PHASE3-DIAGNOSTIC" and loc["collection"]["label"] == "PHASE3-SENSITIVITY"


@pytest.mark.parametrize("name,label,source,rep,mode", [
    *[(f"a5_federated_state_{m}", "PHASE3-DIAGNOSTIC", "federated", "adapter_state", m) for m in A5],
    ("a8_local_state_none", "PHASE3-SENSITIVITY", "local", "adapter_state", "none"),
    ("a8_federated_delta_factor_rms", "PHASE3-SENSITIVITY", "federated", "adapter_delta", "factor_rms"),
])
def test_viability_records_are_consistent(name, label, source, rep, mode):
    v = _load(f"ae_viability/{name}.json")
    stats = _load(f"tgap/{source}_state_seed1_stats.json")
    assert v["label"] == label and v["representation"] == rep and v["normalization"]["mode"] == mode
    assert v["snapshot_index_sha256"] == stats["snapshot_index_sha256"]  # the gated data are the documented TGAP set
    train_idx = [i for i, r in enumerate(stats["manifest"]) if r["time_index"] < 16]  # temporal split, index order
    assert len(train_idx) == 80 and v["normalization"]["fit"]["train_indices"] == train_idx  # fitted on the training split only
    tr = v["ae_training"]
    assert tr["input_shape"] == [1, 2048, 1536] and tr["latent_shape"] == [64, 32, 24] and tr["compression_ratio_elements"] == 1 / 64
    assert tr["normalization_uplink_bytes"] == 0 and tr["config"]["iterations"] == 3000 and tr["config"]["batch_size"] == 4
    assert tr["best_val"]["iteration"] % 50 == 0
    assert v["val"]["num_snapshots"] == 20 and v["train"]["num_snapshots"] == 80
    crit = v["gate"]["criteria"]
    assert len(crit) == 7 and v["gate"]["pass"] == all(c["pass"] for c in crit.values())
    assert v["gate"]["pass"] is False  # recorded outcome: no Phase-3A codec is viable


def test_a6_selection_follows_from_the_recorded_gates():
    s = _load("ae_viability/a6_selection.json")
    gates = {m: _load(f"ae_viability/a5_federated_state_{m}.json")["gate"] for m in A5}
    recomputed = select_primary(gates)
    assert s["selected"] is None and recomputed["selected"] is None and s["verdict"] == "NO PRIMARY CODEC IS VIABLE"
    assert {m: g["pass"] for m, g in s["gates"].items()} == {m: g["pass"] for m, g in gates.items()}


def test_reference_codes_and_microbatch_diagnostic_are_recorded():
    for name in ("federated_state", "federated_delta", "local_state"):
        r = _load(f"reference_codes/{name}.json")
        assert r["label"] == "DERIVED" and set(r["codes"]) == {"lowpass_dct", "topk_dct", "topk_raw", "train_pca"}
        assert r["num_snapshots"] == 20
    m = _load("microbatch/microbatch_diag.json")
    assert m["label"] == "PHASE3-DIAGNOSTIC" and len(m["label_tokens_per_example"]) == 32
    assert m["vs_micro_batch_1"]["1"]["all"]["cosine"] == pytest.approx(1.0, abs=1e-12) and m["vs_token_mean_full_batch"]["32"]["all"]["rel_l2_diff"] == 0.0
