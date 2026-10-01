"""Full Tier-C end-to-end smoke pipeline on the deterministic tiny CPU fixture (item 25)."""

from __future__ import annotations

import json
import math

from cg_fedllm.smoke import run_smoke
from tests.conftest import synthetic_clients


def test_end_to_end_smoke_on_cpu(tiny_cfg, tiny_bundle, tmp_path):
    d1 = synthetic_clients([4, 5, 3, 6], seed=7)
    d2 = synthetic_clients([8, 10, 6, 12], seed=8)
    heldout = synthetic_clients([6], seed=9)[0]
    res = run_smoke(tiny_cfg, tiny_bundle, d1, d2, heldout, tmp_path / "smoke", gpu_tolerance=None, run_controls=True)
    checks = res["checks"]
    assert checks["identity_state_vs_lora_ft"]["bitwise_equal"] is True
    assert checks["identity_state_vs_lora_ft"]["round_hashes_equal"] is True
    assert checks["identity_delta_vs_lora_ft"]["pass"] is True
    assert checks["resume_vs_uninterrupted"]["bitwise_equal"] is True
    assert checks["n1_fl_vs_local"]["bitwise_equal"] is True
    for stage in ("lora_ft", "faf_identity", "faf_autoencoder", "faf_constant_mean", "faf_gaussian_noise", "cent"):
        assert res[stage]["status"] == "complete"
        assert all(math.isfinite(v["loss"]) for v in res[stage]["heldout_by_round"].values())
    # logical uplink accounting: AE sends 1/64 of the identity payload
    assert res["faf_autoencoder"]["uplink_logical_bytes_total"] * 64 == res["faf_identity"]["uplink_logical_bytes_total"]
    assert res["faf_constant_mean"]["uplink_logical_bytes_total"] == 0
    summary = json.loads((tmp_path / "smoke" / "smoke_summary.json").read_text(encoding="utf-8"))
    assert summary["label"] == "PHASE2-SMOKE"
    assert (tmp_path / "smoke" / "smoke_summary.md").exists()
