"""Phase-4 F7 per-round logging: effective (s B A) norms, update norms, throughput and two-way logical bytes."""

from __future__ import annotations

import json
import math
import shutil

from cg_fedllm.federated.regression import compare_runs
from cg_fedllm.federated.simulator import FederatedSimulator
from cg_fedllm.models.adapter import AdapterState
from cg_fedllm.pipeline import simulator_spec
from tests.conftest import synthetic_clients


def test_round_records_effective_and_communication_statistics(tiny_cfg, tiny_bundle, tmp_path):
    spec = simulator_spec(tiny_cfg, 4)
    assert spec.lora_scaling == tiny_cfg.lora.alpha / tiny_cfg.lora.r == 2.0
    spec.num_rounds = 1
    sim = FederatedSimulator(tiny_bundle.trainer, spec, synthetic_clients([6, 9, 5, 8]), tmp_path / "fl", codec=None, layout=None, identity={"t": 1})
    summary = sim.run(tiny_bundle.initial_state)
    rnd = json.loads((tmp_path / "fl" / "rounds" / "r0000" / "round.json").read_text(encoding="utf-8"))
    k = len(rnd["selected_clients"])
    n = tiny_bundle.initial_state.num_elements()
    assert rnd["downlink_logical_bytes"] == k * n * 4 == rnd["uplink_logical_bytes"]
    assert summary["downlink_logical_bytes_total"] == rnd["downlink_logical_bytes"]
    assert rnd["throughput"]["label_tokens"] == sum(c["num_label_tokens"] for c in rnd["clients"]) > 0
    assert rnd["throughput"]["padded_tokens"] >= rnd["throughput"]["label_tokens"]
    new = AdapterState.load(tmp_path / "fl" / "rounds" / "r0000" / "global_adapter.safetensors")
    old = tiny_bundle.initial_state
    dense_new = dense_upd = 0.0
    for key in new.keys():
        if "lora_A" in key:
            kb = key.replace("lora_A", "lora_B")
            m_new = 2.0 * new[kb].double() @ new[key].double()
            m_old = 2.0 * old[kb].double() @ old[key].double()
            dense_new += float((m_new**2).sum())
            dense_upd += float(((m_new - m_old) ** 2).sum())
    assert math.isclose(rnd["effective_global"]["M_fro_sq_total"], dense_new, rel_tol=1e-9)
    assert math.isclose(rnd["update_norms"]["M_update_fro"], math.sqrt(dense_upd), rel_tol=1e-9)
    assert math.isclose(rnd["update_norms"]["B_update_fro"], math.sqrt(new.sub(old).l2_sq("B")), rel_tol=1e-12)


def test_baseline_summary_and_identity_regression(tiny_cfg, tiny_bundle, tmp_path):
    from cg_fedllm.cli import main
    from cg_fedllm.compression.codecs import IdentityCodec
    from cg_fedllm.compression.layout import get_layout

    clients = synthetic_clients([6, 9, 5, 8], seed=41)
    for name, codec in (("lora_ft", None), ("faf_identity", IdentityCodec())):
        spec = simulator_spec(tiny_cfg, 4)
        sim = FederatedSimulator(tiny_bundle.trainer, spec, clients, tmp_path / name, codec=codec, layout=get_layout("layer_major_qkvo_AtB") if codec else None,
                                 identity={"t": name}, heldout_fn=None)
        sim.run(tiny_bundle.initial_state)
    out = tmp_path / "baseline.json"
    assert main(["baseline-summary", "--run", f"lora_ft={(tmp_path / 'lora_ft').as_posix()}", "--run", f"faf_identity={(tmp_path / 'faf_identity').as_posix()}",
                 "--identity-pair", "lora_ft,faf_identity", "--label", "PHASE4-BASELINE", "--out", out.as_posix()]) == 0
    s = json.loads(out.read_text(encoding="utf-8"))
    reg = s["identity_regression"]
    assert reg["round_hashes_equal"] is True and reg["final_adapter_bitwise_equal"] is True and reg["rounds_compared"] == 2
    assert reg["validated"] is True and reg["num_mismatches"] == 0 and reg["config"] is None  # no resolved configs in these run dirs
    assert all(v is True for k, v in reg["checks"].items() if k != "only_expected_config_differences")
    com = s["runs"]["lora_ft"]["communication"]
    n = tiny_bundle.initial_state.num_elements() * 4
    assert com["uplink_per_client_bytes"] == n and com["two_way_total_bytes"] == com["uplink_total_bytes"] + com["downlink_total_bytes"] == 2 * 2 * 2 * n
    assert s["label"] == "PHASE4-BASELINE" and len(s["runs"]["lora_ft"]["per_round"]) == 2

    # negative controls: a changed held-out loss in round 1 and an unexpected config difference must both fail
    tampered = tmp_path / "tampered"
    shutil.copytree(tmp_path / "faf_identity", tampered)
    rj = tampered / "rounds" / "r0001" / "round.json"
    rec = json.loads(rj.read_text(encoding="utf-8"))
    rec["heldout"] = {"loss": 1.0}
    rj.write_text(json.dumps(rec), encoding="utf-8")
    bad = compare_runs(tmp_path / "lora_ft", tampered)
    assert bad["validated"] is False and bad["first_mismatch_round"] == 1 and bad["checks"]["heldout_loss_every_round"] is False
    assert bad["checks"]["global_adapter_hash_every_round"] is True
    for d, codec, seed in ((tmp_path / "lora_ft", "none", 1), (tmp_path / "faf_identity", "identity", 2)):
        (d / "config.resolved.yaml").write_text(f"codec:\n  type: {codec}\nrun:\n  seed: {seed}\n", encoding="utf-8")
    cfg_bad = compare_runs(tmp_path / "lora_ft", tmp_path / "faf_identity")
    assert cfg_bad["config"]["resolved_config_differences"] == {"codec.type": ["none", "identity"], "run.seed": [1, 2]}
    assert cfg_bad["checks"]["only_expected_config_differences"] is False and cfg_bad["validated"] is False
    (tmp_path / "faf_identity" / "config.resolved.yaml").write_text("codec:\n  type: identity\nrun:\n  seed: 1\n", encoding="utf-8")
    assert compare_runs(tmp_path / "lora_ft", tmp_path / "faf_identity")["validated"] is True
    # a pooled (centralized) run reports no communication cost
    (tmp_path / "lora_ft" / "config.resolved.yaml").write_text("codec:\n  type: none\nfederated:\n  pooled: true\n", encoding="utf-8")
    assert main(["baseline-summary", "--run", f"cent={(tmp_path / 'lora_ft').as_posix()}", "--label", "PHASE4-BASELINE", "--out", out.as_posix()]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["runs"]["cent"]["communication"]["applicable"] is False
