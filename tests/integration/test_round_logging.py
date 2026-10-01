"""Phase-4 F7 per-round logging: effective (s B A) norms, update norms, throughput and two-way logical bytes."""

from __future__ import annotations

import json
import math

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
