"""TGAP collection in both source modes shares one snapshot schema (items 12, 15)."""

from __future__ import annotations

from cg_fedllm.compression.layout import get_layout
from cg_fedllm.pipeline import simulator_spec
from cg_fedllm.tgap.collect import collect_federated_pretrain, collect_local_pretrain
from cg_fedllm.tgap.snapshots import SnapshotWriter, read_index
from tests.conftest import synthetic_clients


def test_both_tgap_sources_share_the_schema(tiny_cfg, tiny_bundle, tmp_path):
    lay = get_layout("layer_major_qkvo_AtB")
    d1 = synthetic_clients([4, 5, 3, 6], seed=3)
    w_local = SnapshotWriter(
        tmp_path / "local",
        run_id="l",
        source_mode="local_pretrain",
        representation="adapter_state",
        layout=lay,
    )
    out_l = collect_local_pretrain(
        tiny_bundle.trainer,
        tiny_bundle.initial_state,
        d1,
        clients=[0, 1, 2, 3],
        num_time_steps=2,
        writer=w_local,
    )
    w_fed = SnapshotWriter(
        tmp_path / "fed",
        run_id="f",
        source_mode="federated_pretrain",
        representation="adapter_state",
        layout=lay,
    )
    spec = simulator_spec(tiny_cfg, 4, namespace="tgap_fed")
    out_f = collect_federated_pretrain(
        tiny_bundle.trainer,
        tiny_bundle.initial_state,
        d1,
        spec=spec,
        run_dir=tmp_path / "fed" / "fl",
        identity={"t": 1},
        writer=w_fed,
    )
    rl, rf = read_index(tmp_path / "local"), read_index(tmp_path / "fed")
    assert out_l["num_snapshots"] == len(rl) == 8
    assert out_f["num_snapshots"] == len(rf) == 2 * 2  # 2 rounds x K=2
    assert set(rl[0]) == set(rf[0])
    # local_pretrain: each client continues from its own previous state (no aggregation)
    by_c = {(r["client_id"], r["time_index"]): r for r in rl}
    assert by_c[(0, 1)]["start_adapter_hash"] == by_c[(0, 0)]["end_adapter_hash"]
    assert by_c[(0, 0)]["start_adapter_hash"] == tiny_bundle.initial_state.sha256()
    # federated_pretrain: round-1 snapshots start from the aggregated round-0 global state
    r1 = [r for r in rf if r["time_index"] == 1]
    assert all(r["start_adapter_hash"] not in {x["end_adapter_hash"] for x in rf} for r in r1)
