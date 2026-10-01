"""TGAP snapshot schema (item 15)."""

from __future__ import annotations

import pytest
import torch

from cg_fedllm.compression.layout import get_layout
from cg_fedllm.tgap.snapshots import (
    REQUIRED_FIELDS,
    SnapshotError,
    SnapshotWriter,
    load_states,
    read_index,
    snapshot_tensor,
)
from tests.unit.test_representation_layout import make_state


@pytest.mark.parametrize("mode", ["local_pretrain", "federated_pretrain"])
def test_snapshot_schema_and_derivations(tmp_path, mode):
    lay = get_layout("layer_major_qkvo_AtB")
    w = SnapshotWriter(tmp_path, run_id="r", source_mode=mode, representation="adapter_state", layout=lay)
    start = make_state(64, 2, 4, seed=1)
    ends = [make_state(64, 2, 4, seed=s) for s in (2, 3)]
    w.write(0, 0, start, ends[0], 10)
    w.write(0, 1, start, ends[1], 12)
    recs = read_index(tmp_path)
    assert len(recs) == 2 and all(set(REQUIRED_FIELDS) <= set(r) for r in recs)
    assert {r["source_mode"] for r in recs} == {mode}
    assert len(list((tmp_path / "starts").iterdir())) == 1  # shared start stored once
    r0 = recs[0]
    assert r0["phi_shape"] == [1, 64, 64] and r0["layout_id"] == "layer_major_qkvo_AtB"
    assert r0["derivable_representations"] == ["adapter_state", "adapter_delta"]
    s, e = load_states(tmp_path, r0)
    assert s.equal(start) and e.equal(ends[0])
    x_state, ref_state = snapshot_tensor(tmp_path, r0, "adapter_state", lay)
    x_delta, ref_delta = snapshot_tensor(tmp_path, r0, "adapter_delta", lay)
    assert torch.equal(x_state, lay.forward(ends[0])) and torch.equal(ref_state, lay.forward(start))
    assert torch.equal(x_delta, lay.forward(ends[0].sub(start))) and torch.count_nonzero(ref_delta) == 0
    assert abs(r0["l2"]["delta_sq"] - ends[0].sub(start).l2_sq()) < 1e-6


def test_snapshot_tampering_is_detected(tmp_path):
    lay = get_layout("layer_major_qkvo_AtB")
    w = SnapshotWriter(tmp_path, run_id="r", source_mode="local_pretrain", representation="adapter_state", layout=lay)
    w.write(0, 0, make_state(64, 1, 4, seed=1), make_state(64, 1, 4, seed=2), 5)
    rec = read_index(tmp_path)[0]
    p = tmp_path / rec["file"]
    data = bytearray(p.read_bytes())
    data[-1] ^= 0xFF
    p.write_bytes(bytes(data))
    with pytest.raises(SnapshotError):
        load_states(tmp_path, rec)


def test_unknown_source_mode_rejected(tmp_path):
    with pytest.raises(ValueError):
        SnapshotWriter(tmp_path, run_id="r", source_mode="magic", representation="adapter_state", layout=get_layout("layer_major_qkvo_AtB"))
