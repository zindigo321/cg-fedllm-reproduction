"""Local LoRA training (item 18), one FL round (19), N=1 FL/local equivalence (20), IdentityCodec
baseline equivalence (21), resume determinism (22) -- all on the deterministic tiny CPU fixture."""

from __future__ import annotations

import pytest
import torch

from cg_fedllm.compression.codecs import GaussianNoiseCodec, IdentityCodec
from cg_fedllm.compression.layout import get_layout
from cg_fedllm.federated.client import optimizer_steps
from cg_fedllm.federated.simulator import FederatedSimulator, ResumeError
from cg_fedllm.models.adapter import AdapterState
from cg_fedllm.models.lora import base_parameter_fingerprint
from cg_fedllm.pipeline import simulator_spec
from tests.conftest import synthetic_clients

SIZES = [6, 9, 5, 8]


def _sim(cfg, bundle, clients, run_dir, codec=None, representation="adapter_state", stop=None, aggregation=None):
    spec = simulator_spec(cfg, len(clients))
    spec.representation = representation
    spec.stop_after_round = stop
    if aggregation:
        spec.aggregation = aggregation
    return FederatedSimulator(
        bundle.trainer, spec, clients, run_dir, codec=codec, layout=get_layout("layer_major_qkvo_AtB") if codec else None, identity={"test": "x"}
    )


def test_local_training_updates_only_lora_and_is_deterministic(tiny_bundle):
    b = tiny_bundle
    trainable = [n for n, p in b.peft_model.named_parameters() if p.requires_grad]
    assert trainable and all(".lora_A." in n or ".lora_B." in n for n in trainable)
    assert len(b.params) == 2 * 4 * 2  # A/B x q,k,v,o x 2 layers
    before = base_parameter_fingerprint(b.peft_model)
    data = synthetic_clients([10])[0]
    r1 = b.trainer.train(b.initial_state, data, ("t", 0, 0))
    r2 = b.trainer.train(b.initial_state, data, ("t", 0, 0))  # fresh optimizer: no state carried over
    assert base_parameter_fingerprint(b.peft_model) == before
    assert r1.end_state.equal(r2.end_state)
    assert not r1.end_state.equal(b.initial_state)
    assert all(torch.isfinite(torch.tensor(r1.step_losses)))
    assert r1.num_optimizer_steps == optimizer_steps(10, b.trainer.cfg) == 3  # ceil(ceil(10/2)/2)
    assert r1.start_hash == b.initial_state.sha256()
    r3 = b.trainer.train(b.initial_state, data, ("t", 0, 1))
    assert not r3.end_state.equal(r1.end_state)  # different seed keys -> different data order/dropout


def test_one_round_and_n1_equivalence(tiny_cfg, tiny_bundle, tmp_path):
    clients = synthetic_clients(SIZES)
    summary = _sim(tiny_cfg, tiny_bundle, clients, tmp_path / "fl").run(tiny_bundle.initial_state)
    assert summary["status"] == "complete" and summary["rounds_completed"] == 2
    pooled = [[ex for c in clients for ex in c]]
    spec_cfg = tiny_cfg
    sim = FederatedSimulator(tiny_bundle.trainer, simulator_spec(spec_cfg, 1), pooled, tmp_path / "n1", codec=None, layout=None, identity={"t": 1})
    sim.spec.num_rounds, sim.spec.client_fraction = 1, 1.0
    sim.run(tiny_bundle.initial_state)
    fl_state = AdapterState.load(tmp_path / "n1" / "final_adapter.safetensors")
    direct = tiny_bundle.trainer.train(tiny_bundle.initial_state, pooled[0], ("fl", 0, 0)).end_state
    assert fl_state.equal(direct)  # bitwise on CPU


def test_identity_codec_equals_uncompressed_baseline(tiny_cfg, tiny_bundle, tmp_path):
    clients = synthetic_clients(SIZES)
    _sim(tiny_cfg, tiny_bundle, clients, tmp_path / "base").run(tiny_bundle.initial_state)
    _sim(tiny_cfg, tiny_bundle, clients, tmp_path / "ident", codec=IdentityCodec()).run(tiny_bundle.initial_state)
    _sim(tiny_cfg, tiny_bundle, clients, tmp_path / "delta", codec=IdentityCodec(), representation="adapter_delta").run(tiny_bundle.initial_state)
    base = AdapterState.load(tmp_path / "base" / "final_adapter.safetensors")
    ident = AdapterState.load(tmp_path / "ident" / "final_adapter.safetensors")
    delta = AdapterState.load(tmp_path / "delta" / "final_adapter.safetensors")
    assert ident.equal(base)  # adapter_state identity path is bitwise lossless on CPU
    assert delta.relative_l2_diff(base) < 1e-6  # delta path: exact up to float32 rounding
    import json

    rnd = json.loads((tmp_path / "ident" / "rounds" / "r0000" / "round.json").read_text(encoding="utf-8"))
    c0 = rnd["clients"][0]
    assert c0["recovered_equals_local_bitwise"] is True
    assert c0["payload"]["logical_bytes"] == c0["payload"]["raw_fp32_bytes"] == 2 * 4 * 2 * 8 * 128 * 4


def test_resume_reproduces_uninterrupted_run(tiny_cfg, tiny_bundle, tmp_path):
    clients = synthetic_clients(SIZES)
    _sim(tiny_cfg, tiny_bundle, clients, tmp_path / "full", codec=IdentityCodec()).run(tiny_bundle.initial_state)
    first = _sim(tiny_cfg, tiny_bundle, clients, tmp_path / "resumed", codec=IdentityCodec(), stop=0).run(tiny_bundle.initial_state)
    assert first["status"] == "stopped_after_round_0"
    assert not (tmp_path / "resumed" / "final_adapter.safetensors").exists()
    second = _sim(tiny_cfg, tiny_bundle, clients, tmp_path / "resumed", codec=IdentityCodec()).run(tiny_bundle.initial_state)
    assert second["status"] == "complete"
    a = AdapterState.load(tmp_path / "full" / "final_adapter.safetensors")
    b = AdapterState.load(tmp_path / "resumed" / "final_adapter.safetensors")
    assert a.sha256() == b.sha256()
    with pytest.raises(ResumeError):
        _sim(tiny_cfg, tiny_bundle, clients, tmp_path / "resumed", codec=IdentityCodec(), aggregation="uniform_mean").run(tiny_bundle.initial_state)


def test_all_aggregation_modes_and_noise_codec_run(tiny_cfg, tiny_bundle, tmp_path):
    clients = synthetic_clients(SIZES)
    finals = {}
    for agg in ("sample_weighted_mean", "uniform_mean", "literal_sum"):
        _sim(tiny_cfg, tiny_bundle, clients, tmp_path / agg, aggregation=agg).run(tiny_bundle.initial_state)
        finals[agg] = AdapterState.load(tmp_path / agg / "final_adapter.safetensors")
    assert not finals["sample_weighted_mean"].equal(finals["uniform_mean"])
    assert finals["literal_sum"].l2_sq() > finals["uniform_mean"].l2_sq()
    s = _sim(tiny_cfg, tiny_bundle, clients, tmp_path / "noise", codec=GaussianNoiseCodec(1e-3)).run(tiny_bundle.initial_state)
    assert s["status"] == "complete"


def test_divergence_is_recorded_not_hidden(tiny_cfg, tiny_bundle, tmp_path):
    """A codec that corrupts the update must end the run with an explicit 'diverged' status."""

    class NaNCodec(IdentityCodec):
        codec_id = "nan_test"

        def decode(self, payload, ctx):
            return torch.full_like(payload.tensors["x"], float("nan"))

    clients = synthetic_clients(SIZES)
    out = _sim(tiny_cfg, tiny_bundle, clients, tmp_path / "nan", codec=NaNCodec()).run(tiny_bundle.initial_state)
    assert out["status"] == "diverged_in_round_0"
    assert (tmp_path / "nan" / "rounds" / "r0000" / "diverged.json").exists()
    assert not (tmp_path / "nan" / "rounds" / "r0000" / "DONE").exists()
    assert not (tmp_path / "nan" / "final_adapter.safetensors").exists()
