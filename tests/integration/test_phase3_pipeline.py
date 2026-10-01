"""Phase-3A pipeline on the tiny CPU fixture: TGAP collection -> train-ae (three normalisation modes, best-val
checkpoint) -> ae-viability -> ae-select -> tgap-stats through the real CLI, and FAF with a normalised AE codec."""

from __future__ import annotations

import json

import torch
from safetensors import safe_open

from cg_fedllm.cli import main
from cg_fedllm.compression.layout import get_layout
from cg_fedllm.federated.sampling import shepherd_select_clients
from cg_fedllm.federated.simulator import FederatedSimulator
from cg_fedllm.pipeline import build_codec, simulator_spec
from cg_fedllm.tgap.collect import collect_federated_pretrain, collect_local_pretrain, local_pretrain_clients
from cg_fedllm.tgap.snapshots import SnapshotWriter
from cg_fedllm.utils.seeding import numpy_rng
from tests.conftest import REPO, synthetic_clients

TINY = str(REPO / "configs" / "smoke" / "tiny_cpu.yaml")


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_local_client_selection_modes():
    assert local_pretrain_clients(100, 0.05, 1, "shepherd_round0") == shepherd_select_clients(100, 0.05, 0)
    # seeded_random keeps the Phase-2 behaviour exactly
    expected = sorted(int(c) for c in numpy_rng(7, "tgap_local_clients").choice(100, size=5, replace=False))
    assert local_pretrain_clients(100, 0.05, 7) == local_pretrain_clients(100, 0.05, 7, "seeded_random") == expected
    assert local_pretrain_clients(4, 1.0, 7, "shepherd_round0") == [0, 1, 2, 3]


def test_phase3a_cli_pipeline_on_tiny_snapshots(tiny_cfg, tiny_bundle, tmp_path):
    lay = get_layout("layer_major_qkvo_AtB")
    d1 = synthetic_clients([4, 5, 3, 6], seed=11)
    snaps = tmp_path / "tgap_fed"
    w = SnapshotWriter(snaps, run_id="t", source_mode="federated_pretrain", representation="adapter_state", layout=lay)
    spec = simulator_spec(tiny_cfg, 4, namespace="tgap_fed")
    spec.num_rounds, spec.client_fraction = 5, 0.5
    collect_federated_pretrain(tiny_bundle.trainer, tiny_bundle.initial_state, d1, spec=spec, run_dir=snaps / "fl", identity={"t": 1}, writer=w)
    local = tmp_path / "tgap_local"
    wl = SnapshotWriter(local, run_id="l", source_mode="local_pretrain", representation="adapter_state", layout=lay)
    collect_local_pretrain(tiny_bundle.trainer, tiny_bundle.initial_state, d1, clients=[0, 2], num_time_steps=5, writer=wl)

    common = ["--config", TINY, "--set", f"run.output_root={tmp_path.as_posix()}", "--set", "run.result_label=PHASE3-DIAGNOSTIC"]
    ae_set = ["--set", "autoencoder.checkpoint_policy=final_and_best_val", "--set", "autoencoder.iterations=20", "--set", "autoencoder.eval_every=5"]
    gates = []
    for mode in ("none", "global_rms", "factor_rms"):
        assert main(["train-ae", *common, *ae_set, "--snapshots", snaps.as_posix(), "--stage", f"ae_{mode}", "--set", f"autoencoder.normalization={mode}"]) == 0
        ae_dir = tmp_path / "tiny_cpu" / f"ae_{mode}"
        m = _read(ae_dir / "ae_metrics.json")
        assert m["label"] == "PHASE3-DIAGNOSTIC" and m["normalization"]["mode"] == mode
        assert m["normalization"]["fit"]["train_indices"] == m["train_indices"]  # fitted on the training split only
        assert m["normalization_uplink_bytes"] == 0 and m["compression_ratio_elements"] == 1 / 64
        assert m["best_val"]["iteration"] % 5 == 0
        assert m["best_val"]["val_mse"] == min(c["val_mse"] for c in m["curve"] if c["iteration"] % 5 == 0)
        assert m["normalized_range"]["val"]["mode"] == mode
        with safe_open(str(ae_dir / "autoencoder_best_val.safetensors"), framework="pt") as fh:
            meta = json.loads(fh.metadata()["metadata"])
        assert meta["checkpoint"] == "best_val" and meta["normalization"]["mode"] == mode
        out = tmp_path / f"viability_{mode}.json"
        assert main(["ae-viability", *common, "--snapshots", snaps.as_posix(), "--ae-dir", ae_dir.as_posix(), "--out", out.as_posix()]) == 0
        v = _read(out)
        assert v["label"] == "PHASE3-DIAGNOSTIC" and v["source_mode"] == "federated_pretrain"
        assert v["val"]["num_snapshots"] == 2 and v["train"]["num_snapshots"] == 8  # temporal split: last time index
        preds = v["val"]["predictors"]
        assert set(preds) == {"autoencoder_best_val", "autoencoder_final", "zero", "train_mean", "identity", "tanh_range_ceiling"}
        assert preds["identity"]["innovation"]["all"]["rel_sq_error"] == 0.0
        assert preds["zero"]["transmitted"]["A"]["rel_sq_error"] == 1.0
        for crit in ("finite_outputs", "A_pooled_rel_sq_error_lt_1", "B_pooled_rel_sq_error_lt_1", "innovation_cosine_ge_0_90",
                     "B_norm_ratio_in_0_5_2_0", "aggregate_update_cosine_ge_0_90", "input_dependent"):
            assert isinstance(v["gate"]["criteria"][crit]["pass"], bool)
        if mode == "none":
            assert preds["tanh_range_ceiling"]["transmitted"]["all"]["rel_sq_error"] == 0.0
        gates += ["--gate", f"{mode}={out.as_posix()}"]
    sel = tmp_path / "select.json"
    assert main(["ae-select", *gates, "--label", "PHASE3-DIAGNOSTIC", "--out", sel.as_posix()]) == 0
    s = _read(sel)
    assert set(s["gates"]) == {"none", "global_rms", "factor_rms"} and "selected" in s and s["label"] == "PHASE3-DIAGNOSTIC"

    stats = tmp_path / "stats.json"
    assert main(["tgap-stats", *common, "--snapshots", snaps.as_posix(), "--out", stats.as_posix()]) == 0
    st = _read(stats)
    assert st["num_snapshots"] == 10 and st["num_time_indices"] == 5 and st["label"] == "DERIVED"
    assert sum(int(k) * v for k, v in st["participation_histogram"].items()) == 10
    assert len(st["split"]["val"]) == 2 and st["split"]["val_time_indices"] == [4]
    assert len(st["manifest"]) == 10 and all("file_sha256" in r for r in st["manifest"])
    assert len(st["inter_time_aggregate_innovation_cosine"]["all"]["matrix"]) == 5
    st_l = tmp_path / "stats_local.json"
    assert main(["tgap-stats", *common, "--snapshots", local.as_posix(), "--out", st_l.as_posix()]) == 0
    assert _read(st_l)["unique_clients"] == 2 and _read(st_l)["distinct_start_states"] == 1 + 2 * 4

    # FAF through build_codec: the frozen normaliser travels with the checkpoint, the payload is the latent only
    cfg = tiny_cfg
    cfg.codec.type, cfg.codec.ae_checkpoint = "autoencoder", (tmp_path / "tiny_cpu" / "ae_factor_rms" / "autoencoder_best_val.safetensors").as_posix()
    codec = build_codec(cfg, torch.device("cpu"))
    assert codec.normalizer is not None and codec.normalizer.mode == "factor_rms"
    d2 = synthetic_clients([8, 10, 6, 12], seed=12)
    sim = FederatedSimulator(tiny_bundle.trainer, simulator_spec(cfg, 4), d2, tmp_path / "faf", codec=codec, layout=lay, identity={"t": 2})
    sim.spec.num_rounds = 1
    summary = sim.run(tiny_bundle.initial_state)
    assert summary["status"] == "complete"
    rnd = _read(tmp_path / "faf" / "rounds" / "r0000" / "round.json")
    assert rnd["codec"]["normalization"]["mode"] == "factor_rms"
    assert all(c["payload"]["logical_bytes"] * 64 == c["payload"]["raw_fp32_bytes"] for c in rnd["clients"])
