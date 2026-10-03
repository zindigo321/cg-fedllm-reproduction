"""Phase-4 F3-F6 forensic representations, gradient recording and the screen evaluator on the tiny CPU fixture."""

from __future__ import annotations

import json
import math

import torch

from cg_fedllm.cli import main
from cg_fedllm.compression.gauge import module_pairs
from cg_fedllm.compression.layout import get_layout, infer_geometry
from cg_fedllm.federated.simulator import FederatedSimulator
from cg_fedllm.forensics.gradients import GradientDumper, gradient_statistics, load_client_round, mean_state
from cg_fedllm.forensics.representations import balanced_effective_delta, balanced_effective_state
from cg_fedllm.forensics.screen import ScreenItem, evaluate, group_by_time, product_factors, screen_gate
from cg_fedllm.pipeline import simulator_spec
from cg_fedllm.tgap.collect import collect_federated_pretrain
from cg_fedllm.tgap.snapshots import SnapshotWriter, load_states, read_index
from tests.conftest import REPO, synthetic_clients

TINY = str(REPO / "configs" / "smoke" / "tiny_cpu.yaml")
S = 2.0


def _snapshots(tiny_cfg, tiny_bundle, root, rounds=5):
    lay = get_layout("layer_major_qkvo_AtB")
    w = SnapshotWriter(
        root, run_id="t", source_mode="federated_pretrain", representation="adapter_state", layout=lay
    )
    spec = simulator_spec(tiny_cfg, 4, namespace="tgap_fed")
    spec.num_rounds, spec.client_fraction = rounds, 0.5
    collect_federated_pretrain(
        tiny_bundle.trainer,
        tiny_bundle.initial_state,
        synthetic_clients([6, 9, 5, 8], seed=31),
        spec=spec,
        run_dir=root / "fl",
        identity={"t": 1},
        writer=w,
    )
    return read_index(root)


def test_balanced_representations_on_real_snapshots(tiny_cfg, tiny_bundle, tmp_path):
    recs = _snapshots(tiny_cfg, tiny_bundle, tmp_path / "s")
    start, end = load_states(tmp_path / "s", recs[-1])
    r2, info = balanced_effective_state(end, S)
    assert r2.keys() == end.keys() and info["tied_singular_values"] == 0
    r3, st = balanced_effective_delta(start, end, S, rank=8)
    assert 0.0 < st["energy_weighted_retained_energy"] <= 1.0 + 1e-12
    assert set(st["exact_rank_distribution"]) <= set(range(17))
    for ka, kb in module_pairs(end):
        dense = S * end[kb].double() @ end[ka].double()
        assert torch.allclose(r2[kb].double() @ r2[ka].double(), dense, atol=1e-6, rtol=0)  # float32 storage
        exact = S * (end[kb].double() @ end[ka].double() - start[kb].double() @ start[ka].double())
        u, sv, vh = torch.linalg.svd(exact)
        best = (u[:, :8] * sv[:8]) @ vh[:8]
        assert torch.allclose(r3[kb].double() @ r3[ka].double(), best, atol=1e-6, rtol=0)


def test_screen_metrics_identity_zero_and_first_order(tiny_cfg, tiny_bundle, tmp_path):
    recs = _snapshots(tiny_cfg, tiny_bundle, tmp_path / "s")
    lay = get_layout("layer_major_qkvo_AtB")
    items = []
    for i, r in enumerate(recs):
        start, end = load_states(tmp_path / "s", r)
        rep, _ = balanced_effective_state(end, S)
        items.append(
            ScreenItem(i, int(r["time_index"]), int(r["client_id"]), int(r["num_samples"]), rep, start, end)
        )
    geom = infer_geometry(items[0].rep)
    mean = torch.stack([lay.forward(it.rep, geom) for it in items[:4]]).mean(0)
    preds = {
        "identity": lambda x, it: x.clone(),
        "zero": lambda x, it: torch.zeros_like(x),
        "train_mean": lambda x, it: mean.clone(),
        "autoencoder_best_val": lambda x, it: x.clone(),
    }
    res = evaluate(group_by_time(items), "balanced_effective_state", S, lay, geom, preds)
    ident = res["identity"]
    assert ident["representation"]["all"]["rel_sq_error"] == 0.0 and math.isclose(
        ident["product"]["cosine"], 1.0, rel_tol=1e-9
    )
    assert ident["product"]["rel_fro_error"] < 1e-6 and ident["update_relevant"]["rel_sq_error"] < 1e-10
    assert ident["shift_control"]["matched_rel_sq_error"] < ident["shift_control"]["shifted_rel_sq_error"]
    assert math.isclose(res["zero"]["representation"]["all"]["rel_sq_error"], 1.0)
    gate = screen_gate(res)
    assert gate["pass"] is True  # an exact reconstruction passes every criterion
    # R4: the product target is the first-order effective change at the start state
    it = items[-1]
    g = it.end.sub(it.start)  # any factor-shaped tensor works as a stand-in gradient
    fac = product_factors("mean_step_gradient", g, it, S)
    for ka, kb in module_pairs(it.end):
        lt, rt = fac[ka]
        dense = S * (g[kb].double() @ it.start[ka].double() + it.start[kb].double() @ g[ka].double())
        assert torch.allclose(lt @ rt, dense, atol=1e-12, rtol=0)


def test_gradient_recording_reproduces_training(tiny_cfg, tiny_bundle, tmp_path):
    lay = get_layout("layer_major_qkvo_AtB")
    root = tmp_path / "gf"
    w = SnapshotWriter(
        root, run_id="g", source_mode="federated_pretrain", representation="adapter_state", layout=lay
    )
    clients = synthetic_clients([6, 9, 5, 8], seed=32)
    spec = simulator_spec(tiny_cfg, 4, namespace="tgap_fed")
    spec.num_rounds, spec.client_fraction = 2, 0.5

    def hook(t, cid, start, end, n, rec):
        w.write(t, cid, start, end, n, {})

    sim = FederatedSimulator(
        tiny_bundle.trainer,
        spec,
        clients,
        root / "fl",
        codec=None,
        layout=None,
        identity={"t": 2},
        snapshot_hook=hook,
        observer_factory=lambda t, cid: GradientDumper(root / "gradients", t, cid),
    )
    sim.run(tiny_bundle.initial_state)
    # the observer must not change the optimisation: same schedule without it gives bitwise-identical states
    ref_root = tmp_path / "ref"
    w2 = SnapshotWriter(
        ref_root, run_id="r", source_mode="federated_pretrain", representation="adapter_state", layout=lay
    )
    collect_federated_pretrain(
        tiny_bundle.trainer,
        tiny_bundle.initial_state,
        clients,
        spec=spec,
        run_dir=ref_root / "fl",
        identity={"t": 2},
        writer=w2,
    )
    recs, ref = read_index(root), read_index(ref_root)
    assert [r["end_adapter_hash"] for r in recs] == [r["end_adapter_hash"] for r in ref]
    cr = load_client_round(root / "gradients" / f"t0000_c{int(recs[0]['client_id']):04d}")
    assert len(cr["grads"]) == len(cr["clipped"]) == len(cr["deltas"]) >= 1
    stats = gradient_statistics(root / "gradients", root, recs, S)
    assert stats["client_rounds"] == len(recs) and stats["optimizer_steps"]["total"] >= len(recs)
    fam = stats["families"]
    assert set(fam) == {
        "last_step_gradient",
        "mean_step_gradient",
        "optimizer_step_delta",
        "local_epoch_delta",
        "pre_clip_step_gradient",
    }
    assert fam["mean_step_gradient"]["per_element_rms"] > 0 and fam["local_epoch_delta"]["samples"] == len(
        recs
    )
    assert stats["local_epoch_delta_first_order_vs_exact"]["cosine_mean"] > 0.9
    m = mean_state(cr["grads"])
    assert torch.allclose(_flat(m), sum(_flat(g) for g in cr["grads"]) / len(cr["grads"]), atol=1e-7)


def _flat(st):
    return torch.cat([st.tensors[k].reshape(-1).double() for k in st.keys()])


def test_forensic_screen_cli_mean_step_gradient(tiny_cfg, tiny_bundle, tmp_path):
    lay = get_layout("layer_major_qkvo_AtB")
    root = tmp_path / "gf"
    w = SnapshotWriter(
        root, run_id="g", source_mode="federated_pretrain", representation="adapter_state", layout=lay
    )
    spec = simulator_spec(tiny_cfg, 4, namespace="tgap_fed")
    spec.num_rounds, spec.client_fraction = 2, 0.5

    def hook(t, cid, start, end, n, rec):
        w.write(t, cid, start, end, n, {})

    sim = FederatedSimulator(
        tiny_bundle.trainer,
        spec,
        synthetic_clients([6, 9, 5, 8], seed=33),
        root / "fl",
        codec=None,
        layout=None,
        identity={"t": 3},
        snapshot_hook=hook,
        observer_factory=lambda t, cid: GradientDumper(root / "gradients", t, cid),
    )
    sim.run(tiny_bundle.initial_state)
    common = [
        "--config",
        TINY,
        "--set",
        f"run.output_root={tmp_path.as_posix()}",
        "--set",
        "run.result_label=PHASE4-FORENSIC",
    ]
    ae = [
        "--set",
        "autoencoder.iterations=10",
        "--set",
        "autoencoder.eval_every=5",
        "--set",
        "autoencoder.checkpoint_policy=final_and_best_val",
    ]
    out = tmp_path / "r4.json"
    assert (
        main(
            [
                "forensic-screen",
                *common,
                *ae,
                "--snapshots",
                root.as_posix(),
                "--gradients",
                (root / "gradients").as_posix(),
                "--candidate",
                "mean_step_gradient",
                "--out",
                out.as_posix(),
            ]
        )
        == 0
    )
    sc = json.loads(out.read_text(encoding="utf-8"))
    assert sc["split"] == {"train": 2, "val": 2, "val_time_indices": [1]}  # round 0 trains, round 1 validates
    assert sc["scale_rule"]["modes_run"] == ["none"]  # gradients are far inside the Tanh range
    ident = sc["modes"]["none"]["val"]["identity"]
    assert (
        ident["product"]["rel_fro_error"] < 1e-6
        and ident["aggregate_gradient_space"]["all"]["rel_sq_error"] < 1e-12
    )


def test_forensic_cli_on_tiny_snapshots(tiny_cfg, tiny_bundle, tmp_path):
    _snapshots(tiny_cfg, tiny_bundle, tmp_path / "s")
    common = [
        "--config",
        TINY,
        "--set",
        f"run.output_root={tmp_path.as_posix()}",
        "--set",
        "run.result_label=PHASE4-FORENSIC",
    ]
    out = tmp_path / "stats.json"
    assert (
        main(["forensic-stats", *common, "--snapshots", (tmp_path / "s").as_posix(), "--out", out.as_posix()])
        == 0
    )
    st = json.loads(out.read_text(encoding="utf-8"))
    assert st["label"] == "PHASE4-FORENSIC" and len(st["per_snapshot"]) == 10
    demo = st["gauge_invariance_demo"]
    assert (
        demo["max_relative_singular_value_change"] < 1e-5
        and abs(demo["raw_factor_sq_after"] - demo["raw_factor_sq_before"]) > 1e-3
    )
    assert isinstance(st["r3_structural_prerequisite"]["structurally_lossy"], bool)
    scr = tmp_path / "screen.json"
    ae = [
        "--set",
        "autoencoder.iterations=10",
        "--set",
        "autoencoder.eval_every=5",
        "--set",
        "autoencoder.checkpoint_policy=final_and_best_val",
    ]
    assert (
        main(
            [
                "forensic-screen",
                *common,
                *ae,
                "--snapshots",
                (tmp_path / "s").as_posix(),
                "--candidate",
                "balanced_effective_delta_r8",
                "--out",
                scr.as_posix(),
            ]
        )
        == 0
    )
    sc = json.loads(scr.read_text(encoding="utf-8"))
    assert sc["candidate"] == "balanced_effective_delta_r8" and sc["scale_rule"]["modes_run"][0] == "none"
    g = sc["modes"]["none"]["gate"]
    assert set(g["criteria"]) == {
        "S1_finite_outputs",
        "S2_representation_cosine_ge_0_90",
        "S3_representation_rel_sq_error_le_0_50",
        "S4_product_cosine_ge_0_90",
        "S5_product_rel_fro_error_le_0_50",
        "S6_input_dependent",
        "S7_materially_better_than_train_mean",
    }
    assert "product_vs_exact_delta" in sc["modes"]["none"]["val"]["identity"]
    assert sc["modes"]["none"]["val"]["identity"]["product"]["rel_fro_error"] < 1e-6


def test_distribution_shape_reference_values(tiny_cfg, tiny_bundle, tmp_path):
    from cg_fedllm.forensics.gradients import distribution_shape, family_shapes

    g = torch.Generator().manual_seed(0)
    gauss = distribution_shape([torch.randn(400_000, generator=g)], drop_zeros=False)
    unif = distribution_shape([torch.rand(400_000, generator=g) * 2 - 1, torch.zeros(10)], drop_zeros=True)
    assert abs(gauss["excess_kurtosis"]) < 0.05 and abs(unif["excess_kurtosis"] + 1.2) < 0.02
    assert (
        unif["exact_zeros"] == 10
        and unif["values"] == 400_000
        and abs(unif["abs_quantiles"]["p50"] - 0.5) < 0.01
    )
    assert gauss["fraction_abs_gt_0_01"] > 0.99 and abs(gauss["std"] - 1) < 0.01
    # on recorded tiny-model dumps: every family is present and the deltas are bounded by the learning rate
    lay = get_layout("layer_major_qkvo_AtB")
    root = tmp_path / "gf"
    w = SnapshotWriter(
        root, run_id="g", source_mode="federated_pretrain", representation="adapter_state", layout=lay
    )
    spec = simulator_spec(tiny_cfg, 4, namespace="tgap_fed")
    spec.num_rounds, spec.client_fraction = 2, 0.5

    def hook(t, cid, start, end, n, rec):
        w.write(t, cid, start, end, n, {})

    sim = FederatedSimulator(
        tiny_bundle.trainer,
        spec,
        synthetic_clients([6, 9, 5, 8], seed=34),
        root / "fl",
        codec=None,
        layout=None,
        identity={"t": 4},
        snapshot_hook=hook,
        observer_factory=lambda t, cid: GradientDumper(root / "gradients", t, cid),
    )
    sim.run(tiny_bundle.initial_state)
    shapes = family_shapes(root / "gradients", root, read_index(root))
    assert set(shapes) == {
        "pre_clip_step_gradient",
        "mean_step_gradient",
        "optimizer_step_delta",
        "local_epoch_delta",
    }
    assert shapes["optimizer_step_delta"]["max_abs"] <= tiny_cfg.local_train.learning_rate * (1 + 1e-3)
    assert (
        shapes["mean_step_gradient"]["zeros_excluded"] is True
        and shapes["local_epoch_delta"]["zeros_excluded"] is False
    )
