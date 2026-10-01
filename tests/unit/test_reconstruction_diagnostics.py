"""Phase-3 A1: factor-aware reconstruction diagnostics on synthetic states whose A and B factors differ by 100x."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from cg_fedllm.compression.diagnostics import SnapshotItem, evaluate_predictors, lowrank_inner
from cg_fedllm.compression.layout import LoRAGeometry, get_layout
from cg_fedllm.compression.normalization import factor_column_mask
from cg_fedllm.federated.aggregation import aggregate
from cg_fedllm.models.adapter import AdapterState
from cg_fedllm.tgap.viability import reconstruction_gate, select_primary

GEOM = LoRAGeometry(num_layers=2, modules=("q_proj", "k_proj", "v_proj", "o_proj"), rank=8, hidden=128)
LAY = get_layout("layer_major_qkvo_AtB")
MASK = factor_column_mask(LAY, GEOM)


def _rand_state(g: torch.Generator, a_scale: float, b_scale: float) -> AdapterState:
    t = {}
    for layer in range(GEOM.num_layers):
        for m in GEOM.modules:
            t[f"layers.{layer}.self_attn.{m}.lora_A.weight"] = torch.randn(8, 128, generator=g) * a_scale
            t[f"layers.{layer}.self_attn.{m}.lora_B.weight"] = torch.randn(128, 8, generator=g) * b_scale
    return AdapterState(t)


def _items(num_times: int = 2, clients: int = 3, seed: int = 0) -> list[list[SnapshotItem]]:
    """Federated-style snapshots: all clients of a time index share the start; A ~ 1e-2, B ~ 1e-4 (100x)."""
    g = torch.Generator().manual_seed(seed)
    start = _rand_state(g, 1e-2, 1e-4)
    groups, idx = [], 0
    for t in range(num_times):
        group = []
        for c in range(clients):
            upd = _rand_state(g, 1e-4, 1e-5)
            group.append(SnapshotItem(idx, t, c, 10 + 7 * c, start, start.add(upd)))
            idx += 1
        groups.append(group)
        start = aggregate([it.end for it in group], [it.num_samples for it in group], "sample_weighted_mean")
    return groups


def _eval(predictors, representation="adapter_state", groups=None, **kw):
    return evaluate_predictors(groups or _items(), representation, LAY, GEOM, predictors, **kw)


def test_whole_state_error_hides_a_destroyed_B_factor():
    def inflate_b(x, it):  # A reconstructed exactly, every B value 10x too large
        y = x.clone()
        y[..., ~MASK] *= 10.0
        return y

    r = _eval({"inflate_b": inflate_b})["inflate_b"]
    tx = r["transmitted"]
    assert tx["all"]["rel_sq_error"] < 1e-2  # looks acceptable on the whole state ...
    assert tx["all"]["cosine"] > 0.99
    assert math.isclose(tx["B"]["rel_sq_error"], 81.0, rel_tol=1e-9)  # ... but B is destroyed
    assert math.isclose(tx["B"]["norm_ratio"], 10.0, rel_tol=1e-9)
    assert tx["A"]["rel_sq_error"] == 0.0 and tx["A"]["max_abs_error"] == 0.0
    # the same corruption dominates the innovation, the product B A and the aggregated server update
    assert r["innovation"]["all"]["rel_sq_error"] > 1.0
    assert r["product"]["state"]["norm_ratio"] > 5.0
    assert r["aggregate"]["update"]["B"]["norm_ratio"] > 5.0
    gate = reconstruction_gate({"predictors": {"autoencoder_best_val": r, "train_mean": r}})
    assert gate["pass"] is False
    assert gate["criteria"]["B_pooled_rel_sq_error_lt_1"]["pass"] is False
    assert gate["criteria"]["B_norm_ratio_in_0_5_2_0"]["pass"] is False
    assert gate["criteria"]["A_pooled_rel_sq_error_lt_1"]["pass"] is True


def test_state_accuracy_says_nothing_about_the_innovation():
    """Predicting the round-start state reconstructs the state almost perfectly and the update not at all."""
    groups = _items()
    starts = {it.index: LAY.forward(it.start, GEOM) for g in groups for it in g}
    r = _eval({"start": lambda x, it: starts[it.index].clone(), "identity": lambda x, it: x.clone()}, groups=groups)
    s = r["start"]
    assert s["state"]["all"]["rel_sq_error"] < 1e-3 and s["state"]["A"]["cosine"] > 0.999
    assert math.isclose(s["innovation"]["all"]["rel_sq_error"], 1.0, rel_tol=1e-9)
    assert math.isnan(s["innovation"]["all"]["cosine"])  # zero reconstructed innovation: undefined, never >= 0.9
    i = r["identity"]
    for grp in ("transmitted", "state", "innovation"):
        for f in ("all", "A", "B"):
            assert i[grp][f]["rel_sq_error"] == 0.0 and i[grp][f]["max_abs_error"] == 0.0
            assert math.isclose(i[grp][f]["cosine"], 1.0, rel_tol=1e-12) and math.isclose(i[grp][f]["norm_ratio"], 1.0, rel_tol=1e-12)
    assert i["product"]["state"]["rel_sq_error"] == 0.0 and math.isclose(i["product"]["innovation"]["cosine"], 1.0, rel_tol=1e-9)
    assert i["aggregate"]["update"]["all"]["rel_l2_error"] == 0.0
    assert i["innovation_energy_fraction"] < 1e-2


def test_lowrank_products_match_dense_computation():
    g = torch.Generator().manual_seed(3)
    l1, r1 = torch.randn(64, 16, generator=g, dtype=torch.float64), torch.randn(16, 48, generator=g, dtype=torch.float64)
    l2, r2 = torch.randn(64, 16, generator=g, dtype=torch.float64), torch.randn(16, 48, generator=g, dtype=torch.float64)
    assert math.isclose(lowrank_inner(l1, r1, l2, r2), float(((l1 @ r1) * (l2 @ r2)).sum()), rel_tol=1e-10)
    groups = _items(num_times=1, clients=2)

    def noisy(x, it):
        return x + 1e-4 * torch.randn(x.shape, generator=torch.Generator().manual_seed(it.index))

    r = _eval({"noisy": noisy}, groups=groups)["noisy"]
    sse = sig = sig_u = dot_u = hat_u = 0.0
    for it in groups[0]:
        x_hat = noisy(LAY.forward(it.end, GEOM), it)
        rec = LAY.inverse(x_hat, GEOM)
        for layer in range(GEOM.num_layers):
            for m in GEOM.modules:
                ka, kb = f"layers.{layer}.self_attn.{m}.lora_A.weight", f"layers.{layer}.self_attn.{m}.lora_B.weight"
                p = it.end[kb].double() @ it.end[ka].double()
                ph = rec[kb].double() @ rec[ka].double()
                ps = it.start[kb].double() @ it.start[ka].double()
                sse += float(((ph - p) ** 2).sum())
                sig += float((p**2).sum())
                sig_u += float(((p - ps) ** 2).sum())
                hat_u += float(((ph - ps) ** 2).sum())
                dot_u += float(((p - ps) * (ph - ps)).sum())
    assert math.isclose(r["product"]["state"]["rel_sq_error"], sse / sig, rel_tol=1e-8)
    assert math.isclose(r["product"]["innovation"]["rel_sq_error"], sse / sig_u, rel_tol=1e-8)
    assert math.isclose(r["product"]["innovation"]["cosine"], dot_u / math.sqrt(sig_u * hat_u), rel_tol=1e-8)


def test_aggregate_replay_matches_fedavg_and_quantiles_are_exact():
    groups = _items(num_times=1, clients=3)

    def scaled(x, it):
        return x * 0.5

    r = _eval({"half": scaled}, groups=groups)["half"]
    items = groups[0]
    true = aggregate([it.end for it in items], [it.num_samples for it in items], "sample_weighted_mean")
    hat = aggregate([LAY.inverse(scaled(LAY.forward(it.end, GEOM), it), GEOM) for it in items], [it.num_samples for it in items], "sample_weighted_mean")
    start = items[0].start
    u, uh = true.sub(start), hat.sub(start)
    num = sum(float(((uh[k].double() - u[k].double()) ** 2).sum()) for k in u.keys())
    den = sum(float((u[k].double() ** 2).sum()) for k in u.keys())
    assert math.isclose(r["aggregate"]["update"]["all"]["rel_l2_error"], math.sqrt(num / den), rel_tol=1e-5)
    # absolute-error quantiles are exact numpy quantiles of the pooled transmitted error
    errs = np.concatenate([(0.5 * LAY.forward(it.end, GEOM) - LAY.forward(it.end, GEOM))[..., MASK].abs().reshape(-1).double().numpy() for it in items])
    assert math.isclose(r["transmitted"]["A"]["abs_error_p95"], float(np.quantile(errs, 0.95)), rel_tol=1e-6)
    assert r["transmitted"]["A"]["quantiles_exact"] is True


def test_shift_control_detects_input_independent_decoders():
    groups = _items()
    mean = torch.stack([LAY.forward(it.end, GEOM) for g in groups for it in g]).mean(0)
    r = _eval({"const": lambda x, it: mean.clone(), "identity": lambda x, it: x.clone()}, groups=groups)
    c = r["const"]["shift_control"]
    assert math.isclose(c["innovation_rel_sq_error_matched"], c["innovation_rel_sq_error_shifted"], rel_tol=1e-12)
    i = r["identity"]["shift_control"]
    assert i["innovation_rel_sq_error_matched"] == 0.0 < i["innovation_rel_sq_error_shifted"]
    gate = reconstruction_gate({"predictors": {"autoencoder_best_val": r["const"], "train_mean": r["const"]}})
    assert gate["criteria"]["input_dependent"]["pass"] is False


def test_delta_representation_metrics_and_gate_priority():
    groups = _items()
    r = _eval({"identity": lambda x, it: x.clone(), "zero": lambda x, it: torch.zeros_like(x)}, representation="adapter_delta", groups=groups)
    i, z = r["identity"], r["zero"]
    assert i["transmitted"]["all"]["rel_sq_error"] == 0.0
    assert i["innovation"]["all"]["rel_sq_error"] < 1e-12 and i["state"]["all"]["rel_sq_error"] < 1e-12
    assert math.isclose(z["transmitted"]["B"]["rel_sq_error"], 1.0) and math.isclose(z["innovation"]["all"]["rel_sq_error"], 1.0)
    # a perfect predictor passes every criterion; zero fails
    gate_ok = reconstruction_gate({"predictors": {"autoencoder_best_val": i, "train_mean": z}})
    assert gate_ok["pass"] is True, gate_ok
    gate_zero = reconstruction_gate({"predictors": {"autoencoder_best_val": z, "train_mean": z}})
    assert gate_zero["pass"] is False
    assert select_primary({"none": gate_zero, "global_rms": gate_ok, "factor_rms": gate_ok})["selected"] == "global_rms"
    assert select_primary({"none": gate_ok, "global_rms": gate_ok})["selected"] == "none"
    verdict = select_primary({"none": gate_zero, "global_rms": gate_zero, "factor_rms": gate_zero})
    assert verdict["selected"] is None and verdict["verdict"] == "NO PRIMARY CODEC IS VIABLE"


def test_non_finite_reconstructions_fail_every_criterion():
    groups = _items(num_times=1)
    r = _eval({"nan": lambda x, it: torch.full_like(x, float("nan")), "zero": lambda x, it: torch.zeros_like(x)}, groups=groups)
    assert r["nan"]["finite_outputs"] is False
    gate = reconstruction_gate({"predictors": {"autoencoder_best_val": r["nan"], "train_mean": r["zero"]}})
    assert gate["pass"] is False and not any(c["pass"] for c in gate["criteria"].values())


@pytest.mark.parametrize("bad", [0.49, 2.01])
def test_b_norm_ratio_band_edges(bad):
    groups = _items(num_times=2)

    def scale_b(x, it):
        y = x.clone()
        y[..., ~MASK] *= bad
        return y

    r = _eval({"s": scale_b}, groups=groups)["s"]
    assert math.isclose(r["transmitted"]["B"]["norm_ratio"], bad, rel_tol=1e-6)
    gate = reconstruction_gate({"predictors": {"autoencoder_best_val": r, "train_mean": r}})
    assert gate["criteria"]["B_norm_ratio_in_0_5_2_0"]["pass"] is False
