"""AE training on TGAP snapshots without NaNs, plus a learnability control (item 13)."""

from __future__ import annotations

import math

import torch

from cg_fedllm.compression.layout import get_layout
from cg_fedllm.tgap.collect import collect_local_pretrain
from cg_fedllm.tgap.snapshots import SnapshotWriter, read_index, snapshot_tensor
from cg_fedllm.tgap.train_ae import train_autoencoder
from tests.conftest import synthetic_clients


def test_ae_trains_on_snapshots_and_learns_structured_control(tiny_cfg, tiny_bundle, tmp_path):
    lay = get_layout("layer_major_qkvo_AtB")
    d1 = synthetic_clients([4, 5, 3, 6], seed=4)
    w = SnapshotWriter(tmp_path / "snaps", run_id="s", source_mode="local_pretrain", representation="adapter_state", layout=lay)
    collect_local_pretrain(tiny_bundle.trainer, tiny_bundle.initial_state, d1, clients=[0, 1, 2, 3], num_time_steps=2, writer=w)
    recs = read_index(tmp_path / "snaps")
    pairs = [snapshot_tensor(tmp_path / "snaps", r, "adapter_state", lay) for r in recs]
    m = train_autoencoder([p[0] for p in pairs], [p[1] for p in pairs], recs, tiny_cfg.autoencoder, device="cpu", out_dir=tmp_path / "ae", provenance={"smoke": True})
    assert all(math.isfinite(c["val_mse"]) for c in m["curve"])
    assert m["compression_ratio_elements"] == 1 / 64 and m["latent_shape"] == [64, 2, 2]
    assert m["latent_logical_bytes_fp32"] == 128 * 128 // 64 * 4
    assert (tmp_path / "ae" / "autoencoder.safetensors").exists() and (tmp_path / "ae" / "train_mean.safetensors").exists()
    # Learning control ("can the training loop fit structured data?") on signed, smooth, scale-varying
    # synthetic matrices with the paper-like ~600-iteration budget. Signed inputs converge markedly slower
    # than positive ones for the reconstructed 1-channel ReLU stem, and validation generalisation from 12
    # samples is numerically noisy across thread counts/seeds (Phase-2 diagnostic, docs/phase2_validation.md).
    # Robust assertions: (i) validation MSE drops >= 3x (observed 5.1-14.6x), (ii) on the TRAINING set the AE
    # reconstructs far better than the input-independent train-mean baseline (observed 0.12-0.26 vs 0.96-1.0).
    g = torch.Generator().manual_seed(0)
    u = torch.linspace(-1, 1, 128)
    xs = [0.3 * torch.tanh(torch.randn(1, generator=g) * torch.outer(u, u)).unsqueeze(0) for _ in range(12)]
    recs_c = [{"time_index": i} for i in range(12)]
    cfg_c = tiny_cfg.autoencoder
    cfg_c.iterations, cfg_c.eval_every, cfg_c.learning_rate = 600, 200, 2e-3
    mc = train_autoencoder(xs, [torch.zeros_like(x) for x in xs], recs_c, cfg_c, device="cpu", out_dir=tmp_path / "ctrl", provenance={"smoke": True})
    assert mc["curve"][-1]["val_mse"] < mc["curve"][0]["val_mse"] / 3
    assert mc["train"]["autoencoder"]["pooled_rel_sq_error"] < 0.5 * mc["train"]["train_mean"]["pooled_rel_sq_error"]
