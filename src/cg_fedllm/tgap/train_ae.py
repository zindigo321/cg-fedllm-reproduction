"""AutoEncoder training on TGAP snapshots.

Paper-specified: reconstruction loss = MSE (squared L2), optimizer Adam, learning rate 2e-4.
UNKNOWN in the paper (all configurable, defaults recorded in docs/deviations.md): batch size, iteration
budget, betas, train/validation split, input normalisation, stopping rule.

* ``autoencoder.normalization`` (Phase 3, A2): ``none`` (paper-literal), ``global_rms`` or ``factor_rms``;
  the scales are fitted on the TRAINING split only and stored in the checkpoint metadata. The AE is trained
  and validated in the normalised space; reconstructions are always reported in the original space.
* ``autoencoder.checkpoint_policy``: ``final`` (Phase 2) or ``final_and_best_val`` (Phase 3, D9), which
  also keeps the AE with the lowest D1-validation MSE among the evaluations at multiples of ``eval_every``
  (ties keep the earliest). D1 validation is compressor-training data: no D2 or benchmark result is used.

Metrics are computed per snapshot in eval mode (BatchNorm running statistics) for the AE and for two
trivial reconstruction baselines: ``zero`` (X_hat = 0) and ``train_mean`` (X_hat = mean of the training
snapshots, i.e. an input-independent decoder). The factor-aware Phase-3 report is
:mod:`cg_fedllm.tgap.viability`.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from cg_fedllm.compression.autoencoder import ResNetAutoEncoder, config_from_section, save_autoencoder
from cg_fedllm.compression.codecs import AutoEncoderCodec, CodecContext
from cg_fedllm.compression.layout import Layout, LoRAGeometry
from cg_fedllm.compression.metrics import json_safe, reconstruction_metrics
from cg_fedllm.compression.normalization import Normalizer, fit_normalizer, range_report
from cg_fedllm.config import RESULT_LABELS, AESection, ResultLabel
from cg_fedllm.utils.io import atomic_write_json, save_tensors_atomic
from cg_fedllm.utils.seeding import derive_seed, numpy_rng


def split_indices(records: Sequence[dict], split: str, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    n = len(records)
    if n < 2:
        raise ValueError("need at least two snapshots to form train and validation sets")
    if split == "temporal":
        times = sorted({int(r["time_index"]) for r in records})
        n_val_t = max(1, math.ceil(val_fraction * len(times)))
        if n_val_t >= len(times):
            raise ValueError("temporal split needs at least two distinct time indices")
        val_times = set(times[-n_val_t:])
        val = [i for i, r in enumerate(records) if int(r["time_index"]) in val_times]
    elif split == "random":
        n_val = max(1, int(round(val_fraction * n)))
        val = sorted(int(i) for i in numpy_rng(seed, "ae_split").permutation(n)[:n_val])
    else:
        raise ValueError(f"unknown split {split!r}")
    train = [i for i in range(n) if i not in set(val)]
    if not train:
        raise ValueError("empty training split")
    return train, val


def _eval_set(
    ae: ResNetAutoEncoder,
    xs: Sequence[torch.Tensor],
    refs: Sequence[torch.Tensor],
    idx: Sequence[int],
    mean: torch.Tensor,
    device,
    normalizer: Normalizer | None = None,
) -> dict[str, Any]:
    codec = AutoEncoderCodec(ae, device=device, normalizer=normalizer)
    per = {"autoencoder": [], "zero": [], "train_mean": []}
    for i in idx:
        x, ref = xs[i], refs[i]
        ctx = CodecContext(0, i, 0)
        x_hat = codec.decode(codec.encode(x, ctx), ctx)
        per["autoencoder"].append(reconstruction_metrics(x, x_hat, ref))
        per["zero"].append(reconstruction_metrics(x, torch.zeros_like(x), ref))
        per["train_mean"].append(reconstruction_metrics(x, mean, ref))
    out: dict[str, Any] = {}
    for name, rows in per.items():
        keys = ("mse", "rel_sq_error", "snr_paper", "snr_standard", "snr_db", "cosine", "innovation_ratio", "delta_cosine")
        agg = {}
        for k in keys:
            vals = [r[k] for r in rows if k in r and isinstance(r[k], float) and math.isfinite(r[k])]
            agg[f"mean_{k}"] = sum(vals) / len(vals) if vals else None
        # pooled (energy-weighted) versions
        sse = sum(r["sse"] for r in rows)
        sig = sum(r["signal_sq"] for r in rows)
        agg["pooled_rel_sq_error"] = sse / sig if sig > 0 else None
        agg["pooled_snr_db"] = 10 * math.log10(sig / sse) if sse > 0 and sig > 0 else None
        agg["n"] = len(rows)
        out[name] = agg
    return out


def train_autoencoder(
    xs: Sequence[torch.Tensor],
    refs: Sequence[torch.Tensor],
    records: Sequence[dict],
    cfg: AESection,
    *,
    device: str | torch.device,
    out_dir: Path,
    provenance: dict[str, Any],
    label: ResultLabel = "UNKNOWN",
    layout: Layout | None = None,
    geom: LoRAGeometry | None = None,
) -> dict[str, Any]:
    """Train the ResNet AE on ``xs`` (each ``[1, d, W]``) and write checkpoint(s) + metrics to ``out_dir``."""
    if label not in RESULT_LABELS:
        raise ValueError(f"result label must be one of {RESULT_LABELS}, got {label!r}")
    out_dir = Path(out_dir)
    device = torch.device(device)
    train_idx, val_idx = split_indices(records, cfg.split, cfg.val_fraction, cfg.split_seed)
    shapes = {tuple(x.shape) for x in xs}
    if len(shapes) != 1:
        raise ValueError(f"all snapshots must share one shape, got {shapes}")
    height, width = xs[0].shape[-2], xs[0].shape[-1]
    normalizer = fit_normalizer(xs, train_idx, cfg.normalization, layout=layout, geom=geom)
    xs_n = xs if normalizer.is_identity else [normalizer.normalize(x) for x in xs]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(derive_seed(cfg.init_seed, "ae_init"))
        ae = ResNetAutoEncoder(config_from_section(cfg))
    ae.check_input_hw(height, width)
    ae.to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=cfg.learning_rate, betas=(cfg.adam_beta1, cfg.adam_beta2), eps=cfg.adam_epsilon, weight_decay=cfg.weight_decay)
    stack = torch.stack([xs_n[i] for i in train_idx])  # [N, 1, d, W] on CPU, in the (normalised) training space
    # the train-mean predictor always lives in the original space
    mean = stack.mean(dim=0) if normalizer.is_identity else torch.stack([xs[i] for i in train_idx]).mean(dim=0)
    order_rng = numpy_rng(cfg.init_seed, "ae_batches")
    curve: list[dict[str, Any]] = []
    perm: list[int] = []
    keep_best = cfg.checkpoint_policy == "final_and_best_val"
    best: dict[str, Any] = {"iteration": None, "val_mse": math.inf, "state": None}
    t0 = time.time()
    bs = min(cfg.batch_size, len(train_idx))
    for it in range(1, cfg.iterations + 1):
        ae.train()
        if len(perm) < bs:
            perm = [int(j) for j in order_rng.permutation(len(train_idx))]
        batch_ids, perm = perm[:bs], perm[bs:]
        x = stack[batch_ids].to(device)
        x_hat, _ = ae(x)
        loss = F.mse_loss(x_hat, x)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite AE loss at iteration {it}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if it == 1 or it % cfg.eval_every == 0 or it == cfg.iterations:
            ae.eval()
            with torch.no_grad():
                val = [F.mse_loss(ae(xs_n[i].unsqueeze(0).to(device))[0], xs_n[i].unsqueeze(0).to(device)).item() for i in val_idx]
            val_mse = sum(val) / len(val)
            curve.append({"iteration": it, "train_mse_batch": float(loss.item()), "val_mse": val_mse})
            if keep_best and it % cfg.eval_every == 0 and val_mse < best["val_mse"]:
                best = {"iteration": it, "val_mse": val_mse, "state": {k: v.detach().to("cpu").clone() for k, v in ae.state_dict().items()}}
    train_time = time.time() - t0
    ae.eval()
    meta = {**provenance, "height": height, "width": width, "normalization": normalizer.to_dict()}
    ckpt = out_dir / "autoencoder.safetensors"
    save_autoencoder(ckpt, ae.cpu(), {**meta, "checkpoint": "final", "iteration": cfg.iterations})
    ae.to(device)
    save_tensors_atomic(out_dir / "train_mean.safetensors", {"x": mean}, {"role": "tgap_train_mean", "representation": cfg.representation, "layout": cfg.layout})
    latent = ae.latent_shape(height, width)
    latent_numel = latent[0] * latent[1] * latent[2]
    codec = AutoEncoderCodec(ae, device=device, normalizer=normalizer)
    probe = codec.encode(xs[train_idx[0]], CodecContext(0, 0, 0))
    metrics: dict[str, Any] = {
        "label": label,
        "num_snapshots": len(xs),
        "train_indices": train_idx,
        "val_indices": val_idx,
        "input_shape": [1, height, width],
        "latent_shape": list(latent),
        "input_numel": height * width,
        "latent_numel": latent_numel,
        "compression_ratio_elements": latent_numel / (height * width),
        "latent_logical_bytes_fp32": probe.logical_nbytes(),
        "latent_serialized_bytes": probe.serialized_nbytes(),
        "raw_logical_bytes_fp32": height * width * 4,
        "normalization_uplink_bytes": 0,
        "parameters": ae.num_parameters(),
        "train_time_s": round(train_time, 2),
        "curve": curve,
        "normalization": normalizer.to_dict(),
        "train": _eval_set(ae, xs, refs, train_idx, mean, device, normalizer),
        "val": _eval_set(ae, xs, refs, val_idx, mean, device, normalizer),
        "config": {k: getattr(cfg, k) for k in cfg.__dataclass_fields__},
        "provenance": provenance,
    }
    if layout is not None and geom is not None:
        metrics["normalized_range"] = {
            "train": range_report(xs, train_idx, normalizer, layout, geom),
            "val": range_report(xs, val_idx, normalizer, layout, geom),
        }
    if keep_best:
        best_ae = ResNetAutoEncoder(config_from_section(cfg))
        best_ae.load_state_dict(best["state"], strict=True)
        best_ae.eval()
        save_autoencoder(
            out_dir / "autoencoder_best_val.safetensors", best_ae, {**meta, "checkpoint": "best_val", "iteration": best["iteration"], "val_mse": best["val_mse"]}
        )
        metrics["best_val"] = {
            "iteration": best["iteration"],
            "val_mse": best["val_mse"],
            "selection": f"lowest D1-validation MSE (training space) at multiples of eval_every={cfg.eval_every}; ties keep the earliest",
            "train": _eval_set(best_ae, xs, refs, train_idx, mean, device, normalizer),
            "val": _eval_set(best_ae, xs, refs, val_idx, mean, device, normalizer),
        }
    metrics = json_safe(metrics)
    atomic_write_json(out_dir / "ae_metrics.json", metrics)
    return metrics
