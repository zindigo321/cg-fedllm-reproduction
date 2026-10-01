"""Phase-3A AutoEncoder viability study: the expanded reconstruction report (A1) and the pre-registered
reconstruction gate (A6). Definitions: docs/phase3_preregistration.md.

Predictors evaluated on the D1 temporal train and validation splits of one TGAP snapshot set:

  autoencoder_best_val  the codec under test: normalise -> encoder -> latent -> decoder -> denormalise,
                        using the best-D1-validation checkpoint (the one the gate and the A7 probe use)
  autoencoder_final     the same with the final-iteration checkpoint (reported, not gated)
  zero                  X_hat = 0
  train_mean            X_hat = mean of the D1 training split (an input-independent decoder)
  identity              X_hat = X (the reconstruction ceiling)
  tanh_range_ceiling    denormalise(clamp(normalise(X), -1, 1)): the best that any decoder ending in Tanh can do
                        under the fitted normalisation (DERIVED; equals ``identity`` when nothing exceeds |1|)
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import torch

from cg_fedllm.compression.autoencoder import ResNetAutoEncoder, load_autoencoder
from cg_fedllm.compression.codecs import AutoEncoderCodec, CodecContext
from cg_fedllm.compression.diagnostics import SnapshotItem, evaluate_predictors
from cg_fedllm.compression.layout import Layout, LoRAGeometry, geometry_from_dict
from cg_fedllm.compression.metrics import json_safe
from cg_fedllm.compression.normalization import Normalizer
from cg_fedllm.compression.representation import to_representation
from cg_fedllm.tgap.snapshots import load_states
from cg_fedllm.tgap.train_ae import split_indices
from cg_fedllm.utils.io import load_tensors

# Reviewer-defined MINIMUM VIABILITY thresholds (A6); not paper reproduction targets.
GATE_A_REL_SQ_ERROR_LT = 1.0
GATE_B_REL_SQ_ERROR_LT = 1.0
GATE_INNOVATION_COSINE_GE = 0.90
GATE_B_NORM_RATIO = (0.5, 2.0)
GATE_AGGREGATE_UPDATE_COSINE_GE = 0.90
PRIORITY = ("none", "global_rms", "factor_rms")
ERROR_ELEMENTS_BUDGET = 2**25  # exact abs-error quantiles while the pooled error vector fits this budget


class _SaturationProbe:
    """Wraps the AE codec: records how close the decoder's Tanh output is to saturation and how far the output
    sits from the train-mean prediction (an input-independent decoder would sit on it)."""

    def __init__(self, codec: AutoEncoderCodec, mean: torch.Tensor) -> None:
        self.codec = codec
        self.mean = mean
        self.n = 0
        self.saturated = 0
        self.dist_out_mean = 0.0
        self.dist_in_mean = 0.0

    def __call__(self, x: torch.Tensor, item: SnapshotItem) -> torch.Tensor:
        ctx = CodecContext(item.time_index, item.client_id, 0)
        payload = self.codec.encode(x, ctx)
        with torch.no_grad():
            y = self.codec.ae.decode(payload.tensors["z"].to(self.codec.device, torch.float32).unsqueeze(0))[0]
        self.n += y.numel()
        self.saturated += int((y.abs() > 0.99).sum())
        x_hat = self.codec.decode(payload, ctx)
        self.dist_out_mean += float(((x_hat.double() - self.mean.double()) ** 2).sum())
        self.dist_in_mean += float(((x.double() - self.mean.double()) ** 2).sum())
        return x_hat

    def summary(self) -> dict[str, Any]:
        return {
            "decoder_output_fraction_abs_gt_0_99": self.saturated / self.n if self.n else None,
            "output_to_train_mean_rel_sq_distance": self.dist_out_mean / self.dist_in_mean if self.dist_in_mean > 0 else None,
        }


def iter_groups(snapshot_dir: Path, records: Sequence[dict], indices: Sequence[int]) -> Iterator[list[SnapshotItem]]:
    """Snapshots ``indices`` one time index at a time (hash-verified, loaded lazily to bound memory)."""
    by_t: dict[int, list[int]] = defaultdict(list)
    for i in indices:
        by_t[int(records[i]["time_index"])].append(i)
    for t in sorted(by_t):
        group = []
        for i in by_t[t]:
            start, end = load_states(snapshot_dir, records[i])
            group.append(SnapshotItem(i, t, int(records[i]["client_id"]), int(records[i]["num_samples"]), start, end))
        yield group


def error_stride(num_snapshots: int, geom: LoRAGeometry) -> int:
    per_factor = geom.num_elements // 2
    return max(1, math.ceil(num_snapshots * per_factor / ERROR_ELEMENTS_BUDGET))


def evaluate_split(
    snapshot_dir: Path,
    records: Sequence[dict],
    indices: Sequence[int],
    *,
    representation: str,
    layout: Layout,
    geom: LoRAGeometry,
    codecs: dict[str, AutoEncoderCodec],
    normalizer: Normalizer,
    mean: torch.Tensor,
) -> dict[str, Any]:
    stride = error_stride(len(indices), geom)
    probes = {name: _SaturationProbe(codec, mean) for name, codec in codecs.items()}
    predictors = {
        **probes,
        "zero": lambda x, it: torch.zeros_like(x),
        "train_mean": lambda x, it: mean.clone(),
        "identity": lambda x, it: x.clone(),
        "tanh_range_ceiling": lambda x, it: normalizer.denormalize(normalizer.normalize(x).clamp(-1.0, 1.0)),
    }
    out = evaluate_predictors(iter_groups(snapshot_dir, records, indices), representation, layout, geom, predictors, error_stride=stride)
    for name, probe in probes.items():
        out[name]["autoencoder_probe"] = probe.summary()
    times = sorted({int(records[i]["time_index"]) for i in indices})
    return {"num_snapshots": len(indices), "time_indices": times, "error_stride": stride, "predictors": out}


def reconstruction_gate(val: dict[str, Any], candidate: str = "autoencoder_best_val") -> dict[str, Any]:
    """A6 minimum-viability gate on the validation split (all criteria must pass)."""
    ae, tm = val["predictors"][candidate], val["predictors"]["train_mean"]
    a_rse = ae["transmitted"]["A"]["rel_sq_error"]
    b_rse = ae["transmitted"]["B"]["rel_sq_error"]
    inn_cos = ae["innovation"]["all"]["cosine"]
    b_ratio = ae["transmitted"]["B"]["norm_ratio"]
    agg_cos = ae["aggregate"]["update"]["all"]["cosine"]
    inn_rse, tm_inn_rse = ae["innovation"]["all"]["rel_sq_error"], tm["innovation"]["all"]["rel_sq_error"]
    sh = ae["shift_control"]
    finite = bool(ae["finite_outputs"]) and all(isinstance(v, float) and math.isfinite(v) for v in (a_rse, b_rse, inn_cos, b_ratio, agg_cos, inn_rse))
    criteria = {
        "finite_outputs": {"pass": finite, "value": bool(ae["finite_outputs"])},
        "A_pooled_rel_sq_error_lt_1": {"pass": a_rse < GATE_A_REL_SQ_ERROR_LT, "value": a_rse, "threshold": f"< {GATE_A_REL_SQ_ERROR_LT}"},
        "B_pooled_rel_sq_error_lt_1": {"pass": b_rse < GATE_B_REL_SQ_ERROR_LT, "value": b_rse, "threshold": f"< {GATE_B_REL_SQ_ERROR_LT}"},
        "innovation_cosine_ge_0_90": {"pass": inn_cos >= GATE_INNOVATION_COSINE_GE, "value": inn_cos, "threshold": f">= {GATE_INNOVATION_COSINE_GE}"},
        "B_norm_ratio_in_0_5_2_0": {"pass": GATE_B_NORM_RATIO[0] <= b_ratio <= GATE_B_NORM_RATIO[1], "value": b_ratio, "threshold": list(GATE_B_NORM_RATIO)},
        "aggregate_update_cosine_ge_0_90": {"pass": agg_cos >= GATE_AGGREGATE_UPDATE_COSINE_GE, "value": agg_cos, "threshold": f">= {GATE_AGGREGATE_UPDATE_COSINE_GE}"},
        "input_dependent": {
            "pass": inn_rse < tm_inn_rse and sh["innovation_rel_sq_error_matched"] < sh["innovation_rel_sq_error_shifted"],
            "value": {
                "innovation_rel_sq_error": inn_rse,
                "train_mean_innovation_rel_sq_error": tm_inn_rse,
                "shift_control_matched": sh["innovation_rel_sq_error_matched"],
                "shift_control_shifted": sh["innovation_rel_sq_error_shifted"],
            },
            "threshold": "innovation rel. sq. error below the train-mean predictor's AND below the within-time-index shifted pairing",
        },
    }
    for c in criteria.values():
        c["pass"] = bool(c["pass"]) if finite or c is criteria["finite_outputs"] else False
    return {
        "candidate": candidate,
        "criteria": criteria,
        "pass": all(c["pass"] for c in criteria.values()),
        "reported_not_gated": {"snr_paper": ae["snr_paper"], "snr_standard_db": ae["snr_standard_db"]},
    }


def select_primary(gates_by_mode: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """A6 priority rule: the least modified passing interpretation (none, then global_rms, then factor_rms)."""
    for mode in PRIORITY:
        if mode in gates_by_mode and gates_by_mode[mode]["pass"]:
            return {"selected": mode, "rule": "first passing mode in the order none -> global_rms -> factor_rms", "viable": True}
    return {"selected": None, "rule": "first passing mode in the order none -> global_rms -> factor_rms", "viable": False, "verdict": "NO PRIMARY CODEC IS VIABLE"}


def run_viability(
    snapshot_dir: Path,
    records: Sequence[dict],
    ae_dir: Path,
    *,
    representation: str,
    layout: Layout,
    split: str,
    val_fraction: float,
    device: torch.device,
    split_seed: int = 0,
) -> dict[str, Any]:
    """Evaluate one trained AE run directory (``autoencoder_best_val`` / ``autoencoder`` checkpoints)."""
    geom = geometry_from_dict(records[0]["geometry"])
    train_idx, val_idx = split_indices(records, split, val_fraction, split_seed)
    codecs: dict[str, AutoEncoderCodec] = {}
    normalizer: Normalizer | None = None
    meta_by: dict[str, Any] = {}
    for name, fname in (("autoencoder_best_val", "autoencoder_best_val.safetensors"), ("autoencoder_final", "autoencoder.safetensors")):
        ae: ResNetAutoEncoder
        ae, meta = load_autoencoder(ae_dir / fname, device=device)
        norm = Normalizer.from_dict(meta.get("normalization"))
        if normalizer is not None and norm != normalizer:
            raise ValueError("best-val and final checkpoints carry different normalisers")
        normalizer = norm
        codecs[name] = AutoEncoderCodec(ae, device=device, normalizer=norm)
        meta_by[name] = {k: meta.get(k) for k in ("checkpoint", "iteration", "val_mse")}
    if list(normalizer.fit.get("train_indices", train_idx)) != list(train_idx):
        raise ValueError("the normaliser was not fitted on this run's training split")
    mean = load_tensors(ae_dir / "train_mean.safetensors")["x"]
    # sanity: the stored train mean is the mean of the training split's transmitted representation
    first = load_states(snapshot_dir, records[train_idx[0]])
    if layout.forward(to_representation(first[1], first[0], representation), geom).shape != mean.shape:
        raise ValueError("train-mean shape does not match the snapshots")
    kw = dict(representation=representation, layout=layout, geom=geom, codecs=codecs, normalizer=normalizer, mean=mean)
    val = evaluate_split(snapshot_dir, records, val_idx, **kw)
    train = evaluate_split(snapshot_dir, records, train_idx, **kw)
    gate = reconstruction_gate(val)
    return json_safe(
        {
            "representation": representation,
            "normalization": normalizer.to_dict(),
            "checkpoints": meta_by,
            "split": {
                "rule": f"{split}: first {1 - val_fraction:.0%} of time indices train, last {val_fraction:.0%} validation",
                "train": [[int(records[i]["time_index"]), int(records[i]["client_id"])] for i in train_idx],
                "val": [[int(records[i]["time_index"]), int(records[i]["client_id"])] for i in val_idx],
            },
            "val": val,
            "train": train,
            "gate": gate,
        }
    )
