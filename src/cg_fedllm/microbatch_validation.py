"""Validation of ``virtual_paper_microbatch`` against true physical micro-batches (Phase 4, stage F1).

One optimizer step (exactly ``batch_size`` examples, i.e. one accumulation group) is run from the same start
adapter with the same seeds in two micro-batch modes, and compared on:

* the logged step loss (relative difference);
* the accumulated gradient before clipping (cosine and relative L2, for A, B and all factors);
* the adapter after the optimizer step: relative L2 of the state (``state_rel_l2``) and of the update
  (``update_rel_l2``, the change produced by the step; much stricter).

Pre-registered acceptance with dropout OFF (reviewer F1): loss relative difference <= 1e-6, gradient cosine
>= 0.99999, gradient relative L2 <= 1e-4, post-step adapter relative L2 <= 1e-5. With dropout ON the masks
differ by construction; only reproducibility and the absence of a systematic loss-scale error are checked.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import torch

from cg_fedllm.config import LocalTrainSection
from cg_fedllm.data.formatting import TokenizedExample
from cg_fedllm.federated.client import LocalTrainer, StepObserver
from cg_fedllm.models.adapter import AdapterState, parse_key

ACCEPT = {"loss_rel_diff_max": 1e-6, "grad_cosine_min": 0.99999, "grad_rel_l2_max": 1e-4, "adapter_rel_l2_max": 1e-5}


class GradientRecorder(StepObserver):
    """Keeps a float64 CPU copy of the pre-clip gradient, the clip norm and the step delta of each step."""

    def __init__(self, keep_delta: bool = True) -> None:
        self.grads: list[dict[str, torch.Tensor]] = []
        self.losses: list[float] = []
        self.norms: list[float | None] = []
        self.deltas: list[dict[str, torch.Tensor]] = []
        self.clipped: list[dict[str, torch.Tensor]] = []
        self._keep_delta = keep_delta

    @property
    def wants_step_delta(self) -> bool:
        return self._keep_delta

    def on_gradients(self, step: int, params: dict[str, torch.nn.Parameter], loss: float) -> None:
        self.grads.append({k: (p.grad.detach().to("cpu", torch.float64).clone() if p.grad is not None else torch.zeros(p.shape, dtype=torch.float64)) for k, p in params.items()})
        self.losses.append(loss)

    def on_clipped(self, step: int, params: dict[str, torch.nn.Parameter], total_norm: float | None) -> None:
        self.norms.append(total_norm)
        self.clipped.append({k: (p.grad.detach().to("cpu", torch.float64).clone() if p.grad is not None else torch.zeros(p.shape, dtype=torch.float64)) for k, p in params.items()})

    def on_step(self, step: int, params: dict[str, torch.nn.Parameter], before: dict[str, torch.Tensor]) -> None:
        self.deltas.append({k: (p.detach() - before[k]).to("cpu", torch.float64) for k, p in params.items()})


def _flat(t: dict[str, torch.Tensor], factor: str | None) -> torch.Tensor:
    keys = [k for k in sorted(t) if factor is None or parse_key(k)[3] == factor]
    return torch.cat([t[k].reshape(-1).to(torch.float64) for k in keys])


def _cmp(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> dict[str, Any]:
    out = {}
    for f, name in ((None, "all"), ("A", "A"), ("B", "B")):
        x, y = _flat(a, f), _flat(b, f)
        nx, ny = float(x.norm()), float(y.norm())
        out[name] = {
            "cosine": float(x @ y) / (nx * ny) if nx > 0 and ny > 0 else math.nan,
            "rel_l2": float((x - y).norm()) / ny if ny > 0 else (0.0 if nx == 0 else math.inf),
            "norm_ratio": nx / ny if ny > 0 else math.nan,
        }
    return out


def one_step(trainer: LocalTrainer, start: AdapterState, examples: Sequence[TokenizedExample], seed_keys: tuple) -> dict[str, Any]:
    if len(examples) != trainer.cfg.batch_size:
        raise ValueError("one optimizer step needs exactly batch_size examples")
    rec = GradientRecorder()
    res = trainer.train(start, list(examples), seed_keys, observer=rec)
    if res.num_optimizer_steps != 1:
        raise RuntimeError("expected exactly one optimizer step")
    return {"loss": res.step_losses[0], "grads": rec.grads[0], "clip_norm": rec.norms[0], "end": res.end_state, "start": start, "result": res}


def update_flips(a: dict[str, Any], b: dict[str, Any], lr: float) -> dict[str, Any]:
    """Elements whose first-step AdamW update differs by more than lr/2 between two runs, and how small their
    gradients are: Adam's first step is ~lr * g / (|g| + eps), so near-zero gradients amplify rounding."""
    ua, ub = _flat(a["end"].sub(a["start"]).tensors, None), _flat(b["end"].sub(b["start"]).tensors, None)
    g = _flat(b["grads"], None).abs()
    big = (ua - ub).abs() > 0.5 * lr
    n = int(big.sum())
    out = {"elements": int(ua.numel()), "flipped": n, "flipped_fraction": n / ua.numel()}
    if n:
        gf = g[big]
        out.update({"flipped_grad_abs_max": float(gf.max()), "flipped_grad_abs_median": float(gf.median()), "all_grad_abs_median": float(g.median())})
    return out


def reversed_within_microbatches(examples: Sequence[TokenizedExample], base_seed: int, keys: tuple, micro_batch_size: int) -> list[TokenizedExample]:
    """An example list for which the trainer's own data order yields the SAME micro-batches with their rows reversed
    (mathematically identical objective, different floating-point reduction order): a numerical noise floor."""
    from cg_fedllm.utils.seeding import numpy_rng

    perm = numpy_rng(base_seed, *keys, "data_order").permutation(len(examples))
    out = list(examples)
    for i in range(0, len(perm), micro_batch_size):
        chunk = perm[i : i + micro_batch_size]
        for k, pos in enumerate(chunk):
            out[int(pos)] = examples[int(chunk[len(chunk) - 1 - k])]
    return out


def compare(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Compare run ``a`` against reference ``b``."""
    upd_a, upd_b = a["end"].sub(a["start"]), b["end"].sub(b["start"])
    g = _cmp(a["grads"], b["grads"])
    return {
        "loss_a": a["loss"],
        "loss_b": b["loss"],
        "loss_rel_diff": abs(a["loss"] - b["loss"]) / abs(b["loss"]),
        "clip_norm_a": a["clip_norm"],
        "clip_norm_b": b["clip_norm"],
        "gradient": g,
        "state_rel_l2": a["end"].relative_l2_diff(b["end"]),
        "update_rel_l2": upd_a.relative_l2_diff(upd_b),
        "end_bitwise_equal": a["end"].equal(b["end"]),
    }


def accept(c: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "loss": c["loss_rel_diff"] <= ACCEPT["loss_rel_diff_max"],
        "grad_cosine": c["gradient"]["all"]["cosine"] >= ACCEPT["grad_cosine_min"],
        "grad_rel_l2": c["gradient"]["all"]["rel_l2"] <= ACCEPT["grad_rel_l2_max"],
        "adapter_rel_l2": c["state_rel_l2"] <= ACCEPT["adapter_rel_l2_max"],
    }
    return {"checks": checks, "pass": all(checks.values()), "thresholds": ACCEPT}


def trainer_with(trainer: LocalTrainer, **changes: Any) -> LocalTrainer:
    """A LocalTrainer sharing model/params but with modified local-training settings."""
    cfg: LocalTrainSection = replace(trainer.cfg, **changes)
    cfg.validate("local_train")
    return LocalTrainer(trainer.model, trainer.params, cfg, pad_token_id=trainer.pad_token_id, padding_side=trainer.padding_side, device=trainer.device, base_seed=trainer.base_seed)


def lora_dropouts(model) -> list[torch.nn.Module]:
    return [m for name, m in model.named_modules() if name.endswith("lora_dropout.default") and isinstance(m, torch.nn.Dropout)]


class dropout_set:
    """Context manager: set every LoRA dropout probability to ``p`` (restored on exit)."""

    def __init__(self, model, p: float) -> None:
        self.mods = lora_dropouts(model)
        self.p = p
        self.saved: list[float] = []

    def __enter__(self):
        self.saved = [m.p for m in self.mods]
        for m in self.mods:
            m.p = self.p
        return self

    def __exit__(self, *exc):
        for m, p in zip(self.mods, self.saved):
            m.p = p
        return False


class blas_library:
    """Context manager selecting the CUDA BLAS backend (``cublas``/``cublaslt``) for a kernel-implementation control."""

    def __init__(self, name: str | None) -> None:
        self.name = name
        self.saved = None

    def __enter__(self):
        if self.name:
            self.saved = torch.backends.cuda.preferred_blas_library()
            torch.backends.cuda.preferred_blas_library(self.name)
        return self

    def __exit__(self, *exc):
        if self.name:
            torch.backends.cuda.preferred_blas_library(self.saved)
        return False


def mode_blas(spec: str) -> str | None:
    return spec.partition("@")[2] or None


def parse_mode(spec: str) -> tuple[str, dict[str, Any]]:
    """``physical:16`` (micro-batch 16, physical) or ``virtual:2`` (logical micro-batch from the config, chunk 2);
    an optional ``@cublaslt`` suffix runs the mode with another BLAS implementation (same math)."""
    kind, _, n = spec.partition("@")[0].partition(":")
    if kind == "physical":
        return spec, {"microbatch_mode": "physical_microbatch", "micro_batch_size": int(n), "physical_chunk_size": None}
    if kind == "virtual":
        return spec, {"microbatch_mode": "virtual_paper_microbatch", "physical_chunk_size": int(n)}
    raise ValueError(f"unknown mode {spec!r}")


def _timed_step(trainer: LocalTrainer, start: AdapterState, batch, keys) -> tuple[dict[str, Any], dict[str, Any]]:
    import time

    cuda = torch.cuda.is_available() and trainer.device.type == "cuda"
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    r = one_step(trainer, start, batch, keys)
    if cuda:
        torch.cuda.synchronize()
    stats = {"wall_s": round(time.perf_counter() - t0, 3), "padded_tokens": r["result"].num_padded_tokens, "label_tokens": r["result"].num_label_tokens,
             "forward_chunks": r["result"].num_forward_chunks}
    if cuda:
        stats.update({"peak_allocated_bytes": int(torch.cuda.max_memory_allocated()), "peak_reserved_bytes": int(torch.cuda.max_memory_reserved())})
    return r, stats


def run_validation(trainer: LocalTrainer, start: AdapterState, batches: Sequence[Sequence[TokenizedExample]], modes: Sequence[str], *, dropout_batches: int = 4) -> dict[str, Any]:
    """Dropout OFF: every mode vs the first one on ``batches[0]``. Dropout ON (the configured p): reproducibility of
    each mode and the per-batch loss ratio of every other mode to the first one on up to ``dropout_batches`` batches."""
    parsed = [parse_mode(m) for m in modes]
    trainers = {name: trainer_with(trainer, **changes) for name, changes in parsed}
    ref_name = parsed[0][0]
    out: dict[str, Any] = {"reference_mode": ref_name, "modes": list(modes), "dropout_off": {}, "dropout_on": {}}
    lr = trainer.cfg.learning_rate
    with dropout_set(trainer.model, 0.0):
        runs = {}
        for name, _ in parsed:
            with blas_library(mode_blas(name)):
                runs[name], stats = _timed_step(trainers[name], start, batches[0], ("validate", 0, 0))
            out["dropout_off"][name] = {"stats": stats, "loss": runs[name]["loss"], "clip_norm": runs[name]["clip_norm"]}
        for name, _ in parsed[1:]:
            c = compare(runs[name], runs[ref_name])
            out["dropout_off"][name]["vs_reference"] = c | {"acceptance": accept(c), "update_flips": update_flips(runs[name], runs[ref_name], lr)}
        names = [n for n, _ in parsed]
        out["pairwise"] = {}
        for i in range(1, len(names)):
            for j in range(i + 1, len(names)):
                c = compare(runs[names[i]], runs[names[j]])
                out["pairwise"][f"{names[i]} vs {names[j]}"] = {
                    "loss_rel_diff": c["loss_rel_diff"], "grad_cosine": c["gradient"]["all"]["cosine"], "grad_rel_l2": c["gradient"]["all"]["rel_l2"],
                    "state_rel_l2": c["state_rel_l2"], "update_rel_l2": c["update_rel_l2"], "update_flips": update_flips(runs[names[i]], runs[names[j]], lr)["flipped"],
                }
        # numerical noise floor: the reference mode on the same micro-batches with rows reversed
        ref_tr = trainers[ref_name]
        rev = reversed_within_microbatches(batches[0], ref_tr.base_seed, ("validate", 0, 0), ref_tr.cfg.micro_batch_size)
        floor = one_step(ref_tr, start, rev, ("validate", 0, 0))
        c = compare(floor, runs[ref_name])
        out["noise_floor"] = {
            "description": f"{ref_name} with the rows of every micro-batch reversed (same objective, different reduction order) vs {ref_name}",
            **c,
            "acceptance": accept(c),
            "update_flips": update_flips(floor, runs[ref_name], lr),
        }
        del runs, floor
    p_on = [m.p for m in lora_dropouts(trainer.model)]
    out["dropout_on"]["lora_dropout_p"] = sorted(set(p_on))
    reps = {}
    for name, _ in parsed:
        with blas_library(mode_blas(name)):
            a = one_step(trainers[name], start, batches[0], ("validate_do", 0, 0))
            b = one_step(trainers[name], start, batches[0], ("validate_do", 0, 0))
        reps[name] = {"bitwise_reproducible": a["end"].equal(b["end"]), "loss": a["loss"]}
    out["dropout_on"]["reproducibility"] = reps
    ratios: dict[str, list[float]] = {name: [] for name, _ in parsed[1:]}
    for k, batch in enumerate(batches[:dropout_batches]):
        with blas_library(mode_blas(ref_name)):
            ref = one_step(trainers[ref_name], start, batch, ("validate_do", k, 1))
        for name, _ in parsed[1:]:
            with blas_library(mode_blas(name)):
                ratios[name].append(one_step(trainers[name], start, batch, ("validate_do", k, 1))["loss"] / ref["loss"])
    out["dropout_on"]["loss_ratio_to_reference"] = {
        name: {"per_batch": r, "mean": sum(r) / len(r), "max_abs_dev": max(abs(x - 1) for x in r)} for name, r in ratios.items() if r
    }
    return out
