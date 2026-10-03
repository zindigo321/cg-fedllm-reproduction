"""Phase-3 resource and loss-normalisation calibration (PHASE3-DIAGNOSTIC).

* :func:`token_length_report` -- token lengths after the cutoff, and the exact padded-token totals that the
  trainer's real data order produces for a client schedule at micro-batch 1 and 2.
* :func:`timing_run` -- bounded real training: the actual :class:`LocalTrainer` on actual Dolly client data
  (one client round per listed client) plus a forced worst-case micro-batch of the longest training examples,
  with peak allocated/reserved CUDA memory, step time and throughput.
* :func:`micro_batch_decision` -- the pre-registered A3 rule: switch the primary configuration to
  micro-batch 1 if training exceeds 6.3 GiB allocated or repeatedly comes within 256 MiB of the allocator cap.
* :func:`microbatch_gradient_diagnostic` -- how the trainer's loss normalisation (token-mean per micro-batch,
  mean over the accumulation group) weights examples for different micro-batch sizes on one fixed batch. Every
  decomposition is emulated exactly from per-example token-sum gradients of the *unpadded* examples, so the
  emulated differences are due to the normalisation alone (LoRA dropout is disabled for the measurement only).
  The real trainer path (actual collation) is computed alongside: with left padding the last pad position of a
  padded example predicts its first real token (inherited HF/Shepherd label-shift behaviour), so the real
  objective has one extra label token per left-padded example; that difference is reported, not changed.
  Nothing here changes how training normalises the loss.
"""

from __future__ import annotations

import gc
import math
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from cg_fedllm.config import LocalTrainSection
from cg_fedllm.data.formatting import IGNORE_INDEX, TokenizedExample, collate
from cg_fedllm.federated.client import LocalTrainer, optimizer_steps
from cg_fedllm.models.adapter import AdapterState, parse_key
from cg_fedllm.models.lora import set_adapter_state
from cg_fedllm.utils.seeding import numpy_rng

GiB = 2**30
ALLOCATED_LIMIT_BYTES = int(6.3 * GiB)
CAP_APPROACH_BYTES = 256 * 2**20


def _q(a: np.ndarray) -> dict[str, float]:
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": int(a.max()),
    }


def padded_tokens(
    examples: Sequence[TokenizedExample],
    cfg: LocalTrainSection,
    seed: int,
    seed_keys: tuple,
    micro_batch: int,
) -> tuple[int, int]:
    """(real, padded) tokens of one local epoch in the trainer's data order (same permutation as LocalTrainer)."""
    perm = numpy_rng(seed, *seed_keys, "data_order").permutation(len(examples))
    real = padded = 0
    for i in range(0, len(examples), micro_batch):
        mb = [examples[int(j)] for j in perm[i : i + micro_batch]]
        length = max(ex.num_tokens for ex in mb)
        if cfg.pad_to_multiple_of:
            length = math.ceil(length / cfg.pad_to_multiple_of) * cfg.pad_to_multiple_of
        real += sum(ex.num_tokens for ex in mb)
        padded += length * len(mb)
    return real, padded


def token_length_report(
    splits: dict[str, Sequence[Sequence[TokenizedExample]]],
    schedules: dict[str, tuple[str, list[tuple[int, list[int]]]]],
    *,
    cutoff: int,
    cfg: LocalTrainSection,
    seed: int,
) -> dict[str, Any]:
    """``splits``: name -> per-client examples; ``schedules``: name -> (split, [(round, clients)]) with the
    simulator's seed namespace = name."""
    out: dict[str, Any] = {"cutoff_len": cutoff, "splits": {}, "schedules": {}}
    for name, clients in splits.items():
        a = np.asarray([ex.num_tokens for c in clients for ex in c], dtype=np.int64)
        out["splits"][name] = {**_q(a), "fraction_at_cutoff": float((a >= cutoff).mean())}
    for name, (split, rounds) in schedules.items():
        clients = splits[split]
        a = np.asarray(
            [ex.num_tokens for t, cids in rounds for c in cids for ex in clients[c]], dtype=np.int64
        )
        row: dict[str, Any] = {
            "split": split,
            "rounds": len(rounds),
            "client_rounds": sum(len(c) for _, c in rounds),
            "examples": int(a.size),
            "lengths": _q(a),
        }
        steps = 0
        for mb in (1, 2):
            real = pad = 0
            for t, cids in rounds:
                for c in cids:
                    r, p = padded_tokens(clients[c], cfg, seed, (name, t, c), mb)
                    real, pad = real + r, pad + p
            row[f"micro_batch_{mb}"] = {
                "real_tokens": real,
                "padded_tokens": pad,
                "padding_overhead": pad / real if real else None,
            }
        for _t, cids in rounds:
            for c in cids:
                steps += optimizer_steps(len(clients[c]), cfg)
        row["optimizer_steps"] = steps
        out["schedules"][name] = row
    return out


def _cuda_reset() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def timing_run(
    trainer: LocalTrainer,
    start: AdapterState,
    client_examples: Sequence[Sequence[TokenizedExample]],
    *,
    clients: Sequence[int],
    namespace: str,
    round_index: int,
    worst_case: Sequence[TokenizedExample],
    allocator_cap_bytes: int | None,
) -> dict[str, Any]:
    """One real client round per listed client (fresh optimizer, real seeds) + one forced worst-case step."""
    cfg = trainer.cfg
    rows = []
    for cid in clients:
        ex = client_examples[cid]
        _cuda_reset()
        t0 = time.perf_counter()
        res = trainer.train(start, ex, (namespace, round_index, cid))
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        real, padded = padded_tokens(
            ex, cfg, trainer.base_seed, (namespace, round_index, cid), cfg.micro_batch_size
        )
        rows.append(
            {
                "client_id": int(cid),
                "examples": len(ex),
                "optimizer_steps": res.num_optimizer_steps,
                "micro_batches": res.num_micro_batches,
                "real_tokens": real,
                "padded_tokens": padded,
                "max_example_tokens": max(e.num_tokens for e in ex),
                "wall_s": round(wall, 3),
                "step_time_s": round(wall / res.num_optimizer_steps, 3),
                "padded_tokens_per_s": round(padded / wall, 1),
                "real_tokens_per_s": round(real / wall, 1),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                "finite": all(math.isfinite(x) for x in res.step_losses),
                "mean_loss": res.mean_loss,
            }
        )
    _cuda_reset()
    t0 = time.perf_counter()
    res = trainer.train(start, list(worst_case), ("calibration_worst_case", 0, 0))
    torch.cuda.synchronize()
    wc = {
        "examples": len(worst_case),
        "example_tokens": [e.num_tokens for e in worst_case],
        "micro_batches": res.num_micro_batches,
        "wall_s": round(time.perf_counter() - t0, 3),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "finite": all(math.isfinite(x) for x in res.step_losses),
    }
    tot_wall = sum(r["wall_s"] for r in rows)
    out = {
        "micro_batch_size": cfg.micro_batch_size,
        "batch_size": cfg.batch_size,
        "gradient_accumulation": cfg.batch_size // cfg.micro_batch_size,
        "clients": rows,
        "worst_case_micro_batch": wc,
        "totals": {
            "wall_s": round(tot_wall, 2),
            "optimizer_steps": sum(r["optimizer_steps"] for r in rows),
            "padded_tokens_per_s": round(sum(r["padded_tokens"] for r in rows) / tot_wall, 1),
            "real_tokens_per_s": round(sum(r["real_tokens"] for r in rows) / tot_wall, 1),
            "mean_step_time_s": round(tot_wall / max(1, sum(r["optimizer_steps"] for r in rows)), 3),
            "peak_allocated_bytes": max(
                [r["peak_allocated_bytes"] for r in rows] + [wc["peak_allocated_bytes"]]
            ),
            "peak_reserved_bytes": max(
                [r["peak_reserved_bytes"] for r in rows] + [wc["peak_reserved_bytes"]]
            ),
        },
    }
    out["decision"] = micro_batch_decision(
        [r["peak_allocated_bytes"] for r in rows] + [wc["peak_allocated_bytes"]],
        [r["peak_reserved_bytes"] for r in rows] + [wc["peak_reserved_bytes"]],
        allocator_cap_bytes,
    )
    return out


def micro_batch_decision(
    peak_allocated: Sequence[int], peak_reserved: Sequence[int], cap_bytes: int | None
) -> dict[str, Any]:
    """Pre-registered A3 rule (docs/phase3_preregistration.md)."""
    over = max(peak_allocated) > ALLOCATED_LIMIT_BYTES
    near = sum(1 for r in peak_reserved if cap_bytes is not None and r >= cap_bytes - CAP_APPROACH_BYTES)
    switch = over or near >= 2
    return {
        "rule": "switch to micro-batch 1 if peak allocated > 6.3 GiB or peak reserved within 256 MiB of the allocator cap in >= 2 measurements",
        "max_peak_allocated_gib": round(max(peak_allocated) / GiB, 3),
        "measurements_near_cap": near,
        "allocator_cap_gib": round(cap_bytes / GiB, 3) if cap_bytes else None,
        "headroom_to_cap_gib": round((cap_bytes - max(peak_reserved)) / GiB, 3) if cap_bytes else None,
        "switch_to_micro_batch_1": switch,
    }


def _lora_dropouts(model) -> list[torch.nn.Module]:
    return [
        m
        for name, m in model.named_modules()
        if name.endswith("lora_dropout.default") and isinstance(m, torch.nn.Dropout)
    ]


def _grads(params: dict[str, torch.nn.Parameter]) -> dict[str, torch.Tensor]:
    return {
        k: (
            p.grad.detach().to("cpu", torch.float64).clone()
            if p.grad is not None
            else torch.zeros(p.shape, dtype=torch.float64)
        )
        for k, p in params.items()
    }


def _flat(g: dict[str, torch.Tensor], factor: str | None) -> torch.Tensor:
    return torch.cat([g[k].reshape(-1) for k in sorted(g) if factor is None or parse_key(k)[3] == factor])


def _cmp(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> dict[str, Any]:
    out = {}
    for f, name in ((None, "all"), ("A", "A"), ("B", "B")):
        x, y = _flat(a, f), _flat(b, f)
        nx, ny = float(x.norm()), float(y.norm())
        out[name] = {
            "cosine": float(x @ y) / (nx * ny) if nx > 0 and ny > 0 else math.nan,
            "rel_l2_diff": float((x - y).norm()) / ny if ny > 0 else math.nan,
            "norm_ratio": nx / ny if ny > 0 else math.nan,
        }
    return out


def microbatch_gradient_diagnostic(
    trainer: LocalTrainer,
    start: AdapterState,
    batch: Sequence[TokenizedExample],
    *,
    decompositions: Sequence[int] = (1, 2, 4, 8, 16, 32),
    real_micro_batch: int = 2,
    reference: int = 2,
) -> dict[str, Any]:
    model, params = trainer.model, trainer.params
    set_adapter_state(params, start)
    model.train()
    drops = _lora_dropouts(model)
    saved_p = [d.p for d in drops]
    for d in drops:
        d.p = 0.0
    try:
        per_example: list[dict[str, torch.Tensor]] = []
        n_tok: list[int] = []
        for ex in batch:
            for p in params.values():
                p.grad = None
            b = {
                k: v.to(trainer.device)
                for k, v in collate([ex], trainer.pad_token_id, trainer.padding_side, None).items()
            }  # no padding
            logits = (
                model(
                    input_ids=b["input_ids"],
                    attention_mask=b["attention_mask"],
                    position_ids=b["position_ids"],
                )
                .logits[:, :-1, :]
                .float()
            )
            labels = b["labels"][:, 1:]
            loss_sum = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=IGNORE_INDEX,
                reduction="sum",
            )
            loss_sum.backward()
            per_example.append(_grads(params))
            n_tok.append(int((labels != IGNORE_INDEX).sum()))
            del logits, loss_sum
        keys = sorted(per_example[0])

        def emulate(m: int) -> dict[str, torch.Tensor]:
            groups = [list(range(i, min(i + m, len(batch)))) for i in range(0, len(batch), m)]
            out = {k: torch.zeros_like(per_example[0][k]) for k in keys}
            for g in groups:
                tok = sum(n_tok[i] for i in g)
                for k in keys:
                    out[k] += sum(per_example[i][k] for i in g) / tok / len(groups)
            return out

        emulated = {m: emulate(m) for m in decompositions if len(batch) % m == 0}
        # the real trainer path: token-mean loss per micro-batch, (loss / G).backward() accumulated over the group
        for p in params.values():
            p.grad = None
        groups = [batch[i : i + real_micro_batch] for i in range(0, len(batch), real_micro_batch)]
        padded_examples = 0
        for mb in groups:
            coll = collate(
                list(mb), trainer.pad_token_id, trainer.padding_side, trainer.cfg.pad_to_multiple_of
            )
            padded_examples += int((coll["attention_mask"][:, 0] == 0).sum())
            loss, _ = trainer._loss(coll)
            (loss / len(groups)).backward()
        real = _grads(params)
        for p in params.values():
            p.grad = None
    finally:
        for d, p in zip(drops, saved_p):
            d.p = p
    token_mean = emulated[len(batch)] if len(batch) in emulated else None
    weights = {}
    for m in emulated:
        groups_m = [list(range(i, i + m)) for i in range(0, len(batch), m)]
        w = np.zeros(len(batch))
        for g in groups_m:
            tok = sum(n_tok[i] for i in g)
            for i in g:
                w[i] = n_tok[i] / tok / len(groups_m)  # total weight of example i's mean token loss
        weights[str(m)] = {
            "min": float(w.min()),
            "max": float(w.max()),
            "max_over_min": float(w.max() / w.min()),
        }
    return {
        "batch_examples": len(batch),
        "label_tokens_per_example": n_tok,
        "lora_dropout_disabled_for_measurement": bool(drops),
        "real_vs_emulated_micro_batch": {
            "micro_batch": real_micro_batch,
            "left_padded_examples": padded_examples,
            "note": "real trainer collation vs unpadded emulation: differs by one first-token label term per left-padded example and by kernel numerics",
            **_cmp(real, emulated[real_micro_batch]),
        },
        f"vs_micro_batch_{reference}": {str(m): _cmp(g, emulated[reference]) for m, g in emulated.items()},
        "vs_token_mean_full_batch": {str(m): _cmp(g, token_mean) for m, g in emulated.items()}
        if token_mean is not None
        else None,
        "per_example_weight_spread": weights,
        "note": "gradients before clipping and before AdamW; per-coordinate Adam normalisation and clipping further transform the update",
    }
