"""Bounded GPU feasibility micro-benchmarks (LOCAL-MICROBENCH; no FL/TGAP/evaluation).

For each (model, dtype/quantization, gradient checkpointing, micro-batch) configuration we run a few
LoRA training steps on synthetic token sequences of fixed length using the *same* loss computation as
:class:`cg_fedllm.federated.client.LocalTrainer` (full-vocabulary logits cast to float32), and record
peak allocated/reserved CUDA memory and training throughput. Out-of-memory is recorded, not hidden.

On Windows (WDDM) device allocations beyond dedicated VRAM silently spill into shared system memory, which
makes an oversized configuration look feasible but run >10x slower; ``cgfed bench-gpu`` therefore caps the
allocator first (:func:`cg_fedllm.utils.gpu.cap_allocator_to_free_vram`), so such configurations fail with
a recorded OOM.
"""

from __future__ import annotations

import gc
import time
from dataclasses import replace
from typing import Any

import torch

from cg_fedllm.config import LocalTrainSection, LoRASection, ModelSection
from cg_fedllm.federated.client import LocalTrainer
from cg_fedllm.models.loading import load_model
from cg_fedllm.models.lora import attach_lora, lora_parameters


def _cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def bench_config(
    model_cfg: ModelSection,
    lora_cfg: LoRASection,
    *,
    seq_len: int,
    micro_batches: list[int],
    gradient_checkpointing: bool,
    steps: int = 5,
    warmup: int = 2,
    seed: int = 0,
) -> list[dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("GPU benchmark requires CUDA")
    device = torch.device("cuda")
    _cleanup()
    torch.cuda.reset_peak_memory_stats()
    t_load = time.perf_counter()
    # no tokenizer is needed (synthetic ids, no padding); the pad id is irrelevant but required by the loader
    pad = 0 if model_cfg.pad_token_id is None else model_cfg.pad_token_id
    loaded = load_model(
        replace(model_cfg, gradient_checkpointing=gradient_checkpointing, pad_token_id=pad),
        device,
        with_tokenizer=False,
    )
    model = loaded.model
    if gradient_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    peft_model = attach_lora(model, lora_cfg)
    params = lora_parameters(peft_model)
    load_s = time.perf_counter() - t_load
    torch.cuda.synchronize()
    weights_alloc = torch.cuda.memory_allocated()
    vocab = int(peft_model.get_input_embeddings().weight.shape[0])
    trainer = LocalTrainer(
        peft_model,
        params,
        LocalTrainSection(
            batch_size=1, micro_batch_size=1, precision="bf16" if model_cfg.dtype == "bfloat16" else "fp32"
        ),
        pad_token_id=0,
        padding_side="left",
        device=device,
        base_seed=seed,
    )
    results = []
    for mb in micro_batches:
        _cleanup()
        torch.cuda.reset_peak_memory_stats()
        g = torch.Generator().manual_seed(seed)
        ids = torch.randint(10, vocab - 10, (mb, seq_len), generator=g)
        batch = {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "labels": ids.clone(),
            "position_ids": torch.arange(seq_len).unsqueeze(0).expand(mb, -1).contiguous(),
        }
        opt = torch.optim.AdamW(list(params.values()), lr=1e-4, foreach=False)
        rec: dict[str, Any] = {
            "model": model_cfg.id or "tiny",
            "revision": model_cfg.revision,
            "dtype": model_cfg.dtype,
            "quantization": model_cfg.quantization,
            "attn_implementation": model_cfg.attn_implementation,
            "gradient_checkpointing": gradient_checkpointing,
            "seq_len": seq_len,
            "micro_batch": mb,
            "lora": {"r": lora_cfg.r, "alpha": lora_cfg.alpha, "targets": list(lora_cfg.target_modules)},
            "trainable_params": int(sum(p.numel() for p in params.values())),
            "weights_allocated_bytes": int(weights_alloc),
            "model_load_s": round(load_s, 2),
            "label": "LOCAL-MICROBENCH",
        }
        try:
            peft_model.train()
            times = []
            for i in range(warmup + steps):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                loss, _ = trainer._loss(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(params.values()), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                if i >= warmup:
                    times.append(time.perf_counter() - t0)
            step_s = sum(times) / len(times)
            rec.update(
                {
                    "status": "ok",
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                    "step_time_s": round(step_s, 4),
                    "tokens_per_s": round(mb * seq_len / step_s, 1),
                    "final_loss": float(loss.detach()),
                }
            )
        except torch.OutOfMemoryError as exc:
            rec.update(
                {
                    "status": "OOM",
                    "error": str(exc).splitlines()[0][:200],
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                }
            )
        finally:
            del opt
            _cleanup()
        results.append(rec)
    del trainer, params, peft_model, model, loaded
    _cleanup()
    return results
