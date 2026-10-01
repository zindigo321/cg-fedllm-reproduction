"""Local LoRA training primitive (one client, one round).

BEHAVIOR RECONSTRUCTED FROM SHEPHERD / HF Trainer (``fed_utils/client.py`` @ bcffa00 with the
``transformers.Trainer`` defaults it relied on), made explicit:

* a *fresh* ``torch.optim.AdamW`` (betas, eps, weight decay from config) for every client invocation --
  optimizer state never crosses rounds (Shepherd builds a new Trainer per client per round);
* linear learning-rate decay to 0 over the round's optimizer steps with ``warmup_steps`` warm-up
  (identical formula to ``get_linear_schedule_with_warmup``);
* gradient clipping at ``max_grad_norm`` (Trainer default 1.0), gradient accumulation of
  ``batch_size / micro_batch_size`` micro-batches;
* ``loss = token-mean cross-entropy per micro-batch``; the accumulated gradient is the MEAN over the
  micro-batches of an accumulation group, and a partial final group is flushed at the epoch end
  (explicit choices; the 2023 Trainer divided by the nominal accumulation count instead).

Deviation from Shepherd: data order and dropout RNG are derived from ``(seed, *seed_keys)`` instead of
from global RNG state, which makes resumed runs bit-identical (documented in docs/deviations.md).

Micro-batch modes (``local_train.microbatch_mode``):

* ``physical_microbatch`` (Phase 2/3): one forward/backward per micro-batch.
* ``virtual_paper_microbatch`` (Phase 4): each LOGICAL micro-batch of ``micro_batch_size`` examples is
  collated exactly as a real batch would be (same padding length, labels, attention mask, left-padding label
  behaviour) and then streamed through the model in chunks of ``physical_chunk_size`` rows. Each chunk's
  summed token loss is back-propagated with scale ``1 / (T_logical * G)`` (T_logical = label tokens of the
  logical micro-batch, G = micro-batches in the accumulation group), so the summed gradient equals the
  physical path's ``mean_token_loss / G`` up to floating-point order. Dropout masks differ (different RNG
  call shapes); with dropout off the two modes agree to rounding.

An optional observer sees the accumulated gradient before and after clipping and the parameter change of
every optimizer step; it only reads tensors, so it cannot change the optimisation.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from cg_fedllm.config import LocalTrainSection
from cg_fedllm.data.formatting import IGNORE_INDEX, TokenizedExample, collate
from cg_fedllm.models.adapter import AdapterState
from cg_fedllm.models.lora import get_adapter_state, set_adapter_state
from cg_fedllm.utils.seeding import derive_seed, numpy_rng


def linear_schedule_factor(step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    return max(0.0, float(total_steps - step) / float(max(1, total_steps - warmup_steps)))


def optimizer_steps(num_examples: int, cfg: LocalTrainSection) -> int:
    accum = cfg.batch_size // cfg.micro_batch_size
    micro_batches = math.ceil(num_examples / cfg.micro_batch_size)
    return math.ceil(micro_batches / accum) * cfg.epochs


class StepObserver:
    """Read-only hooks around each optimizer step (Phase-4 gradient forensics)."""

    def on_gradients(self, step: int, params: dict[str, torch.nn.Parameter], loss: float) -> None:  # before clipping
        pass

    def on_clipped(self, step: int, params: dict[str, torch.nn.Parameter], total_norm: float | None) -> None:
        pass

    def on_step(self, step: int, params: dict[str, torch.nn.Parameter], before: dict[str, torch.Tensor]) -> None:
        pass

    @property
    def wants_step_delta(self) -> bool:
        return False


@dataclass
class LocalTrainResult:
    end_state: AdapterState
    start_hash: str
    end_hash: str
    num_samples: int
    num_optimizer_steps: int
    num_micro_batches: int
    num_label_tokens: int
    step_losses: list[float] = field(default_factory=list)
    wall_time_s: float = 0.0
    num_padded_tokens: int = 0
    num_forward_chunks: int = 0

    @property
    def mean_loss(self) -> float:
        return sum(self.step_losses) / len(self.step_losses) if self.step_losses else float("nan")

    def summary(self) -> dict:
        return {
            "num_samples": self.num_samples,
            "num_optimizer_steps": self.num_optimizer_steps,
            "num_micro_batches": self.num_micro_batches,
            "num_label_tokens": self.num_label_tokens,
            "mean_loss": self.mean_loss,
            "first_step_loss": self.step_losses[0] if self.step_losses else None,
            "last_step_loss": self.step_losses[-1] if self.step_losses else None,
            "start_hash": self.start_hash,
            "end_hash": self.end_hash,
            "wall_time_s": round(self.wall_time_s, 3),
            "num_padded_tokens": self.num_padded_tokens,
            "num_forward_chunks": self.num_forward_chunks,
        }


class LocalTrainer:
    def __init__(
        self,
        model,
        params: dict[str, torch.nn.Parameter],
        cfg: LocalTrainSection,
        *,
        pad_token_id: int,
        padding_side: str,
        device: torch.device | str,
        base_seed: int,
    ) -> None:
        self.model = model
        self.params = params
        self.cfg = cfg
        self.pad_token_id = pad_token_id
        self.padding_side = padding_side
        self.device = torch.device(device)
        self.base_seed = base_seed

    def _loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, int]:
        batch = {k: v.to(self.device) for k, v in batch.items()}
        out = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], position_ids=batch["position_ids"])
        logits = out.logits[:, :-1, :].float()
        labels = batch["labels"][:, 1:]
        n_tokens = int((labels != IGNORE_INDEX).sum())
        if n_tokens == 0:
            raise ValueError("micro-batch has no label tokens")
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=IGNORE_INDEX, reduction="mean")
        return loss, n_tokens

    def _sum_loss(self, chunk: dict[str, torch.Tensor]) -> torch.Tensor:
        chunk = {k: v.to(self.device) for k, v in chunk.items()}
        out = self.model(input_ids=chunk["input_ids"], attention_mask=chunk["attention_mask"], position_ids=chunk["position_ids"])
        logits = out.logits[:, :-1, :].float()
        labels = chunk["labels"][:, 1:]
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=IGNORE_INDEX, reduction="sum")

    def _virtual_backward(self, batch: dict[str, torch.Tensor], group_size: int) -> tuple[float, int, int]:
        """One logical micro-batch, physically streamed in row chunks; returns (token-mean loss, label tokens, chunks)."""
        n_tokens = int((batch["labels"][:, 1:] != IGNORE_INDEX).sum())
        if n_tokens == 0:
            raise ValueError("micro-batch has no label tokens")
        rows, chunk = batch["input_ids"].shape[0], int(self.cfg.physical_chunk_size)
        total, n_chunks = 0.0, 0
        for i in range(0, rows, chunk):
            loss_sum = self._sum_loss({k: v[i : i + chunk] for k, v in batch.items()})
            (loss_sum / (n_tokens * group_size)).backward()
            total += float(loss_sum.detach())
            n_chunks += 1
        return total / n_tokens, n_tokens, n_chunks

    def train(
        self,
        start_state: AdapterState,
        examples: Sequence[TokenizedExample],
        seed_keys: tuple[int | str, ...],
        observer: StepObserver | None = None,
    ) -> LocalTrainResult:
        """Train the LoRA factors starting from ``start_state``; return the post-training state."""
        if not examples:
            raise ValueError("a client needs at least one example")
        cfg = self.cfg
        t0 = time.time()
        set_adapter_state(self.params, start_state)
        self.model.train()
        accum = cfg.batch_size // cfg.micro_batch_size
        total_steps = optimizer_steps(len(examples), cfg)
        params = list(self.params.values())
        opt = torch.optim.AdamW(
            params,
            lr=cfg.learning_rate,
            betas=(cfg.adam_beta1, cfg.adam_beta2),
            eps=cfg.adam_epsilon,
            weight_decay=cfg.weight_decay,
            foreach=False,
        )
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: linear_schedule_factor(s, total_steps, cfg.warmup_steps))
        torch_seed = derive_seed(self.base_seed, *seed_keys, "torch")
        torch.manual_seed(torch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(torch_seed)
        order_rng = numpy_rng(self.base_seed, *seed_keys, "data_order")
        step_losses: list[float] = []
        n_micro = 0
        n_tokens = 0
        n_padded = 0
        n_chunks = 0
        virtual = cfg.microbatch_mode == "virtual_paper_microbatch"
        for _epoch in range(cfg.epochs):
            perm = order_rng.permutation(len(examples))
            micro = [perm[i : i + cfg.micro_batch_size] for i in range(0, len(examples), cfg.micro_batch_size)]
            for g in range(0, len(micro), accum):
                group = micro[g : g + accum]
                opt.zero_grad(set_to_none=True)
                group_loss = 0.0
                for mb in group:
                    batch = collate([examples[int(j)] for j in mb], self.pad_token_id, self.padding_side, cfg.pad_to_multiple_of)
                    if virtual:
                        mean_loss, toks, chunks = self._virtual_backward(batch, len(group))
                        group_loss += mean_loss / len(group)
                        n_chunks += chunks
                    else:
                        loss, toks = self._loss(batch)
                        (loss / len(group)).backward()
                        group_loss += float(loss.detach()) / len(group)
                        n_chunks += 1
                    n_micro += 1
                    n_tokens += toks
                    n_padded += int(batch["input_ids"].numel())
                if not math.isfinite(group_loss):
                    raise FloatingPointError(f"non-finite training loss at step {len(step_losses)}")
                step = len(step_losses)
                if observer is not None:
                    observer.on_gradients(step, self.params, group_loss)
                total_norm = None
                if cfg.max_grad_norm is not None:
                    total_norm = float(torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm))
                before = None
                if observer is not None:
                    observer.on_clipped(step, self.params, total_norm)
                    if observer.wants_step_delta:
                        before = {k: p.detach().clone() for k, p in self.params.items()}
                opt.step()
                sched.step()
                if before is not None:
                    observer.on_step(step, self.params, before)
                step_losses.append(group_loss)
        end_state = get_adapter_state(self.params)
        del opt, sched
        return LocalTrainResult(
            end_state=end_state,
            start_hash=start_state.sha256(),
            end_hash=end_state.sha256(),
            num_samples=len(examples),
            num_optimizer_steps=len(step_losses),
            num_micro_batches=n_micro,
            num_label_tokens=n_tokens,
            step_losses=step_losses,
            wall_time_s=time.time() - t0,
            num_padded_tokens=n_padded,
            num_forward_chunks=n_chunks,
        )
