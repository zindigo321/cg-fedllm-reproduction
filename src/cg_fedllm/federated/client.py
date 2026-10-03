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
        out = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            position_ids=batch["position_ids"],
        )
        logits = out.logits[:, :-1, :].float()
        labels = batch["labels"][:, 1:]
        n_tokens = int((labels != IGNORE_INDEX).sum())
        if n_tokens == 0:
            raise ValueError("micro-batch has no label tokens")
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=IGNORE_INDEX,
            reduction="mean",
        )
        return loss, n_tokens

    def train(
        self,
        start_state: AdapterState,
        examples: Sequence[TokenizedExample],
        seed_keys: tuple[int | str, ...],
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
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: linear_schedule_factor(s, total_steps, cfg.warmup_steps)
        )
        torch_seed = derive_seed(self.base_seed, *seed_keys, "torch")
        torch.manual_seed(torch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(torch_seed)
        order_rng = numpy_rng(self.base_seed, *seed_keys, "data_order")
        step_losses: list[float] = []
        n_micro = 0
        n_tokens = 0
        for _epoch in range(cfg.epochs):
            perm = order_rng.permutation(len(examples))
            micro = [
                perm[i : i + cfg.micro_batch_size] for i in range(0, len(examples), cfg.micro_batch_size)
            ]
            for g in range(0, len(micro), accum):
                group = micro[g : g + accum]
                opt.zero_grad(set_to_none=True)
                group_loss = 0.0
                for mb in group:
                    batch = collate(
                        [examples[int(j)] for j in mb],
                        self.pad_token_id,
                        self.padding_side,
                        cfg.pad_to_multiple_of,
                    )
                    loss, toks = self._loss(batch)
                    (loss / len(group)).backward()
                    group_loss += float(loss.detach()) / len(group)
                    n_micro += 1
                    n_tokens += toks
                if not math.isfinite(group_loss):
                    raise FloatingPointError(f"non-finite training loss at step {len(step_losses)}")
                if cfg.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                opt.step()
                sched.step()
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
        )
