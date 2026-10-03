"""Held-out language-modelling loss (the Phase-2 primary training-quality metric, reviewer R12).

The loss is the token-weighted mean cross-entropy over all label tokens of the held-out set (Shepherd's
80-record ``global_test`` holdout, or a configured prefix of it), computed in float32, in manifest order.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from cg_fedllm.data.formatting import IGNORE_INDEX, TokenizedExample, collate


@torch.no_grad()
def heldout_loss(
    model,
    examples: Sequence[TokenizedExample],
    *,
    micro_batch_size: int,
    pad_token_id: int,
    padding_side: str,
    device: torch.device | str,
    pad_to_multiple_of: int | None = 8,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    loss_sum = 0.0
    tokens = 0
    for start in range(0, len(examples), micro_batch_size):
        batch = collate(
            examples[start : start + micro_batch_size], pad_token_id, padding_side, pad_to_multiple_of
        )
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            position_ids=batch["position_ids"],
        ).logits
        shift_logits = logits[:, :-1, :].float()
        shift_labels = batch["labels"][:, 1:]
        loss_sum += float(
            F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                ignore_index=IGNORE_INDEX,
                reduction="sum",
            )
        )
        tokens += int((shift_labels != IGNORE_INDEX).sum())
    if was_training:
        model.train()
    return {"loss": loss_sum / max(tokens, 1), "tokens": float(tokens), "num_examples": float(len(examples))}
