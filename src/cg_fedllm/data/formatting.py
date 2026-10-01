"""Instruction formatting, tokenisation and batch collation for LoRA training.

BEHAVIOR RECONSTRUCTED FROM SHEPHERD (``main.py`` + ``utils/prompter.py`` @ bcffa00): the Alpaca
prompt template (text supplied by the config, originally from alpaca-lora, Apache-2.0), truncation to
``cutoff_len``, an EOS token appended when there is room, labels equal to the input IDs when
``train_on_inputs`` (Shepherd's default) and ``DataCollatorForSeq2Seq``-style padding (inputs padded
with the pad ID, labels with -100, lengths rounded up to ``pad_to_multiple_of``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from cg_fedllm.config import DataSection, PromptTemplate
from cg_fedllm.data.dolly import DollyRecord

IGNORE_INDEX = -100


class TokenizerLike(Protocol):
    eos_token_id: int | None

    def __call__(self, text: str, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class TokenizedExample:
    source_id: int
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]

    @property
    def num_tokens(self) -> int:
        return len(self.input_ids)


def render_prompt(template: PromptTemplate, instruction: str, context: str, response: str = "") -> str:
    """Shepherd ``Prompter.generate_prompt``: input template iff the context is non-empty, then the label."""
    if context:
        text = template.prompt_input.format(instruction=instruction, input=context)
    else:
        text = template.prompt_no_input.format(instruction=instruction)
    return f"{text}{response}" if response else text


def _encode(tokenizer: TokenizerLike, text: str, cutoff_len: int) -> list[int]:
    out = tokenizer(text, truncation=True, max_length=cutoff_len, padding=False, return_tensors=None)
    return list(out["input_ids"])


def tokenize_record(tokenizer: TokenizerLike, record: DollyRecord, cfg: DataSection) -> TokenizedExample:
    full = render_prompt(cfg.prompt_template, record.instruction, record.context, record.response)
    ids = _encode(tokenizer, full, cfg.cutoff_len)
    eos = tokenizer.eos_token_id
    if cfg.add_eos_token and eos is not None and (not ids or ids[-1] != eos) and len(ids) < cfg.cutoff_len:
        ids.append(eos)
    labels = list(ids)
    if not cfg.train_on_inputs:
        user = render_prompt(cfg.prompt_template, record.instruction, record.context)
        n_user = len(_encode(tokenizer, user, cfg.cutoff_len))
        labels = [IGNORE_INDEX] * min(n_user, len(ids)) + labels[n_user:]
    return TokenizedExample(record.source_id, tuple(ids), tuple(labels))


def collate(
    batch: Sequence[TokenizedExample],
    pad_token_id: int,
    padding_side: str = "left",
    pad_to_multiple_of: int | None = 8,
) -> dict[str, torch.Tensor]:
    """Pad a micro-batch. Also returns explicit ``position_ids`` that ignore padding."""
    if not batch:
        raise ValueError("empty batch")
    length = max(ex.num_tokens for ex in batch)
    if pad_to_multiple_of:
        length = ((length + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
    input_ids = torch.full((len(batch), length), pad_token_id, dtype=torch.long)
    labels = torch.full((len(batch), length), IGNORE_INDEX, dtype=torch.long)
    attention = torch.zeros((len(batch), length), dtype=torch.long)
    for i, ex in enumerate(batch):
        n = ex.num_tokens
        sl = slice(length - n, length) if padding_side == "left" else slice(0, n)
        input_ids[i, sl] = torch.tensor(ex.input_ids, dtype=torch.long)
        labels[i, sl] = torch.tensor(ex.labels, dtype=torch.long)
        attention[i, sl] = 1
    position_ids = (attention.cumsum(dim=1) - 1).clamp(min=0)
    return {"input_ids": input_ids, "attention_mask": attention, "labels": labels, "position_ids": position_ids}
