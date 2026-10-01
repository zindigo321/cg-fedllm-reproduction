"""Answer-choice log-likelihood scoring (``reference_eval_v1``).

Scoring semantics mirror lm-evaluation-harness ``multiple_choice`` / ``loglikelihood`` exactly:

* ``context_ids = enc(context)`` and ``continuation_ids = enc(context + continuation)[len(context_ids):]``;
* score(choice) = sum of log-probabilities of ``continuation_ids`` conditioned on ``context_ids``;
* prediction = first argmax over the four choices.

When every continuation is a single token (the normal case for "A".."D" / " A".." D") the four scores are
read from ONE forward pass over the context using only the last position's logits
(``logits_to_keep=1``); otherwise a full-sequence fallback is used. Batches are left-padded with explicit
``position_ids``; all log-softmax computations are done in float32.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from cg_fedllm.evaluation.benchmarks import LETTERS, MCQuestion
from cg_fedllm.evaluation.prompts import CONTINUATIONS, build_prompt


@dataclass
class ScoreRequest:
    qid: str
    subject: str
    answer: str
    num_shots: int
    context_ids: list[int]
    continuation_ids: list[list[int]]
    truncated: bool = False


@dataclass
class ScoredItem:
    qid: str
    subject: str
    answer: str
    prediction: str
    correct: bool
    logprobs: list[float]
    num_shots: int
    context_tokens: int
    path: str
    truncated: bool = False
    extra: dict = field(default_factory=dict)


def _encode_batch(tokenizer, texts: list[str], add_special_tokens: bool) -> list[list[int]]:
    out = tokenizer(texts, add_special_tokens=add_special_tokens, padding=False, truncation=False)
    return [list(x) for x in out["input_ids"]]


def build_requests(
    tokenizer,
    benchmark: str,
    questions: dict[str, list[MCQuestion]],
    dev: dict[str, list[MCQuestion]],
    display_names: dict[str, str],
    num_shots: int,
    max_context: int | None,
    add_special_tokens: bool = True,
) -> list[ScoreRequest]:
    """Build scoring requests; shots are dropped from the end only if a prompt exceeds ``max_context``."""
    conts = CONTINUATIONS[benchmark]
    requests: list[ScoreRequest] = []
    for subject in sorted(questions):
        shots_all = dev[subject][:num_shots]
        for q in questions[subject]:
            k = len(shots_all)
            while True:
                prompt = build_prompt(benchmark, display_names[subject], shots_all[:k], q)
                enc = _encode_batch(tokenizer, [prompt] + [prompt + c for c in conts], add_special_tokens)
                ctx, wholes = enc[0], enc[1:]
                cont_ids = [w[len(ctx) :] for w in wholes]
                if any(len(c) == 0 for c in cont_ids):
                    raise ValueError(f"{q.qid}: empty continuation encoding")
                need = len(ctx) + max(len(c) for c in cont_ids) - 1
                if max_context is None or need <= max_context or k == 0:
                    truncated = False
                    if max_context is not None and need > max_context:
                        keep = max_context - (max(len(c) for c in cont_ids) - 1)
                        ctx = ctx[-keep:]
                        truncated = True
                    requests.append(ScoreRequest(q.qid, subject, q.answer, k, ctx, cont_ids, truncated))
                    break
                k -= 1
    return requests


def _batches(
    indices: list[int], lengths: list[int], max_batch_tokens: int, max_batch_size: int, max_batch_attention: int | None = None
) -> list[list[int]]:
    """Greedy batching of length-sorted requests under three budgets: batch size, padded tokens (B * L_max)
    and attention-matrix elements (B * L_max^2). The last bound matters because padded batches use a
    materialised attention mask whose memory grows with L^2 (on Windows/WDDM an overflow silently spills to
    shared system memory instead of raising OOM, which slows scoring by orders of magnitude)."""
    batches, cur, cur_max = [], [], 0
    for i in indices:
        new_max = max(cur_max, lengths[i])
        n = len(cur) + 1
        over = n > max_batch_size or new_max * n > max_batch_tokens
        if max_batch_attention is not None:
            over = over or new_max * new_max * n > max_batch_attention
        if cur and over:
            batches.append(cur)
            cur, new_max = [], lengths[i]
        cur.append(i)
        cur_max = new_max
    if cur:
        batches.append(cur)
    return batches


def batch_plan(requests: list[ScoreRequest], max_batch_tokens: int = 16384, max_batch_size: int = 32, max_batch_attention: int | None = None) -> dict[str, int]:
    """The forward passes ``score_requests`` will run (same order and budgets), as cost statistics; no model needed."""
    fast = [i for i, r in enumerate(requests) if all(len(c) == 1 for c in r.continuation_ids)]
    lengths = [len(r.context_ids) for r in requests]
    batches = _batches(sorted(fast, key=lambda i: (-lengths[i], i)), lengths, max_batch_tokens, max_batch_size, max_batch_attention)
    slow = [r for r in requests if not all(len(c) == 1 for c in r.continuation_ids)]
    return {
        "questions": len(requests),
        "context_tokens": sum(lengths),
        "single_token_batches": len(batches),
        "padded_tokens": sum(len(b) * max(lengths[i] for i in b) for b in batches),
        "attention_elements": sum(len(b) * max(lengths[i] for i in b) ** 2 for b in batches),
        "multi_token_requests": len(slow),
        "multi_token_forward_tokens": sum(len(r.context_ids) + len(c) - 1 for r in slow for c in r.continuation_ids),
    }


def _left_pad(seqs: Sequence[Sequence[int]], pad_id: int, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    length = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), length), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), length), dtype=torch.long)
    for r, s in enumerate(seqs):
        ids[r, length - len(s) :] = torch.tensor(s, dtype=torch.long)
        mask[r, length - len(s) :] = 1
    pos = (mask.cumsum(dim=1) - 1).clamp(min=0)
    return ids.to(device), mask.to(device), pos.to(device)


@torch.no_grad()
def score_requests(
    model,
    requests: list[ScoreRequest],
    pad_token_id: int,
    device: torch.device | str,
    max_batch_tokens: int = 16384,
    max_batch_size: int = 32,
    max_batch_attention: int | None = None,
) -> list[ScoredItem]:
    model.eval()
    results: dict[int, ScoredItem] = {}
    fast = [i for i, r in enumerate(requests) if all(len(c) == 1 for c in r.continuation_ids)]
    fast_set = set(fast)
    slow = [i for i in range(len(requests)) if i not in fast_set]
    lengths = [len(r.context_ids) for r in requests]
    order = sorted(fast, key=lambda i: (-lengths[i], i))
    for batch in _batches(order, lengths, max_batch_tokens, max_batch_size, max_batch_attention):
        ids, mask, pos = _left_pad([requests[i].context_ids for i in batch], pad_token_id, device)
        # no KV cache: scoring is a single forward, and a returned cache (192 KiB/token for Qwen1.5-1.8B bf16) would
        # stay alive in ``out`` during the next batch's forward
        out = model(input_ids=ids, attention_mask=mask, position_ids=pos, logits_to_keep=1, use_cache=False)
        logp = torch.log_softmax(out.logits[:, -1, :].float(), dim=-1).cpu()
        for row, i in enumerate(batch):
            r = requests[i]
            lps = [float(logp[row, c[0]]) for c in r.continuation_ids]
            results[i] = _item(r, lps, "single_token")
    for i in slow:
        r = requests[i]
        lps = []
        for cont in r.continuation_ids:
            seq = r.context_ids + cont
            ids, mask, pos = _left_pad([seq[:-1]], pad_token_id, device)
            out = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False)
            logp = torch.log_softmax(out.logits[0].float(), dim=-1).cpu()
            start = len(r.context_ids) - 1
            lps.append(float(sum(logp[start + j, tok] for j, tok in enumerate(cont))))
        results[i] = _item(r, lps, "multi_token")
    return [results[i] for i in range(len(requests))]


def _item(r: ScoreRequest, lps: list[float], path: str) -> ScoredItem:
    best = max(range(len(lps)), key=lambda j: (lps[j], -j))  # first argmax on ties
    pred = LETTERS[best]
    return ScoredItem(r.qid, r.subject, r.answer, pred, pred == r.answer, lps, r.num_shots, len(r.context_ids), path, r.truncated)
