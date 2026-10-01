"""Orchestration of ``reference_eval_v1`` for one benchmark specification."""

from __future__ import annotations

import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from cg_fedllm.config import RESULT_LABELS, BenchmarkSpec, EvalSection, ResultLabel
from cg_fedllm.evaluation.aggregate import aggregate_ceval, aggregate_mmlu
from cg_fedllm.evaluation.benchmarks import ceval_subject_mapping, data_digest, download_benchmark, load_split
from cg_fedllm.evaluation.prompts import PROTOCOL_ID, mmlu_display_name
from cg_fedllm.evaluation.scorer import build_requests, score_requests
from cg_fedllm.utils.io import append_jsonl, atomic_write_json
from cg_fedllm.utils.seeding import numpy_rng


def stratified_subset(questions: dict[str, list], fraction: float, seed: int, *keys: str) -> dict[str, list]:
    """ceil(fraction * n) questions per subject, seeded per subject, in dataset order (independent of other subjects)."""
    out = {}
    for subject, qs in questions.items():
        k = min(len(qs), max(1, math.ceil(fraction * len(qs))))
        keep = sorted(int(i) for i in numpy_rng(seed, "eval_subset", *keys, subject).permutation(len(qs))[:k])
        out[subject] = [qs[i] for i in keep]
    return out


def evaluate_benchmark(
    model,
    tokenizer,
    spec: BenchmarkSpec,
    eval_cfg: EvalSection,
    *,
    device,
    pad_token_id: int,
    out_dir: Path | None = None,
    add_special_tokens: bool = True,
    tag: str = "",
    label: ResultLabel = "UNKNOWN",
) -> dict[str, Any]:
    if label not in RESULT_LABELS:
        raise ValueError(f"result label must be one of {RESULT_LABELS}, got {label!r}")
    repo, rev = (eval_cfg.ceval_repo, eval_cfg.ceval_revision) if spec.name == "ceval" else (eval_cfg.mmlu_repo, eval_cfg.mmlu_revision)
    t0 = time.time()
    root = download_benchmark(repo, rev, spec.name, sorted({spec.split, "dev"}))
    questions = load_split(root, spec.name, spec.split, spec.subjects)
    dev = load_split(root, spec.name, "dev", sorted(questions))
    if spec.limit_per_subject is not None:
        questions = {s: qs[: spec.limit_per_subject] for s, qs in questions.items()}
    if spec.subset_fraction is not None:
        questions = stratified_subset(questions, spec.subset_fraction, spec.subset_seed, spec.name, spec.split)
    if spec.name == "ceval":
        mapping = ceval_subject_mapping()
        display = {s: mapping[s][1] for s in questions}
    else:
        display = {s: mmlu_display_name(s) for s in questions}
    requests = build_requests(tokenizer, spec.name, questions, dev, display, spec.num_shots, eval_cfg.max_context, add_special_tokens)
    t1 = time.time()
    items = score_requests(model, requests, pad_token_id, device, eval_cfg.max_batch_tokens, eval_cfg.max_batch_size, eval_cfg.max_batch_attention)
    t2 = time.time()
    agg = aggregate_ceval(items) if spec.name == "ceval" else aggregate_mmlu(items)
    result = {
        "protocol": PROTOCOL_ID,
        "label": label,  # model/adapter provenance: run_metadata.json of the run directory
        "tag": tag,
        "benchmark": asdict(spec),
        "dataset": {"repo": repo, "revision": rev, **data_digest(root, spec.name, sorted({spec.split, "dev"}))},
        "tokenization": {"add_special_tokens": add_special_tokens, "max_context": eval_cfg.max_context},
        "scoring": {
            "method": "lm-eval-compatible loglikelihood of answer continuations; argmax",
            "single_token_requests": sum(1 for it in items if it.path == "single_token"),
            "multi_token_requests": sum(1 for it in items if it.path == "multi_token"),
            "truncated_requests": sum(1 for it in items if it.truncated),
            "reduced_shot_requests": sum(1 for it in items if it.num_shots < spec.num_shots),
        },
        "timing_s": {"prepare": round(t1 - t0, 2), "score": round(t2 - t1, 2)},
        "aggregates": agg,
    }
    if out_dir is not None:
        stem = f"{spec.name}_{spec.split}_{spec.num_shots}shot{('_' + tag) if tag else ''}"
        atomic_write_json(Path(out_dir) / f"{stem}.json", result)
        if eval_cfg.save_predictions:
            pred_path = Path(out_dir) / f"{stem}_predictions.jsonl"
            pred_path.unlink(missing_ok=True)
            for it in items:
                append_jsonl(pred_path, {"qid": it.qid, "answer": it.answer, "pred": it.prediction, "logprobs": it.logprobs, "shots": it.num_shots, "ctx_tokens": it.context_tokens, "path": it.path})
    return result
