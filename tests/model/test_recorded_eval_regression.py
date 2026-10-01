"""Evaluator regression on the real Qwen1.5-0.5B (needs the pinned model + benchmarks in the local HF cache;
never run in CI: ``CGFED_RUN_MODEL_TESTS=1 HF_HUB_OFFLINE=1 pytest -m model``).

Re-scores 36 recorded MMLU-test / C-Eval-val questions (incl. ~3k-token contexts) of the Phase-2 evaluator
validation run and requires identical predictions. Log-probabilities may differ slightly because the batch
composition differs from the full run (padding changes the fp32 reduction order)."""

from __future__ import annotations

import json
from collections import defaultdict

import pytest
import torch

from cg_fedllm.config import load_config
from cg_fedllm.evaluation.benchmarks import ceval_subject_mapping, download_benchmark, load_split
from cg_fedllm.evaluation.prompts import mmlu_display_name
from cg_fedllm.evaluation.scorer import build_requests, score_requests
from cg_fedllm.models.loading import load_model
from tests.conftest import FIXTURES, REPO

pytestmark = [pytest.mark.model, pytest.mark.gpu]

LOGPROB_TOL = 1e-3


def test_qwen05b_reproduces_recorded_evaluator_predictions():
    recorded = json.loads((FIXTURES / "qwen15_0p5b_recorded_eval_subset.json").read_text(encoding="utf-8"))["items"]
    cfg = load_config(REPO / "configs" / "eval" / "qwen15_0p5b_validation.yaml", ["model.local_files_only=true"])
    device = torch.device("cuda")
    loaded = load_model(cfg.model, device)
    wanted: dict[tuple[str, str], set[str]] = defaultdict(set)
    for r in recorded:
        bench, _, split, _ = r["qid"].split("/")
        wanted[(bench, split)].add(r["qid"])
    scored = {}
    for (bench, split), qids in sorted(wanted.items()):
        repo, rev = (cfg.eval.ceval_repo, cfg.eval.ceval_revision) if bench == "ceval" else (cfg.eval.mmlu_repo, cfg.eval.mmlu_revision)
        subjects = sorted({q.split("/")[1] for q in qids})
        root = download_benchmark(repo, rev, bench, sorted({split, "dev"}))
        questions = {s: [q for q in qs if q.qid in qids] for s, qs in load_split(root, bench, split, subjects).items()}
        dev = load_split(root, bench, "dev", subjects)
        display = {s: ceval_subject_mapping()[s][1] for s in subjects} if bench == "ceval" else {s: mmlu_display_name(s) for s in subjects}
        reqs = build_requests(loaded.tokenizer, bench, questions, dev, display, 5, cfg.eval.max_context, True)
        for it in score_requests(loaded.model, reqs, loaded.pad_token_id, device, cfg.eval.max_batch_tokens, cfg.eval.max_batch_size, cfg.eval.max_batch_attention):
            scored[it.qid] = it
    assert set(scored) == {r["qid"] for r in recorded}
    for r in recorded:
        it = scored[r["qid"]]
        assert it.context_tokens == r["ctx_tokens"] and it.num_shots == r["shots"]
        assert it.prediction == r["pred"], r["qid"]
        assert max(abs(a - b) for a, b in zip(it.logprobs, r["logprobs"], strict=True)) < LOGPROB_TOL, r["qid"]
