"""Cross-check the reference_eval_v1 MMLU scorer against lm-evaluation-harness (hard gate: |delta| <= 0.5 pp).

Usage (after ``cgfed evaluate`` produced ``mmlu_test_5shot.json`` + predictions for the same model):

    python scripts/crosscheck_mmlu_lmeval.py --ours <run_dir>/mmlu_test_5shot.json \
        --model-path <local HF snapshot dir> --dtype float32 --out results/phase2/mmlu_lmeval_crosscheck.json

Matched configuration: lm-eval task ``mmlu`` (cais/mmlu), 5-shot (first_n from dev), same weights/dtype,
answer-choice log-likelihood; overall accuracy is question-weighted in both tools.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from cg_fedllm.utils.io import atomic_write_json, read_jsonl


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ours", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--batch-size", default="auto:4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tolerance-pp", type=float, default=0.5)
    args = ap.parse_args()

    import lm_eval
    from huggingface_hub import HfApi
    from lm_eval import simple_evaluate

    ours = json.loads(Path(args.ours).read_text(encoding="utf-8"))
    preds_path = Path(args.ours).with_name(Path(args.ours).stem + "_predictions.jsonl")
    ours_q = {r["qid"]: (r["pred"] == r["answer"]) for r in read_jsonl(preds_path)}

    t0 = time.time()
    res = simple_evaluate(
        model="hf",
        model_args=f"pretrained={args.model_path},dtype={args.dtype}",
        tasks=["mmlu"],
        num_fewshot=5,
        batch_size=args.batch_size,
        device="cuda:0",
        log_samples=True,
        random_seed=0,
        numpy_random_seed=0,
        torch_random_seed=0,
        fewshot_random_seed=0,
    )
    elapsed = time.time() - t0

    lm_overall = res["results"]["mmlu"]["acc,none"]
    lm_subject, agree, total = {}, 0, 0
    for task, samples in res["samples"].items():
        subject = task.removeprefix("mmlu_")
        lm_subject[subject] = res["results"][task]["acc,none"]
        for s in samples:
            qid = f"mmlu/{subject}/test/{s['doc_id']}"
            if qid in ours_q:
                total += 1
                agree += int(bool(s["acc"]) == ours_q[qid])
    ours_subject = {s: v["acc"] for s, v in ours["aggregates"]["per_subject"].items()}
    diffs = {s: 100 * (ours_subject[s] - lm_subject[s]) for s in ours_subject if s in lm_subject}
    delta_pp = 100 * (ours["aggregates"]["overall"] - lm_overall)
    out = {
        "label": "LOCAL-EVALUATOR-VALIDATION",
        "gate": f"|ours - lm_eval| <= {args.tolerance_pp} percentage points (overall MMLU test, 5-shot)",
        "ours_overall": ours["aggregates"]["overall"],
        "lm_eval_overall": lm_overall,
        "delta_pp": delta_pp,
        "pass": abs(delta_pp) <= args.tolerance_pp,
        "per_subject_max_abs_delta_pp": max(abs(v) for v in diffs.values()) if diffs else None,
        "per_subject_delta_pp": diffs,
        "question_level_agreement": agree / total if total else None,
        "questions_compared": total,
        "lm_eval": {
            "version": getattr(lm_eval, "__version__", None),
            "task": "mmlu",
            "num_fewshot": 5,
            "model_args": f"pretrained={args.model_path},dtype={args.dtype}",
            "batch_size": args.batch_size,
            "cais_mmlu_main_sha_at_run": HfApi().dataset_info("cais/mmlu").sha,
            "elapsed_s": round(elapsed, 1),
        },
        "ours": {"protocol": ours["protocol"], "dataset": {k: ours["dataset"][k] for k in ("repo", "revision", "digest")}, "timing_s": ours["timing_s"]},
    }
    atomic_write_json(Path(args.out), out)
    print(json.dumps({k: out[k] for k in ("ours_overall", "lm_eval_overall", "delta_pp", "pass", "question_level_agreement", "per_subject_max_abs_delta_pp")}, indent=2))


if __name__ == "__main__":
    main()
