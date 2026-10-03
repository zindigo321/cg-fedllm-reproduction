"""Cross-check the reference_eval_v1 MMLU scorer against lm-evaluation-harness (hard gate: |delta| <= 0.5 pp).

lm-eval runs unmodified. Its fp32 reference keeps logits and a full-vocabulary log-softmax for every position
(~7.5 GB at MMLU's longest 3.1k-token contexts with Qwen's 152k vocabulary), which does not fit the dedicated
VRAM of an 8 GB laptop GPU, and WDDM silently spills into shared memory instead of failing. The test set is
therefore scored in parts on different devices and merged per question:

    # 1) subjects whose contexts fit (allocator capped at free VRAM, so a misfit fails loudly)
    python scripts/crosscheck_mmlu_lmeval.py run --model-path <snapshot> --device cuda:0 \
        --exclude-subjects high_school_european_history high_school_us_history --part-out <dir>/gpu.json
    # 2) the longest subjects on the CPU (fp32 as well)
    python scripts/crosscheck_mmlu_lmeval.py run --model-path <snapshot> --device cpu \
        --subjects high_school_european_history high_school_us_history --part-out <dir>/cpu.json
    # 3) merge and compare with our run (question-weighted overall, as lm-eval's mmlu group)
    python scripts/crosscheck_mmlu_lmeval.py compare --ours <run_dir>/mmlu_test_5shot.json \
        --parts <dir>/gpu.json <dir>/cpu.json --out results/phase2/mmlu_lmeval_crosscheck.json

Matched configuration: lm-eval tasks ``mmlu_<subject>`` (cais/mmlu), 5-shot (first_n from dev), same weights
and dtype, answer-choice log-likelihood.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from cg_fedllm.config import RESULT_LABELS
from cg_fedllm.utils.io import atomic_write_json, read_jsonl


def _all_subjects() -> list[str]:
    from cg_fedllm.evaluation.benchmarks import mmlu_categories

    return sorted(mmlu_categories()["subcategories"])


def cmd_run(args) -> None:
    import lm_eval
    from huggingface_hub import HfApi
    from lm_eval import simple_evaluate

    from cg_fedllm.utils.provenance import collect_environment

    environment = collect_environment()  # Git state / packages at start
    subjects = sorted(args.subjects) if args.subjects else _all_subjects()
    unknown = set(subjects) - set(_all_subjects())
    if unknown:
        raise SystemExit(f"unknown MMLU subjects: {sorted(unknown)}")
    subjects = [s for s in subjects if s not in set(args.exclude_subjects)]
    vram_guard = None
    if args.device.startswith("cuda"):
        from cg_fedllm.utils.gpu import cap_allocator_to_free_vram

        vram_guard = cap_allocator_to_free_vram(args.vram_margin_mb * 2**20)
    t0 = time.time()
    res = simple_evaluate(
        model="hf",
        model_args=f"pretrained={args.model_path},dtype={args.dtype}",
        tasks=[f"mmlu_{s}" for s in subjects],
        num_fewshot=5,
        batch_size=args.batch_size,
        device=args.device,
        log_samples=True,
        random_seed=0,
        numpy_random_seed=0,
        torch_random_seed=0,
        fewshot_random_seed=0,
        limit=args.limit,
    )
    elapsed = time.time() - t0
    per_subject, questions = {}, {}
    for task, samples in res["samples"].items():
        subject = task.removeprefix("mmlu_")
        per_subject[subject] = {"acc": res["results"][task]["acc,none"], "n": len(samples)}
        for s in samples:
            questions[f"mmlu/{subject}/test/{s['doc_id']}"] = bool(s["acc"])
    atomic_write_json(
        Path(args.part_out),
        {
            "device": args.device,
            "subjects": subjects,
            "limit": args.limit,
            "lm_eval": {
                "version": getattr(lm_eval, "__version__", None),
                "tasks": "mmlu_<subject>",
                "num_fewshot": 5,
                "model_args": f"pretrained={args.model_path},dtype={args.dtype}",
                "batch_size": args.batch_size,
                "cais_mmlu_main_sha_at_run": HfApi().dataset_info("cais/mmlu").sha,
                "elapsed_s": round(elapsed, 1),
                "vram_guard": vram_guard,
            },
            "environment": environment,
            "per_subject": per_subject,
            "questions": questions,
        },
    )
    print(f"{len(questions)} questions from {len(per_subject)} subjects on {args.device} in {elapsed:.0f} s")


def cmd_compare(args) -> None:
    ours = json.loads(Path(args.ours).read_text(encoding="utf-8"))
    preds_path = Path(args.ours).with_name(Path(args.ours).stem + "_predictions.jsonl")
    ours_q = {r["qid"]: (r["pred"] == r["answer"]) for r in read_jsonl(preds_path)}
    parts = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.parts]
    lm_q: dict[str, bool] = {}
    lm_subject: dict[str, float] = {}
    for part in parts:
        overlap = set(part["questions"]) & set(lm_q)
        if overlap:
            raise SystemExit(f"parts overlap in {len(overlap)} questions")
        lm_q.update(part["questions"])
        for subject, v in part["per_subject"].items():
            lm_subject[subject] = v["acc"]
    limited = any(p["limit"] is not None for p in parts)
    covered = set(lm_q) == set(ours_q)
    # lm-eval's mmlu group accuracy is the size-weighted mean of subject accuracies = mean over questions
    lm_overall = sum(lm_q.values()) / len(lm_q)
    agree = sum(int(lm_q[q] == ours_q[q]) for q in lm_q if q in ours_q)
    total = sum(1 for q in lm_q if q in ours_q)
    ours_subject = {s: v["acc"] for s, v in ours["aggregates"]["per_subject"].items()}
    diffs = {s: 100 * (ours_subject[s] - lm_subject[s]) for s in ours_subject if s in lm_subject}
    delta_pp = 100 * (ours["aggregates"]["overall"] - lm_overall)
    out = {
        "label": args.label,
        "gate": f"|ours - lm_eval| <= {args.tolerance_pp} percentage points (overall MMLU test, 5-shot, all questions)",
        "ours_overall": ours["aggregates"]["overall"],
        "lm_eval_overall": lm_overall,
        "delta_pp": delta_pp,
        "pass": abs(delta_pp) <= args.tolerance_pp and covered and not limited,
        "all_questions_covered_once": covered,
        "questions_compared": total,
        "question_level_agreement": agree / total if total else None,
        "per_subject_max_abs_delta_pp": max(abs(v) for v in diffs.values()) if diffs else None,
        "per_subject_delta_pp": diffs,
        "lm_eval_parts": [
            {k: p[k] for k in ("device", "subjects", "limit", "lm_eval", "environment")}
            | {"num_questions": len(p["questions"])}
            for p in parts
        ],
        "ours": {
            "protocol": ours["protocol"],
            "dataset": {k: ours["dataset"][k] for k in ("repo", "revision", "digest")},
            "timing_s": ours["timing_s"],
        },
    }
    atomic_write_json(Path(args.out), out)
    keys = (
        "ours_overall",
        "lm_eval_overall",
        "delta_pp",
        "pass",
        "questions_compared",
        "question_level_agreement",
        "per_subject_max_abs_delta_pp",
    )
    print(json.dumps({k: out[k] for k in keys}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="score (a subset of) MMLU with lm-eval and write a part file")
    r.add_argument("--model-path", required=True)
    r.add_argument("--device", default="cuda:0")
    r.add_argument("--dtype", default="float32")
    r.add_argument("--batch-size", default="1")
    r.add_argument("--subjects", nargs="*", default=None, help="default: all 57 subjects")
    r.add_argument("--exclude-subjects", nargs="*", default=[])
    r.add_argument(
        "--limit",
        type=int,
        default=None,
        help="questions per subject (script smoke test only; never passes the gate)",
    )
    r.add_argument(
        "--vram-margin-mb",
        type=int,
        default=256,
        help="CUDA: allocator cap = free dedicated VRAM - margin (WDDM spill guard)",
    )
    r.add_argument("--part-out", required=True)
    c = sub.add_parser("compare", help="merge part files and compare with our predictions")
    c.add_argument("--ours", required=True)
    c.add_argument("--parts", nargs="+", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--tolerance-pp", type=float, default=0.5)
    c.add_argument(
        "--label",
        choices=RESULT_LABELS,
        default="PHASE2-SMOKE",
        help="result label (Phase 2: evaluator validation = PHASE2-SMOKE)",
    )
    args = ap.parse_args()
    {"run": cmd_run, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    main()
