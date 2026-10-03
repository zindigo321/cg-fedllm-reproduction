"""Score aggregation for ``reference_eval_v1``.

* C-Eval (official definition, verified by recomputing the C-Eval paper's published GPT-4 row): every
  category score is the unweighted mean of its subjects' accuracies; ``average`` is the mean over all
  52 subjects (not over the 4 categories); ``hard`` is the mean over the 8 C-Eval Hard subjects.
* MMLU (official ``evaluate.py`` and lm-eval ``weight_by_size: True``): overall and category scores are
  question-weighted (micro) averages. The unweighted subject mean is reported as a secondary number.

All accuracies are fractions in [0, 1].
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from cg_fedllm.evaluation.benchmarks import (
    ceval_hard_subjects,
    ceval_subject_mapping,
    mmlu_categories,
    mmlu_subject_category,
)
from cg_fedllm.evaluation.scorer import ScoredItem

CEVAL_CATEGORIES = ("STEM", "Social Science", "Humanities", "Other")


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def per_subject(items: Iterable[ScoredItem]) -> dict[str, dict[str, Any]]:
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for it in items:
        counts[it.subject][0] += int(it.correct)
        counts[it.subject][1] += 1
    return {s: {"correct": c, "n": n, "acc": c / n} for s, (c, n) in sorted(counts.items())}


def aggregate_ceval(items: list[ScoredItem]) -> dict[str, Any]:
    mapping = ceval_subject_mapping()
    hard = set(ceval_hard_subjects())
    ps = per_subject(items)
    unknown = sorted(set(ps) - set(mapping))
    if unknown:
        raise KeyError(f"subjects not in the official mapping: {unknown}")
    by_cat: dict[str, list[float]] = defaultdict(list)
    for s, st in ps.items():
        by_cat[mapping[s][2]].append(st["acc"])
    total_c = sum(st["correct"] for st in ps.values())
    total_n = sum(st["n"] for st in ps.values())
    return {
        "per_subject": ps,
        "categories": {c: _mean(by_cat.get(c, [])) for c in CEVAL_CATEGORIES},
        "average": _mean([st["acc"] for st in ps.values()]),
        "hard": _mean([st["acc"] for s, st in ps.items() if s in hard]),
        "secondary_question_weighted_overall": total_c / total_n if total_n else None,
        "num_subjects": len(ps),
        "num_hard_subjects": len(set(ps) & hard),
        "num_questions": total_n,
        "complete": set(ps) == set(mapping),
        "aggregation": "category/average/hard = unweighted mean over subjects (official C-Eval)",
    }


def aggregate_mmlu(items: list[ScoredItem]) -> dict[str, Any]:
    cats = mmlu_categories()
    ps = per_subject(items)
    cat_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    cat_subject_acc: dict[str, list[float]] = defaultdict(list)
    for s, st in ps.items():
        c = mmlu_subject_category(s, cats)
        cat_counts[c][0] += st["correct"]
        cat_counts[c][1] += st["n"]
        cat_subject_acc[c].append(st["acc"])
    total_c = sum(st["correct"] for st in ps.values())
    total_n = sum(st["n"] for st in ps.values())
    return {
        "per_subject": ps,
        "overall": total_c / total_n if total_n else None,
        "categories": {c: (v[0] / v[1]) for c, v in sorted(cat_counts.items())},
        "secondary_subject_macro_average": _mean([st["acc"] for st in ps.values()]),
        "secondary_categories_subject_macro": {c: _mean(v) for c, v in sorted(cat_subject_acc.items())},
        "num_subjects": len(ps),
        "num_questions": total_n,
        "complete": set(ps) == set(cats["subcategories"]),
        "aggregation": "overall/categories = question-weighted (official evaluate.py; lm-eval weight_by_size)",
    }


def paired_comparison(correct_a: dict[str, bool], correct_b: dict[str, bool]) -> dict[str, float | int]:
    """Two models on the same questions: discordant counts and the exact two-sided McNemar (binomial) p-value."""
    if set(correct_a) != set(correct_b):
        raise ValueError("the two runs scored different question sets")
    both = sum(1 for q in correct_a if correct_a[q] and correct_b[q])
    a_only = sum(1 for q in correct_a if correct_a[q] and not correct_b[q])
    b_only = sum(1 for q in correct_a if correct_b[q] and not correct_a[q])
    n = len(correct_a)
    disc = a_only + b_only
    k = min(a_only, b_only)
    p = min(1.0, 2 * sum(math.comb(disc, i) for i in range(k + 1)) / 2**disc) if disc else 1.0
    return {
        "questions": n,
        "accuracy_a": (both + a_only) / n,
        "accuracy_b": (both + b_only) / n,
        "both_correct": both,
        "a_only_correct": a_only,
        "b_only_correct": b_only,
        "neither_correct": n - both - a_only - b_only,
        "mcnemar_exact_two_sided_p": p,
    }
