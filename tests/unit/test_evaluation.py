"""reference_eval_v1 fixtures: prompts, few-shot selection, answer extraction, macro/micro (item 23)."""

from __future__ import annotations

from collections import Counter

import pytest
import torch

from cg_fedllm.evaluation.aggregate import aggregate_ceval, aggregate_mmlu
from cg_fedllm.evaluation.benchmarks import (
    MCQuestion,
    ceval_hard_subjects,
    ceval_subject_mapping,
    mmlu_categories,
    mmlu_subject_category,
)
from cg_fedllm.evaluation.prompts import build_prompt, mmlu_display_name
from cg_fedllm.evaluation.scorer import ScoredItem, build_requests, score_requests


def q(bench, subject, i, answer="B", question="1+1=?"):
    return MCQuestion(bench, subject, "test", i, question, ("0", "2", "3", "4"), answer)


def test_ceval_prompt_is_the_official_answer_only_format():
    shots = [q("ceval", "computer_network", 0, "A", "题一"), q("ceval", "computer_network", 1, "C", "题二")]
    p = build_prompt("ceval", "计算机网络", shots, q("ceval", "computer_network", 9, "B", "题三"))
    expected = (
        "以下是中国关于计算机网络考试的单项选择题，请选出其中的正确答案。\n\n"
        "题一\nA. 0\nB. 2\nC. 3\nD. 4\n答案：A\n\n"
        "题二\nA. 0\nB. 2\nC. 3\nD. 4\n答案：C\n\n"
        "题三\nA. 0\nB. 2\nC. 3\nD. 4\n答案："
    )
    assert p == expected


def test_mmlu_prompt_matches_lm_eval_template():
    shots = [q("mmlu", "abstract_algebra", 0, "D", " What? ")]
    p = build_prompt("mmlu", mmlu_display_name("abstract_algebra"), shots, q("mmlu", "abstract_algebra", 5, "B", "Why?\n"))
    expected = (
        "The following are multiple choice questions (with answers) about abstract algebra.\n\n"
        "What?\nA. 0\nB. 2\nC. 3\nD. 4\nAnswer: D\n\n"
        "Why?\nA. 0\nB. 2\nC. 3\nD. 4\nAnswer:"
    )
    assert p == expected


def test_few_shot_selection_is_first_n_and_reduced_only_when_too_long(toy_tokenizer):
    dev = {"s": [q("ceval", "s", i, "A", f"dev{i}") for i in range(5)]}
    qs = {"s": [q("ceval", "s", 0, "B", "test")]}
    reqs = build_requests(toy_tokenizer, "ceval", qs, dev, {"s": "科目"}, 5, None)
    assert reqs[0].num_shots == 5
    full_len = len(reqs[0].context_ids)
    short = build_requests(toy_tokenizer, "ceval", qs, dev, {"s": "科目"}, 5, full_len - 5)
    assert short[0].num_shots < 5 and not short[0].truncated
    assert reqs[0].continuation_ids == [[3 + ord(c)] for c in "ABCD"]  # C-Eval: single-token continuations
    mm = build_requests(toy_tokenizer, "mmlu", {"s": [q("mmlu", "s", 0)]}, {"s": [q("mmlu", "s", 1)]}, {"s": "s"}, 1, None)
    assert all(len(c) == 2 for c in mm[0].continuation_ids)  # " A" is two toy tokens -> multi-token path


def test_scorer_batched_fast_path_matches_unbatched_reference(tiny_hf_llama, toy_tokenizer):
    model = tiny_hf_llama
    # toy ids are 3 + code point mod 997; the tiny model's vocab is 128 -> map into range via ASCII-only text
    dev = {"s": [MCQuestion("ceval", "s", "dev", i, f"q{i}?", ("a", "b", "c", "d"), "ABCD"[i % 4]) for i in range(3)]}
    qs = {"s": [MCQuestion("ceval", "s", "test", i, "x" * (i + 1), ("e", "f", "g", "h"), "ABCD"[i % 4]) for i in range(6)]}

    class Ascii(type(toy_tokenizer)):
        def _enc(self, text):
            return [3 + (ord(c) % 120) for c in text]

    tok = Ascii()
    reqs = build_requests(tok, "ceval", qs, dev, {"s": "s"}, 3, None, add_special_tokens=True)
    items = score_requests(model, reqs, pad_token_id=0, device="cpu", max_batch_tokens=10_000, max_batch_size=4)
    for r, it in zip(reqs, items):
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([r.context_ids])).logits[0, -1].float()
        lp = torch.log_softmax(logits, dim=-1)
        ref = [float(lp[c[0]]) for c in r.continuation_ids]
        assert it.path == "single_token"
        assert max(abs(a - b) for a, b in zip(ref, it.logprobs)) < 1e-4
        assert it.prediction == "ABCD"[max(range(4), key=lambda j: (ref[j], -j))]


def test_multi_token_fallback_runs(tiny_hf_llama):
    class Ascii:
        eos_token_id = 2

        def __call__(self, text, add_special_tokens=True, **kw):
            enc = lambda t: ([1] if add_special_tokens else []) + [3 + (ord(c) % 120) for c in t]  # noqa: E731
            return {"input_ids": [enc(t) for t in text]} if isinstance(text, list) else {"input_ids": enc(text)}

    qs = {"s": [MCQuestion("mmlu", "s", "test", 0, "why", ("a", "b", "c", "d"), "C")]}
    dev = {"s": [MCQuestion("mmlu", "s", "dev", 0, "how", ("a", "b", "c", "d"), "A")]}
    reqs = build_requests(Ascii(), "mmlu", qs, dev, {"s": "s"}, 1, None)
    items = score_requests(tiny_hf_llama, reqs, pad_token_id=0, device="cpu")
    assert items[0].path == "multi_token" and len(items[0].logprobs) == 4


def _items(spec: dict[str, tuple[int, int]]) -> list[ScoredItem]:
    out = []
    for subject, (correct, n) in spec.items():
        for i in range(n):
            ok = i < correct
            out.append(ScoredItem(f"{subject}/{i}", subject, "A", "A" if ok else "B", ok, [0, 0, 0, 0], 5, 10, "single_token"))
    return out


def test_ceval_mapping_and_subject_macro_aggregation():
    mapping = ceval_subject_mapping()
    assert len(mapping) == 52
    assert Counter(v[2] for v in mapping.values()) == {"STEM": 20, "Social Science": 10, "Humanities": 11, "Other": 11}
    hard = ceval_hard_subjects()
    assert len(hard) == 8 and set(hard) <= set(mapping)
    agg = aggregate_ceval(_items({"advanced_mathematics": (1, 2), "computer_network": (9, 10), "law": (1, 4)}))
    assert agg["categories"]["STEM"] == pytest.approx((0.5 + 0.9) / 2)  # mean over subjects, not questions
    assert agg["average"] == pytest.approx((0.5 + 0.9 + 0.25) / 3)
    assert agg["hard"] == pytest.approx(0.5)
    assert agg["secondary_question_weighted_overall"] == pytest.approx(11 / 16)
    assert agg["complete"] is False


def test_mmlu_question_weighted_aggregation():
    cats = mmlu_categories()
    assert len(cats["subcategories"]) == 57
    assert mmlu_subject_category("abstract_algebra") == "STEM"
    assert mmlu_subject_category("philosophy") == "humanities"
    agg = aggregate_mmlu(_items({"abstract_algebra": (1, 2), "college_physics": (9, 10), "philosophy": (0, 4)}))
    assert agg["overall"] == pytest.approx(10 / 16)  # question-weighted
    assert agg["categories"]["STEM"] == pytest.approx(10 / 12)
    assert agg["secondary_subject_macro_average"] == pytest.approx((0.5 + 0.9 + 0.0) / 3)


def test_batching_respects_token_size_and_attention_budgets():
    from cg_fedllm.evaluation.scorer import _batches

    lengths = [900, 850, 400, 390, 380, 100, 90, 80, 70, 60]
    order = list(range(len(lengths)))
    for budget in (None, 2_000_000, 500_000):
        batches = _batches(order, lengths, max_batch_tokens=2000, max_batch_size=4, max_batch_attention=budget)
        assert sorted(i for b in batches for i in b) == order
        for b in batches:
            lmax = max(lengths[i] for i in b)
            assert len(b) <= 4 and lmax * len(b) <= 2000 or len(b) == 1
            if budget is not None:
                assert lmax * lmax * len(b) <= budget or len(b) == 1


def test_scores_are_invariant_to_batching(tiny_hf_llama, toy_tokenizer):
    class Ascii(type(toy_tokenizer)):
        def _enc(self, text):
            return [3 + (ord(c) % 120) for c in text]

    dev = {"s": [MCQuestion("ceval", "s", "dev", i, f"q{i}?", ("a", "b", "c", "d"), "ABCD"[i % 4]) for i in range(3)]}
    qs = {"s": [MCQuestion("ceval", "s", "test", i, "y" * (3 * i + 1), ("e", "f", "g", "h"), "ABCD"[i % 4]) for i in range(8)]}
    reqs = build_requests(Ascii(), "ceval", qs, dev, {"s": "s"}, 3, None)
    a = score_requests(tiny_hf_llama, reqs, 0, "cpu", max_batch_tokens=100_000, max_batch_size=8)
    b = score_requests(tiny_hf_llama, reqs, 0, "cpu", max_batch_tokens=100_000, max_batch_size=8, max_batch_attention=1)
    assert [x.prediction for x in a] == [x.prediction for x in b]
    assert max(abs(u - v) for x, y in zip(a, b) for u, v in zip(x.logprobs, y.logprobs)) < 1e-4


def test_stratified_subset_is_seeded_per_subject_and_keeps_order():
    from cg_fedllm.config import BenchmarkSpec, ConfigError, build_dataclass
    from cg_fedllm.evaluation.reference_eval import stratified_subset

    qs = {"a": [q("mmlu", "a", i) for i in range(10)], "b": [q("mmlu", "b", i) for i in range(3)]}
    sub = stratified_subset(qs, 0.25, 7, "mmlu", "test")
    assert [len(v) for v in sub.values()] == [3, 1]  # ceil(2.5), ceil(0.75)
    assert [x.index for x in sub["a"]] == sorted(x.index for x in sub["a"])
    assert sub == stratified_subset(qs, 0.25, 7, "mmlu", "test")
    assert stratified_subset({"a": qs["a"]}, 0.25, 7, "mmlu", "test")["a"] == sub["a"]  # independent of other subjects
    assert stratified_subset(qs, 1.0, 7, "mmlu", "test") == qs
    with pytest.raises(ConfigError):
        build_dataclass(BenchmarkSpec, {"name": "mmlu", "split": "test", "subset_fraction": 0.2, "limit_per_subject": 5})
    with pytest.raises(ConfigError):
        build_dataclass(BenchmarkSpec, {"name": "mmlu", "split": "test", "subset_fraction": 0.0})


def test_batch_plan_counts_the_forward_passes_of_the_scorer(tiny_hf_llama, toy_tokenizer):
    from cg_fedllm.evaluation.scorer import batch_plan

    class Ascii(type(toy_tokenizer)):  # ids inside the tiny model's vocabulary (C-Eval template is not ASCII)
        def _enc(self, text):
            return [3 + (ord(c) % 120) for c in text]

    tok = Ascii()
    model, orig = tiny_hf_llama, tiny_hf_llama.forward
    for bench, display in (("ceval", "s"), ("mmlu", "s")):  # per-character ids: C-Eval single-token, MMLU multi-token
        questions = {"s": [q(bench, "s", i, question="x " * (3 + 5 * i)) for i in range(7)]}
        dev = {"s": [q(bench, "s", 100 + i) for i in range(5)]}
        reqs = build_requests(tok, bench, questions, dev, {"s": display}, 2, None)
        calls = []

        def spy(*a, calls=calls, **k):
            calls.append(tuple(k["input_ids"].shape))
            return orig(*a, **k)

        model.forward = spy
        try:
            score_requests(model, reqs, 0, "cpu", max_batch_tokens=200, max_batch_size=3)
        finally:
            model.forward = orig
        plan = batch_plan(reqs, max_batch_tokens=200, max_batch_size=3)
        assert plan["questions"] == 7 and plan["context_tokens"] == sum(len(r.context_ids) for r in reqs)
        if bench == "ceval":
            assert plan["multi_token_requests"] == 0 and plan["single_token_batches"] == len(calls) >= 3
            assert plan["padded_tokens"] == sum(b * n for b, n in calls) >= plan["context_tokens"]
        else:
            assert plan["single_token_batches"] == 0 and plan["multi_token_requests"] == 7
            assert plan["multi_token_forward_tokens"] == sum(b * n for b, n in calls) and len(calls) == 7 * 4
