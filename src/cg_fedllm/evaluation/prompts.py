"""Prompt construction for ``reference_eval_v1``.

This is OUR stable reference protocol, not a recovered CG-FedLLM protocol (the paper does not state its
evaluation prompts, shots or scoring).

C-Eval (official answer-only format of ``hkust-nlp/ceval`` ``code/evaluator_series``)::

    以下是中国关于{中文科目}考试的单项选择题，请选出其中的正确答案。\\n\\n
    {question}\\nA. {A}\\nB. {B}\\nC. {C}\\nD. {D}\\n答案：{X}\\n\\n        (x k shots from dev)
    {question}\\nA. {A}\\nB. {B}\\nC. {C}\\nD. {D}\\n答案：
    continuations: "A" "B" "C" "D"

We use the Chinese subject name from the official ``subject_mapping.json`` (the official LLaMA script
passes the English handle, a known quirk).

MMLU (identical to lm-evaluation-harness 0.4.x ``mmlu`` default template, i.e. Hendrycks' format
without its double-space quirk)::

    The following are multiple choice questions (with answers) about {subject with spaces}.\\n\\n
    {question.strip()}\\nA. ..\\nB. ..\\nC. ..\\nD. ..\\nAnswer: {X}\\n\\n   (x k shots: first k dev items)
    {question.strip()}\\nA. ..\\nB. ..\\nC. ..\\nD. ..\\nAnswer:
    continuations: " A" " B" " C" " D"
"""

from __future__ import annotations

from collections.abc import Sequence

from cg_fedllm.evaluation.benchmarks import LETTERS, MCQuestion

PROTOCOL_ID = "reference_eval_v1"

CEVAL_HEADER = "以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。\n\n"
MMLU_HEADER = "The following are multiple choice questions (with answers) about {subject}.\n\n"

CONTINUATIONS = {"ceval": tuple(LETTERS), "mmlu": tuple(f" {x}" for x in LETTERS)}


def ceval_example(q: MCQuestion, include_answer: bool) -> str:
    text = q.question
    for letter, choice in zip(LETTERS, q.choices):
        text += f"\n{letter}. {choice}"
    return text + (f"\n答案：{q.answer}\n\n" if include_answer else "\n答案：")


def mmlu_example(q: MCQuestion, include_answer: bool) -> str:
    text = q.question.strip()
    for letter, choice in zip(LETTERS, q.choices):
        text += f"\n{letter}. {choice}"
    text += "\nAnswer:"
    return text + (f" {q.answer}\n\n" if include_answer else "")


def build_prompt(benchmark: str, subject_display: str, shots: Sequence[MCQuestion], q: MCQuestion) -> str:
    if benchmark == "ceval":
        return CEVAL_HEADER.format(subject=subject_display) + "".join(ceval_example(s, True) for s in shots) + ceval_example(q, False)
    if benchmark == "mmlu":
        return MMLU_HEADER.format(subject=subject_display) + "".join(mmlu_example(s, True) for s in shots) + mmlu_example(q, False)
    raise ValueError(f"unknown benchmark {benchmark!r}")


def mmlu_display_name(subject: str) -> str:
    return subject.replace("_", " ")
