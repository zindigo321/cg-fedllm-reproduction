"""Pinned loading of C-Eval and MMLU from Hugging Face parquet files.

Only the per-subject ``dev``/``val(idation)``/``test`` parquet files are downloaded (never MMLU's
``auxiliary_train``), from an explicit dataset revision, into the configured HF cache. Every file's
SHA-256 is recorded so an evaluation result identifies its exact data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from cg_fedllm.utils.hashing import canonical_json_sha256, sha256_file

LETTERS = ("A", "B", "C", "D")
CEVAL_SPLITS = {"dev": "dev", "val": "val", "test": "test"}
MMLU_SPLITS = {"dev": "dev", "val": "validation", "validation": "validation", "test": "test"}


@dataclass(frozen=True)
class MCQuestion:
    benchmark: str
    subject: str
    split: str
    index: int
    question: str
    choices: tuple[str, str, str, str]
    answer: str

    @property
    def qid(self) -> str:
        return f"{self.benchmark}/{self.subject}/{self.split}/{self.index}"


def _resource_json(name: str) -> Any:
    return json.loads(
        resources.files("cg_fedllm.evaluation.resources").joinpath(name).read_text(encoding="utf-8")
    )


def ceval_subject_mapping() -> dict[str, list[str]]:
    """subject -> [English name, Chinese name, category] (official hkust-nlp/ceval mapping)."""
    return _resource_json("ceval_subject_mapping.json")


def ceval_hard_subjects() -> list[str]:
    return list(_resource_json("ceval_hard_subjects.json")["hard_subjects"])


def mmlu_categories() -> dict[str, Any]:
    return _resource_json("mmlu_categories.json")


def mmlu_subject_category(subject: str, cats: dict[str, Any] | None = None) -> str:
    cats = cats or mmlu_categories()
    sub = cats["subcategories"][subject][0]
    for category, subs in cats["categories"].items():
        if sub in subs:
            return category
    raise KeyError(subject)


def download_benchmark(repo: str, revision: str, benchmark: str, splits: list[str]) -> Path:
    """Snapshot only the needed per-subject parquet files at a pinned revision; return the local root."""
    from huggingface_hub import snapshot_download

    if len(revision) < 40:
        raise ValueError(f"benchmark revision must be a full commit SHA, got {revision!r}")
    names = CEVAL_SPLITS if benchmark == "ceval" else MMLU_SPLITS
    patterns = sorted({f"*/{names[s]}-*.parquet" for s in splits})
    local = snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        revision=revision,
        allow_patterns=patterns,
        ignore_patterns=["all/*", "auxiliary_train/*"],
    )
    return Path(local)


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def load_split(
    root: Path, benchmark: str, split: str, subjects: list[str] | None = None
) -> dict[str, list[MCQuestion]]:
    names = CEVAL_SPLITS if benchmark == "ceval" else MMLU_SPLITS
    fname = names[split]
    available = sorted(
        p.name for p in root.iterdir() if p.is_dir() and p.name not in ("all", "auxiliary_train")
    )
    wanted = subjects or available
    out: dict[str, list[MCQuestion]] = {}
    for subject in wanted:
        files = sorted((root / subject).glob(f"{fname}-*.parquet"))
        if len(files) != 1:
            raise FileNotFoundError(
                f"{benchmark}/{subject}: expected exactly one {fname} parquet, found {files}"
            )
        rows = _read_parquet(files[0])
        qs = []
        for i, row in enumerate(rows):
            if benchmark == "ceval":
                choices = (row["A"], row["B"], row["C"], row["D"])
                answer = row["answer"]
            else:
                choices = tuple(row["choices"])
                answer = LETTERS[int(row["answer"])]
            if len(choices) != 4 or answer not in LETTERS:
                raise ValueError(f"{benchmark}/{subject}/{split}#{i}: malformed row")
            qs.append(
                MCQuestion(
                    benchmark, subject, split, i, str(row["question"]), tuple(map(str, choices)), answer
                )
            )
        out[subject] = qs
    return out


def data_digest(root: Path, benchmark: str, splits: list[str]) -> dict[str, Any]:
    names = CEVAL_SPLITS if benchmark == "ceval" else MMLU_SPLITS
    files = {}
    for s in splits:
        for p in sorted(root.glob(f"*/{names[s]}-*.parquet")):
            files[p.relative_to(root).as_posix()] = sha256_file(p)
    return {"files": files, "num_files": len(files), "digest": canonical_json_sha256(files)}
