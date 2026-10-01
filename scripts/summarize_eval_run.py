"""Condense a ``cgfed evaluate`` run directory into one lightweight, labelled result file.

The run directory (outside Git) keeps the full per-benchmark JSON and per-question predictions; the summary
records their SHA-256 so the committed numbers can be traced back to the exact files.

    python scripts/summarize_eval_run.py --run-dir <runs>/<name>/<stage> --label PHASE2-SMOKE \
        --out results/phase2/eval_validation_qwen15_0p5b.json [--note "..."]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from cg_fedllm.config import RESULT_LABELS
from cg_fedllm.utils.io import atomic_write_json


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", required=True, choices=RESULT_LABELS)
    ap.add_argument("--note", action="append", default=[], help="provenance note (repeatable)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    meta = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    benchmarks = {}
    for path in sorted(run_dir.glob("*shot*.json")):
        res = json.loads(path.read_text(encoding="utf-8"))
        preds = path.with_name(path.stem + "_predictions.jsonl")
        key = f"{res['benchmark']['name']}/{res['benchmark']['split']}"
        benchmarks[key] = {
            "protocol": res["protocol"],
            "benchmark": res["benchmark"],
            "dataset": {k: res["dataset"][k] for k in ("repo", "revision", "digest", "num_files")},
            "tokenization": res["tokenization"],
            "scoring": res["scoring"],
            "timing_s": res["timing_s"],
            "aggregates": res["aggregates"],
            "source_files": {
                path.name: _sha256(path),
                **({preds.name: _sha256(preds)} if preds.exists() else {}),
            },
            "label_in_source_file": res.get("label"),
        }
    env = meta["environment"]
    summary = {
        "label": args.label,
        "run_dir": str(run_dir),
        "stage": meta["stage"],
        "config_sha256": meta["config_sha256"],
        "model": meta.get("model"),
        "adapter": meta.get("adapter"),
        "git_at_run": env.get("git"),
        "packages": env.get("packages"),
        "cuda": env.get("cuda"),
        "notes": args.note,
        "benchmarks": benchmarks,
    }
    atomic_write_json(Path(args.out), summary)
    for key, b in benchmarks.items():
        agg = b["aggregates"]
        headline = agg.get("overall", agg.get("average"))
        print(f"{key}: n={agg['num_questions']} headline={headline:.4f} complete={agg['complete']}")


if __name__ == "__main__":
    main()
