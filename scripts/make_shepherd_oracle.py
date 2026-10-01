"""AUDIT TOOL (not part of the pipeline): regenerate the Shepherd partition oracle fixtures.

It downloads the *pinned* Shepherd ``client_data_allocation.py`` and ``new-databricks-dolly-15k.json``
(commit bcffa00e9642990ecc6210363a7f0dab91bef4dc), verifies their SHA-256, inserts ONE line that adds a
``source_id`` column right after the JSON is read (sorting/sampling only use ``category``, so assignments
are unchanged -- verified against an unmodified run), executes it in a temporary directory and writes the
per-client source-ID lists to ``tests/fixtures/shepherd_oracle_*.json``. No Shepherd code is stored in
this repository and none is executed by the pipeline.

IMPORTANT: run it under pandas < 3 (e.g. a throw-away venv with ``pandas==2.3.3`` and the same numpy as
the project). pandas >= 3 sorts stably by default (different partition) and its copy-on-write makes
Shepherd's in-place ``np.random.shuffle`` of an Index raise ``ValueError: array is read-only``.

    python scripts/make_shepherd_oracle.py --python <python-with-pandas<3> --out tests/fixtures
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
import urllib.request
from pathlib import Path

COMMIT = "bcffa00e9642990ecc6210363a7f0dab91bef4dc"
BASE = f"https://raw.githubusercontent.com/JayZhang42/FederatedGPT-Shepherd/{COMMIT}/"
SHA256 = {
    "client_data_allocation.py": "4f250129f8ffd6ed7df59cf4da2ee936cf54847d874dba0848fb9b2ec524d84c",
    "new-databricks-dolly-15k.json": "52e0c44e2155bfc920927f94f28cb06fd27122338ce3f0bafb2955d86f58e066",
}
ANCHOR = "df = pd.read_json(\"new-databricks-dolly-15k.json\", orient='records')"
MODES = {"dirichlet100_seed42": ("100", "1"), "shards10_seed42": ("10", "0")}


def fetch(name: str, dest: Path) -> None:
    with urllib.request.urlopen(BASE + name, timeout=300) as r:  # noqa: S310 (pinned https URL)
        data = r.read()
    if hashlib.sha256(data).hexdigest() != SHA256[name]:
        raise SystemExit(f"{name}: SHA-256 mismatch")
    (dest / name).write_bytes(data)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--python", required=True, help="interpreter with pandas<3")
    ap.add_argument("--out", default="tests/fixtures")
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name in SHA256:
            fetch(name, tmp)
        src = (tmp / "client_data_allocation.py").read_text(encoding="utf-8")
        if ANCHOR not in src:
            raise SystemExit("anchor line not found; refusing to instrument")
        (tmp / "oracle.py").write_text(src.replace(ANCHOR, ANCHOR + "\ndf['source_id'] = list(range(len(df)))"), encoding="utf-8")
        versions = subprocess.run(
            [args.python, "-c", "import sys, numpy, pandas; print(pandas.__version__, numpy.__version__, sys.version.split()[0])"],
            check=True, capture_output=True, text=True,
        ).stdout.split()
        about = (
            "ORACLE: per-client source_id assignment produced by running the pinned Shepherd client_data_allocation.py "
            f"(JayZhang42/FederatedGPT-Shepherd@{COMMIT}) with an added source_id column, "
            f"under pandas {versions[0]} / numpy {versions[1]} / Python {versions[2]} (see docs/provenance.md). IDs only."
        )
        for tag, (n_clients, diff_quantity) in MODES.items():
            subprocess.run([args.python, "oracle.py", n_clients, diff_quantity], cwd=tmp, check=True, capture_output=True)
            d = tmp / "data" / n_clients
            load = lambda f, d=d: json.loads((d / f).read_text(encoding="utf-8"))  # noqa: E731
            out = {
                "_about": about,
                "source_sha256": SHA256["new-databricks-dolly-15k.json"],
                "mode": "dirichlet" if diff_quantity == "1" else "shards",
                "num_clients": int(n_clients),
                "holdout_per_category": 10,
                "seed": 42,
                "dirichlet_alpha": 0.5,
                "min_require_size": 40,
                "shards_per_client": 2,
                "holdout_ids": [r["source_id"] for r in load("global_test.json")],
                "remaining_ids": [r["source_id"] for r in load("global_training.json")],
                "client_ids": [[r["source_id"] for r in load(f"local_training_{i}.json")] for i in range(int(n_clients))],
            }
            Path(args.out, f"shepherd_oracle_{tag}.json").write_text(json.dumps(out, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")
            print("wrote", tag)


if __name__ == "__main__":
    main()
