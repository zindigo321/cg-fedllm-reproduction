"""Capture the software/hardware environment as JSON: ``python scripts/capture_env.py [--out env.json]``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cg_fedllm.utils.io import atomic_write_json
from cg_fedllm.utils.provenance import collect_environment

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    env = collect_environment()
    if args.out:
        atomic_write_json(Path(args.out), env)
    print(json.dumps(env, indent=2, default=str))
