"""Bounded GPU feasibility micro-benchmarks (LOCAL-MICROBENCH).

Thin wrapper around the package CLI: `python scripts/bench_gpu.py --config <yaml> [...]` == `cgfed bench-gpu --config <yaml> [...]`.
"""

from __future__ import annotations

import sys

from cg_fedllm.cli import main

if __name__ == "__main__":
    sys.exit(main(["bench-gpu", *sys.argv[1:]]))
