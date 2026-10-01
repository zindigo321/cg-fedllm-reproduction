"""Federated LoRA (LoRA-FT / FAF) or pooled centralized run from a config.

Thin wrapper around the package CLI: `python scripts/run_fl.py --config <yaml> [...]` == `cgfed run-fl --config <yaml> [...]`.
"""

from __future__ import annotations

import sys

from cg_fedllm.cli import main

if __name__ == "__main__":
    sys.exit(main(["run-fl", *sys.argv[1:]]))
