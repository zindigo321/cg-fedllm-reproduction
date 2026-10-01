"""Tier-C end-to-end smoke (correctness only; outputs are PHASE2-SMOKE).

Thin wrapper around the package CLI: `python scripts/run_smoke.py --config <yaml> [...]` == `cgfed smoke --config <yaml> [...]`.
"""

from __future__ import annotations

import sys

from cg_fedllm.cli import main

if __name__ == "__main__":
    sys.exit(main(["smoke", *sys.argv[1:]]))
