"""reference_eval_v1 evaluation of a base model or adapter.

Thin wrapper around the package CLI: `python scripts/evaluate.py --config <yaml> [...]` == `cgfed evaluate --config <yaml> [...]`.
"""

from __future__ import annotations

import sys

from cg_fedllm.cli import main

if __name__ == "__main__":
    sys.exit(main(["evaluate", *sys.argv[1:]]))
