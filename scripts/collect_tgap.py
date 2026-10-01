"""TGAP snapshot collection (local_pretrain or federated_pretrain).

Thin wrapper around the package CLI: `python scripts/collect_tgap.py --config <yaml> [...]` == `cgfed collect-tgap --config <yaml> [...]`.
"""

from __future__ import annotations

import sys

from cg_fedllm.cli import main

if __name__ == "__main__":
    sys.exit(main(["collect-tgap", *sys.argv[1:]]))
