"""Train the ResNet-3 AutoEncoder on a TGAP snapshot directory.

Thin wrapper around the package CLI: `python scripts/train_ae.py --config <yaml> [...]` == `cgfed train-ae --config <yaml> [...]`.
"""

from __future__ import annotations

import sys

from cg_fedllm.cli import main

if __name__ == "__main__":
    sys.exit(main(["train-ae", *sys.argv[1:]]))
