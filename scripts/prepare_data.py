"""Download/verify the pinned Dolly source, rebuild the Shepherd-compatible partition and per-client
D1/D2 split, and verify it byte-for-byte against the committed manifest.

    python scripts/prepare_data.py --config configs/base/shepherd_dolly.yaml [--write]

``--write`` (re)writes the committed manifest; use it only for an intended partition change.
"""

from __future__ import annotations

import argparse
import json

from cg_fedllm.config import load_config
from cg_fedllm.data.manifest import prepare_manifest

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    cfg.require("data")
    manifest, status = prepare_manifest(cfg.data, write=args.write)
    sizes = [len(c["ids"]) for c in manifest.data["clients"]]
    print(
        json.dumps(
            {
                **status,
                "num_clients": len(sizes),
                "min_client": min(sizes),
                "max_client": max(sizes),
                "holdout": len(manifest.holdout_ids),
            },
            indent=2,
        )
    )
