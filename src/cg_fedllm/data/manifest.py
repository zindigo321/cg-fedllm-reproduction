"""Canonical, hash-stable partition manifests.

A manifest records *which source records* (by integer ``source_id``) belong to the held-out test set,
to every client, and to every client's D1/D2 split, together with the full provenance of the source
file and of the partition algorithm. Manifests are serialised with :func:`canonical_json`, so the file's
SHA-256 identifies the partition exactly and is stable across operating systems. Manifests contain no
dataset text and can be committed to Git.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cg_fedllm.config import DataSection, resolve_path
from cg_fedllm.data.dolly import DollyRecord, load_source, source_cache_path
from cg_fedllm.data.partition import d1_count, select_subset, shepherd_partition, split_d1_d2
from cg_fedllm.utils.hashing import canonical_json, sha256_bytes, sha256_file
from cg_fedllm.utils.io import atomic_write_bytes

MANIFEST_SCHEMA = "cg_fedllm.partition_manifest/v1"
SHEPHERD_REFERENCE = (
    "JayZhang42/FederatedGPT-Shepherd@bcffa00e9642990ecc6210363a7f0dab91bef4dc:client_data_allocation.py"
)


class ManifestError(RuntimeError):
    pass


@dataclass(frozen=True)
class PartitionManifest:
    data: dict[str, Any]

    @property
    def num_clients(self) -> int:
        return len(self.data["clients"])

    def client_ids(self, client_id: int) -> list[int]:
        return list(self.data["clients"][client_id]["ids"])

    def split_ids(self, client_id: int, split: str) -> list[int]:
        entry = self.data["clients"][client_id]
        if split == "all":
            return list(entry["ids"])
        if split not in ("d1", "d2"):
            raise ManifestError(f"unknown split {split!r}")
        return list(entry[f"{split}_ids"])

    @property
    def holdout_ids(self) -> list[int]:
        return list(self.data["holdout_ids"])

    def sha256(self) -> str:
        return sha256_bytes(canonical_json(self.data))


def build_manifest(
    cfg: DataSection, records: Sequence[DollyRecord], source_file_sha256: str
) -> PartitionManifest:
    """Run the Shepherd-compatible partition (+ per-client D1/D2 split) and assemble the manifest."""
    if source_file_sha256 != cfg.source_sha256:
        raise ManifestError("source file sha256 does not match the pinned config value")
    all_ids = [r.source_id for r in records]
    cat_of = {r.source_id: r.category for r in records}
    subset = None
    ids = all_ids
    if cfg.subset_per_category is not None:
        ids = select_subset(all_ids, [cat_of[i] for i in all_ids], cfg.subset_per_category, cfg.subset_seed)
        subset = {"per_category": cfg.subset_per_category, "seed": cfg.subset_seed, "num_selected": len(ids)}
    part = shepherd_partition(
        ids,
        [cat_of[i] for i in ids],
        num_clients=cfg.num_clients,
        mode=cfg.partition_mode,
        holdout_per_category=cfg.holdout_per_category,
        dirichlet_alpha=cfg.dirichlet_alpha,
        min_require_size=cfg.min_require_size,
        shards_per_client=cfg.shards_per_client,
        seed=cfg.partition_seed,
    )
    d1, d2 = split_d1_d2(part.client_ids, cfg.d1_fraction, cfg.d1d2_split_seed)
    clients = []
    for cid, cids in enumerate(part.client_ids):
        cats = sorted(cat_of[i] for i in cids)
        comp = {c: cats.count(c) for c in sorted(set(cats))}
        clients.append(
            {"client_id": cid, "ids": cids, "d1_ids": d1[cid], "d2_ids": d2[cid], "category_counts": comp}
        )
    data = {
        "schema": MANIFEST_SCHEMA,
        "source": {
            "name": cfg.source_name,
            "url": cfg.source_url,
            "revision": cfg.source_revision,
            "sha256": cfg.source_sha256,
            "license": cfg.source_license,
            "num_records": len(records),
        },
        "subset": subset,
        "partition": {
            "algorithm": "shepherd_client_data_allocation (independent reimplementation)",
            "reference": SHEPHERD_REFERENCE,
            "ordering_semantics": "pandas<3 sort_values == numpy argsort(kind='quicksort') on object dtype",
            "mode": cfg.partition_mode,
            "num_clients": cfg.num_clients,
            "holdout_per_category": cfg.holdout_per_category,
            "dirichlet_alpha": cfg.dirichlet_alpha,
            "min_require_size": cfg.min_require_size,
            "shards_per_client": cfg.shards_per_client,
            "seed": cfg.partition_seed,
            "dirichlet_attempts": part.dirichlet_attempts,
            "numpy_version": np.__version__,
            "status": "RECONSTRUCTED-FROM-SHEPHERD",
        },
        "split": {
            "mode": "per_client_after_partition",
            "d1_fraction": cfg.d1_fraction,
            "seed": cfg.d1d2_split_seed,
            "d1_count_rule": "floor(d1_fraction * n + 0.5)",
            "rng": "numpy PCG64(derive_seed(seed, 'd1d2', client_id)).permutation(n)",
            "status": "INFERRED (paper gives 3:7 but not the granularity; reviewer decision R7)",
        },
        "holdout_ids": part.holdout_ids,
        "clients": clients,
    }
    manifest = PartitionManifest(data)
    validate_manifest(manifest)
    return manifest


def validate_manifest(m: PartitionManifest) -> None:
    """Structural invariants: disjoint clients, exact D1/D2 split per client, no holdout leakage."""
    data = m.data
    if data.get("schema") != MANIFEST_SCHEMA:
        raise ManifestError(f"unexpected schema {data.get('schema')!r}")
    holdout = set(data["holdout_ids"])
    seen: set[int] = set()
    frac = data["split"]["d1_fraction"]
    for entry in data["clients"]:
        ids, d1, d2 = entry["ids"], entry["d1_ids"], entry["d2_ids"]
        sid = set(ids)
        if len(sid) != len(ids):
            raise ManifestError(f"client {entry['client_id']} has duplicate IDs")
        if sid & seen:
            raise ManifestError(f"client {entry['client_id']} overlaps another client")
        if sid & holdout:
            raise ManifestError(f"client {entry['client_id']} contains held-out IDs")
        if set(d1) & set(d2):
            raise ManifestError(f"client {entry['client_id']}: D1 and D2 overlap")
        if sorted(d1 + d2) != sorted(ids):
            raise ManifestError(f"client {entry['client_id']}: D1 u D2 != client data")
        if len(d1) != d1_count(len(ids), frac):
            raise ManifestError(f"client {entry['client_id']}: wrong D1 size")
        seen |= sid


def write_manifest(path: str | Path, m: PartitionManifest) -> str:
    """Write the manifest as canonical JSON; returns the file's SHA-256."""
    atomic_write_bytes(path, canonical_json(m.data))
    return sha256_file(path)


def load_manifest(path: str | Path, expected_sha256: str | None = None) -> PartitionManifest:
    path = Path(path)
    if expected_sha256 is not None:
        digest = sha256_file(path)
        if digest != expected_sha256:
            raise ManifestError(f"manifest {path} sha256 {digest} != expected {expected_sha256}")
    import json

    m = PartitionManifest(json.loads(path.read_text(encoding="utf-8")))
    validate_manifest(m)
    return m


def prepare_manifest(cfg: DataSection, write: bool = False) -> tuple[PartitionManifest, dict[str, Any]]:
    """Rebuild the manifest from the pinned source and verify it against the committed file.

    The committed manifest is the reference: a mismatch raises unless ``write=True`` (an explicit decision
    to regenerate it, e.g. after an intended partition change)."""
    records = load_source(cfg)
    built = build_manifest(cfg, records, sha256_file(source_cache_path(cfg)))
    path = resolve_path(cfg.manifest_path)
    status: dict[str, Any] = {"manifest_path": str(path), "built_sha256": built.sha256()}
    if path.exists():
        committed = load_manifest(path)
        status["committed_sha256"] = sha256_file(path)
        status["identical"] = committed.sha256() == built.sha256() == status["committed_sha256"]
        if not status["identical"] and not write:
            raise ManifestError(
                f"rebuilt manifest differs from committed {path}; rerun with --write only if intended"
            )
    if write or not path.exists():
        status["written_sha256"] = write_manifest(path, built)
    return built, status
