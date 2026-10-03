"""Common TGAP snapshot schema (shared by ``local_pretrain`` and ``federated_pretrain``).

Each snapshot stores the client's complete post-training LoRA state (``end_state``) as safetensors and
references the start state it trained from (stored once per distinct start state under ``starts/``).
Because both states are available, either representation (``adapter_state`` or ``adapter_delta``) and
any Phi layout can be derived later without re-collection (reviewer decisions R2/R4). Every record in
``index.jsonl`` carries the schema below; payload files stay outside Git.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from cg_fedllm.compression.layout import Layout, LoRAGeometry, geometry_from_dict, infer_geometry
from cg_fedllm.compression.representation import to_representation
from cg_fedllm.models.adapter import AdapterState
from cg_fedllm.utils.hashing import sha256_file
from cg_fedllm.utils.io import append_jsonl, read_jsonl

SNAPSHOT_SCHEMA = "cg_fedllm.tgap_snapshot/v1"
REQUIRED_FIELDS = (
    "schema",
    "run_id",
    "source_mode",
    "time_index",
    "client_id",
    "representation_mode",
    "derivable_representations",
    "start_adapter_hash",
    "end_adapter_hash",
    "start_file",
    "start_file_sha256",
    "num_samples",
    "layout_id",
    "layout_key_order_sha256",
    "geometry",
    "phi_shape",
    "dtype",
    "l2",
    "file",
    "file_sha256",
    "created_utc",
)


class SnapshotError(RuntimeError):
    pass


class SnapshotWriter:
    def __init__(
        self, root: str | Path, *, run_id: str, source_mode: str, representation: str, layout: Layout
    ) -> None:
        if source_mode not in ("local_pretrain", "federated_pretrain"):
            raise ValueError(f"unknown TGAP source mode {source_mode!r}")
        self.root = Path(root)
        self.run_id = run_id
        self.source_mode = source_mode
        self.representation = representation
        self.layout = layout
        (self.root / "snapshots").mkdir(parents=True, exist_ok=True)
        (self.root / "starts").mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.jsonl"

    def write(
        self,
        time_index: int,
        client_id: int,
        start: AdapterState,
        end: AdapterState,
        num_samples: int,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        geom = infer_geometry(end)
        if infer_geometry(start) != geom:
            raise SnapshotError("start and end states have different geometry")
        start_hash, end_hash = start.sha256(), end.sha256()
        start_rel = Path("starts") / f"{start_hash}.safetensors"
        if not (self.root / start_rel).exists():
            start.save(self.root / start_rel, {"role": "start_state"})
        rel = Path("snapshots") / f"t{time_index:04d}_c{client_id:04d}.safetensors"
        end.save(
            self.root / rel,
            {
                "role": "end_state",
                "run_id": self.run_id,
                "time_index": str(time_index),
                "client_id": str(client_id),
            },
        )
        delta = end.sub(start)
        meta = self.layout.metadata(geom)
        record = {
            "schema": SNAPSHOT_SCHEMA,
            "run_id": self.run_id,
            "source_mode": self.source_mode,
            "time_index": int(time_index),
            "client_id": int(client_id),
            "representation_mode": self.representation,
            "derivable_representations": ["adapter_state", "adapter_delta"],
            "stored_tensors": "end_state (file) + start_state (start_file)",
            "start_adapter_hash": start_hash,
            "end_adapter_hash": end_hash,
            "start_file": start_rel.as_posix(),
            "start_file_sha256": sha256_file(self.root / start_rel),
            "num_samples": int(num_samples),
            "layout_id": self.layout.layout_id,
            "layout_key_order_sha256": meta["key_order_sha256"],
            "geometry": geom.to_dict(),
            "phi_shape": list(geom.phi_shape),
            "dtype": "float32",
            "l2": {
                "state_sq": end.l2_sq(),
                "state_A_sq": end.l2_sq("A"),
                "state_B_sq": end.l2_sq("B"),
                "delta_sq": delta.l2_sq(),
                "delta_A_sq": delta.l2_sq("A"),
                "delta_B_sq": delta.l2_sq("B"),
            },
            "file": rel.as_posix(),
            "file_sha256": sha256_file(self.root / rel),
            "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "extra": dict(extra or {}),
        }
        append_jsonl(self.index_path, record)
        return record


def read_index(root: str | Path) -> list[dict[str, Any]]:
    records = read_jsonl(Path(root) / "index.jsonl")
    for r in records:
        missing = [f for f in REQUIRED_FIELDS if f not in r]
        if missing or r["schema"] != SNAPSHOT_SCHEMA:
            raise SnapshotError(f"snapshot record missing {missing} or wrong schema")
    return records


def load_states(
    root: str | Path, record: Mapping[str, Any], verify: bool = True
) -> tuple[AdapterState, AdapterState]:
    root = Path(root)
    if verify:
        for key_file, key_sha in (("file", "file_sha256"), ("start_file", "start_file_sha256")):
            if sha256_file(root / record[key_file]) != record[key_sha]:
                raise SnapshotError(f"{record[key_file]}: file hash mismatch")
    end = AdapterState.load(root / record["file"])
    start = AdapterState.load(root / record["start_file"])
    if verify and (
        end.sha256() != record["end_adapter_hash"] or start.sha256() != record["start_adapter_hash"]
    ):
        raise SnapshotError("adapter hash mismatch")
    return start, end


def snapshot_tensor(
    root: str | Path, record: Mapping[str, Any], representation: str, layout: Layout
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(X, R)``: Phi of the requested representation and the matching innovation reference
    (Phi of the start state for ``adapter_state``; zeros for ``adapter_delta``)."""
    start, end = load_states(root, record)
    geom: LoRAGeometry = geometry_from_dict(record["geometry"])
    rep = to_representation(end, start, representation)
    x = layout.forward(rep, geom)
    ref = layout.forward(start, geom) if representation == "adapter_state" else torch.zeros_like(x)
    return x, ref
