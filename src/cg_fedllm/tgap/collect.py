"""TGAP snapshot collection (reviewer decision R6: the source mode is scientifically ambiguous).

``local_pretrain``     -- (v1 wording: "train M_i with its own data D_i without FL"). Each participating
                          client starts from the same initial adapter and runs ``num_time_steps`` local
                          rounds on D1, each continuing from its *own* previous state with a fresh
                          optimizer; one snapshot per (time step, client); no server aggregation.
``federated_pretrain`` -- (ECAI/v3 wording: "D1 is utilized for the pre-training of FedLLM"). A normal
                          FL loop (uncompressed, configured aggregation) over D1; one snapshot per selected
                          client per round.

Both write the same snapshot schema. Neither is claimed to be the paper's procedure.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from cg_fedllm.data.formatting import TokenizedExample
from cg_fedllm.federated.client import LocalTrainer
from cg_fedllm.federated.sampling import num_selected
from cg_fedllm.federated.simulator import FederatedSimulator, SimulatorSpec
from cg_fedllm.models.adapter import AdapterState
from cg_fedllm.tgap.snapshots import SnapshotWriter
from cg_fedllm.utils.seeding import numpy_rng


def local_pretrain_clients(num_clients: int, fraction: float, seed: int) -> list[int]:
    if fraction >= 1.0:
        return list(range(num_clients))
    k = num_selected(num_clients, fraction)
    return sorted(int(c) for c in numpy_rng(seed, "tgap_local_clients").choice(num_clients, size=k, replace=False))


def collect_local_pretrain(
    trainer: LocalTrainer,
    initial: AdapterState,
    client_examples: Sequence[Sequence[TokenizedExample]],
    *,
    clients: Sequence[int],
    num_time_steps: int,
    writer: SnapshotWriter,
) -> dict[str, Any]:
    n = 0
    for cid in clients:
        state = initial
        for t in range(num_time_steps):
            res = trainer.train(state, client_examples[cid], ("tgap_local", t, cid))
            writer.write(t, cid, state, res.end_state, res.num_samples, res.summary())
            state = res.end_state
            n += 1
    return {"source_mode": "local_pretrain", "clients": list(clients), "num_time_steps": num_time_steps, "num_snapshots": n}


def collect_federated_pretrain(
    trainer: LocalTrainer,
    initial: AdapterState,
    client_examples: Sequence[Sequence[TokenizedExample]],
    *,
    spec: SimulatorSpec,
    run_dir: Path,
    identity: dict[str, Any],
    writer: SnapshotWriter,
) -> dict[str, Any]:
    counter = {"n": 0}

    def hook(t: int, cid: int, start: AdapterState, end: AdapterState, n_samples: int, rec: dict) -> None:
        writer.write(t, cid, start, end, n_samples, {k: v for k, v in rec.items() if k != "payload"})
        counter["n"] += 1

    sim = FederatedSimulator(trainer, spec, client_examples, run_dir, codec=None, layout=None, identity=identity, snapshot_hook=hook)
    summary = sim.run(initial)
    return {"source_mode": "federated_pretrain", "num_snapshots": counter["n"], "fl_summary": summary}
