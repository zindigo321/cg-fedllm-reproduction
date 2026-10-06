"""Synthetic fixtures for the P5-A v2 runner and preflight tests (no real payload, no CUDA, no CGFED_RUNS).

Populations are random synthetic Phi tensors of a legal small geometry ([1, 128, 128] -> latent [64, 2, 2]); the
frozen facts are derived from them with independent arithmetic (Python maxima, float64 RMS).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import torch

from cg_fedllm.config import load_config
from cg_fedllm.phase5 import p5a
from cg_fedllm.phase5 import p5a_run as run
from cg_fedllm.phase5.p5a_artifacts import ArtifactStore
from cg_fedllm.phase5.p5a_inputs import FROZEN, Population
from cg_fedllm.phase5.p5a_preflight import run_preflight
from cg_fedllm.utils.hashing import tensor_sha256

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "phase5" / "p5a_v2_seed1.yaml"
H = W = 128
N = 5  # R4-like population: single [2], subset [0, 1, 3, 4]
CLIENTS = (2, 26, 55, 75, 86)
IDENTITY = {
    "execution_commit": "c" * 40,
    "config_sha256": "f" * 64,
    "launch_record": None,
    "command": "pytest",
}


def xs_for(seed: int, scale: float = 0.01) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(1, H, W, generator=g) * scale for _ in range(N)]


def population(rid: str, xs) -> Population:
    recs = [{"time_index": 0, "client_id": c, "file": f"s/{c}", "file_sha256": "f" * 64, "start_file": "st",
             "start_file_sha256": "e" * 64, "end_adapter_hash": "a" * 64, "start_adapter_hash": "b" * 64}
            for c in CLIENTS]  # fmt: skip
    rows = [{"position": i, "time_index": 0, "client_id": c, "phi_sha256": tensor_sha256(x)}
            for i, (c, x) in enumerate(zip(CLIENTS, xs))]  # fmt: skip
    prov = {
        "root": "syn",
        "index_sha256": "9" * 64,
        "n_train": N,
        "records": rows,
        "gradient_files_read": None,
    }
    return Population(rid, recs, list(xs), prov)


def frozen_for(rid: str, xs):
    m = max(abs(v) for x in xs for v in x.reshape(-1).tolist())
    flat = [v for x in xs for v in x.reshape(-1).double().tolist()]
    rms = float(
        torch.stack(list(xs)).pow(2).mean().sqrt()
    )  # the F6 expression; checked vs a float64 oracle below
    assert abs(rms - (sum(v * v for v in flat) / len(flat)) ** 0.5) <= 1e-6 * rms
    geom = {"num_layers": 1, "modules": ["q_proj", "k_proj", "v_proj", "o_proj"], "rank": 16, "hidden": H}
    return dataclasses.replace(FROZEN[rid], max_abs=m, scale=m / 0.95, train_rms=rms, geometry=geom,
                               phi_shape=(1, H, W), n_train=N)  # fmt: skip


def cfg():
    return load_config(CONFIG)


class FakeClock:
    def __init__(self, step: float = 1.0) -> None:
        self.t, self.step = 0.0, step

    def __call__(self) -> float:
        self.t += self.step
        return self.t


def loader(pops):
    return lambda frozen: population(frozen.rid, pops[frozen.rid])


def context(root: Path, pops, iterations: int = 2, **kw) -> run.RunContext:
    frozen = {rid: frozen_for(rid, xs) for rid, xs in pops.items()}
    for rid in p5a.REPRESENTATIONS:  # representations without a population still need a frozen entry
        frozen.setdefault(rid, FROZEN[rid])
    defaults = dict(
        profile=run.synthetic_profile(frozen, iterations),
        store=ArtifactStore(root),
        ae_section=cfg().autoencoder,
        device=torch.device("cpu"),
        identity=dict(IDENTITY),
        clock=FakeClock(),
        run_files={"config.resolved.yaml": b"x: 1\n"},
    )
    defaults.update(kw)
    return run.RunContext(**defaults)


def preflight(ctx: run.RunContext, pops, identity=None) -> dict:
    """The CPU preflight on the same synthetic inputs (it must pass before any training invocation)."""
    return run_preflight(
        ctx.store, identity=identity or ctx.identity, load=loader(pops), frozen=ctx.profile.frozen
    )


def all_pops(seed: int = 0):
    return {"R2": xs_for(seed), "R3": xs_for(seed + 1), "R4": xs_for(seed + 2)}


def fake_trainer(recon):
    """A trainer stand-in with the real interface: ``recon(xs) -> reconstructions``; the clock runs normally."""

    def trainer(ae, xs, norm, *, iterations, device, gpu):
        gpu.start()
        gpu.stop()
        return {
            "curve": [],
            "reconstructions": recon(xs),
            "state": ae.state_dict(),
            "iterations_done": iterations,
        }

    return trainer


def exact(xs):
    return [x.clone() for x in xs]
