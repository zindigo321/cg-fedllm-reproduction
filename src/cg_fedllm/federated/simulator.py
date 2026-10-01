"""Sequential federated LoRA simulator (LoRA-FT baseline and FAF share this single code path).

One communication round t (reviewer Stage P12):

 1. the server's global adapter state ``G_t`` is "sent" to the selected clients (Shepherd sampler);
 2. every selected client trains locally from exactly ``G_t`` (fresh optimizer, seeds keyed by t and id);
 3. the configured representation is formed (``adapter_state`` or ``adapter_delta = end - G_t``);
 4. Phi (the configured layout) stacks it into ``X in R^{1 x d x W}``;
 5. the codec encodes X on the client;  6. the logical payload size is logged;
 7. the codec decodes on the server;    8. Phi^-1 restores the factor tensors;
 9. a delta representation is turned back into a state (``G_t + delta_hat``);
10. the recovered states are aggregated (A and B independently) with the configured strategy;
11. the aggregate replaces the global adapter;  12. an atomic round checkpoint is written;  13. repeat.

``codec=None`` is the *uncompressed* baseline: steps 3-9 are skipped and the clients' post-training
states are aggregated directly. It exists so that the IdentityCodec path can be shown to be lossless
(bitwise on CPU for ``adapter_state``); everything else is shared.

Resume: a round is complete iff ``rounds/rNNNN/DONE`` exists. All randomness is derived from
``(seed, namespace, round, client)``, so resuming after round k reproduces an uninterrupted run exactly.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from cg_fedllm.compression.codecs import Codec, CodecContext
from cg_fedllm.compression.layout import Layout, infer_geometry
from cg_fedllm.compression.metrics import json_safe, reconstruction_metrics
from cg_fedllm.compression.representation import recover_state, to_representation
from cg_fedllm.data.formatting import TokenizedExample
from cg_fedllm.federated.aggregation import aggregate
from cg_fedllm.federated.client import LocalTrainer
from cg_fedllm.federated.sampling import shepherd_select_clients
from cg_fedllm.models.adapter import AdapterState
from cg_fedllm.utils.io import atomic_write_json, atomic_write_text, read_json

SnapshotHook = Callable[[int, int, AdapterState, AdapterState, int, dict], None]
HeldoutFn = Callable[[AdapterState], dict]


class ResumeError(RuntimeError):
    pass


@dataclass
class SimulatorSpec:
    num_clients: int
    client_fraction: float
    num_rounds: int
    aggregation: str
    representation: str
    seed: int
    namespace: str = "fl"
    heldout_eval_every: int = 1
    stop_after_round: int | None = None


def round_dir(run_dir: Path, t: int) -> Path:
    return run_dir / "rounds" / f"r{t:04d}"


def completed_rounds(run_dir: Path) -> list[int]:
    base = run_dir / "rounds"
    if not base.exists():
        return []
    out = []
    for d in sorted(base.iterdir()):
        if d.is_dir() and d.name.startswith("r") and (d / "DONE").exists():
            out.append(int(d.name[1:]))
    return out


class FederatedSimulator:
    def __init__(
        self,
        trainer: LocalTrainer,
        spec: SimulatorSpec,
        client_examples: Sequence[Sequence[TokenizedExample]],
        run_dir: Path,
        *,
        codec: Codec | None,
        layout: Layout | None,
        identity: dict[str, Any],
        heldout_fn: HeldoutFn | None = None,
        snapshot_hook: SnapshotHook | None = None,
        result_label: str = "UNKNOWN",
    ) -> None:
        if len(client_examples) != spec.num_clients:
            raise ValueError(f"expected data for {spec.num_clients} clients, got {len(client_examples)}")
        if codec is not None and layout is None:
            raise ValueError("a codec requires a Phi layout")
        self.trainer = trainer
        self.spec = spec
        self.client_examples = client_examples
        self.run_dir = Path(run_dir)
        self.codec = codec
        self.layout = layout
        self.identity = identity
        self.heldout_fn = heldout_fn
        self.snapshot_hook = snapshot_hook
        self.result_label = result_label

    # ------------------------------------------------------------------------------------------------
    def _check_identity(self, initial: AdapterState) -> int:
        """Create or verify the run identity; return the first round to execute."""
        ident_path = self.run_dir / "run_identity.json"
        spec = {k: v for k, v in self.spec.__dict__.items() if k != "stop_after_round"}  # interruption is not identity
        ident = json.loads(json.dumps(json_safe({**self.identity, "initial_adapter_sha256": initial.sha256(), "spec": spec})))
        if ident_path.exists():
            old = read_json(ident_path)
            if old != ident:
                raise ResumeError(f"{self.run_dir} belongs to a different run configuration; refusing to resume")
            done = completed_rounds(self.run_dir)
            if done and done != list(range(len(done))):
                raise ResumeError(f"non-contiguous completed rounds {done}")
            return len(done)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(ident_path, ident)
        initial.save(self.run_dir / "initial_adapter.safetensors")
        return 0

    def _client_round(self, t: int, cid: int, global_state: AdapterState, geom) -> tuple[AdapterState, int, dict]:
        res = self.trainer.train(global_state, self.client_examples[cid], (self.spec.namespace, t, cid))
        rec: dict[str, Any] = {"client_id": cid, **res.summary()}
        if self.snapshot_hook is not None:
            self.snapshot_hook(t, cid, global_state, res.end_state, res.num_samples, rec)
        raw_numel = res.end_state.num_elements()
        if self.codec is None:
            recovered = res.end_state
            rec["payload"] = {"codec_id": "none", "numel": raw_numel, "logical_bytes": raw_numel * 4, "raw_fp32_bytes": raw_numel * 4}
        else:
            rep = to_representation(res.end_state, global_state, self.spec.representation)
            ctx = CodecContext(t, cid, self.spec.seed)
            x = self.layout.forward(rep, geom)
            t_enc = time.time()
            payload = self.codec.encode(x, ctx)
            t_dec = time.time()
            x_hat = self.codec.decode(payload, ctx)
            t_end = time.time()
            rep_hat = self.layout.inverse(x_hat, geom)
            recovered = recover_state(rep_hat, global_state, self.spec.representation)
            ref = self.layout.forward(global_state, geom) if self.spec.representation == "adapter_state" else torch.zeros_like(x)
            rec["payload"] = {
                "codec_id": self.codec.codec_id,
                "numel": payload.numel(),
                "input_numel": int(x.numel()),
                "logical_bytes": payload.logical_nbytes(),
                "serialized_bytes": payload.serialized_nbytes(),
                "raw_fp32_bytes": raw_numel * 4,
                "element_ratio": payload.numel() / x.numel(),
                "encode_s": round(t_dec - t_enc, 6),
                "decode_s": round(t_end - t_dec, 6),
            }
            rec["reconstruction"] = json_safe(reconstruction_metrics(x, x_hat, reference=ref))
            rec["state_relative_l2_error"] = recovered.relative_l2_diff(res.end_state)
            rec["recovered_hash"] = recovered.sha256()
            rec["recovered_equals_local_bitwise"] = recovered.equal(res.end_state)
        return recovered, res.num_samples, rec

    def run(self, initial: AdapterState) -> dict[str, Any]:
        spec = self.spec
        start_round = self._check_identity(initial)
        global_state = initial if start_round == 0 else AdapterState.load(round_dir(self.run_dir, start_round - 1) / "global_adapter.safetensors")
        if start_round == 0 and self.heldout_fn is not None:
            atomic_write_json(self.run_dir / "initial_eval.json", {"round": -1, "heldout": self.heldout_fn(initial), "global_hash": initial.sha256()})
        geom = infer_geometry(global_state) if self.codec is not None else None
        status = "complete"
        for t in range(start_round, spec.num_rounds):
            t0 = time.time()
            selected = shepherd_select_clients(spec.num_clients, spec.client_fraction, t)
            states, counts, records = [], [], []
            rdir = round_dir(self.run_dir, t)
            try:
                for cid in selected:
                    st, n, rec = self._client_round(t, cid, global_state, geom)
                    states.append(st)
                    counts.append(n)
                    records.append(rec)
                new_global = aggregate(states, counts, spec.aggregation)
                if not new_global.is_finite():
                    raise FloatingPointError("aggregated global adapter contains non-finite values")
            except FloatingPointError as exc:
                # A diverged run is a scientific outcome: record it explicitly (no DONE marker, no final
                # adapter) and stop; the summary status makes it impossible to mistake for a complete run.
                atomic_write_json(
                    rdir / "diverged.json",
                    json_safe({"round": t, "selected_clients": selected, "error": str(exc), "clients_completed": records}),
                )
                return self.summarize(global_state, f"diverged_in_round_{t}")
            new_global.save(rdir / "global_adapter.safetensors", {"round": str(t)})
            heldout = None
            if self.heldout_fn is not None and ((t + 1) % spec.heldout_eval_every == 0 or t == spec.num_rounds - 1):
                heldout = self.heldout_fn(new_global)
            record = {
                "round": t,
                "selected_clients": selected,
                "aggregation": spec.aggregation,
                "representation": spec.representation,
                "codec": self.codec.describe() if self.codec is not None else {"codec_id": "none"},
                "clients": records,
                "sample_counts": counts,
                "global_hash": new_global.sha256(),
                "global_l2_sq": {"A": new_global.l2_sq("A"), "B": new_global.l2_sq("B")},
                "uplink_logical_bytes": sum(r["payload"]["logical_bytes"] for r in records),
                "uplink_raw_fp32_bytes": sum(r["payload"]["raw_fp32_bytes"] for r in records),
                "heldout": heldout,
                "wall_time_s": round(time.time() - t0, 3),
            }
            atomic_write_json(rdir / "round.json", json_safe(record))
            atomic_write_text(rdir / "DONE", "ok\n")
            global_state = new_global
            if spec.stop_after_round is not None and t >= spec.stop_after_round and t < spec.num_rounds - 1:
                status = f"stopped_after_round_{t}"
                break
        return self.summarize(global_state, status)

    def summarize(self, final_state: AdapterState, status: str) -> dict[str, Any]:
        rounds = [read_json(round_dir(self.run_dir, t) / "round.json") for t in completed_rounds(self.run_dir)]
        summary = {
            "label": self.result_label,
            "status": status,
            "rounds_completed": len(rounds),
            "final_global_hash": final_state.sha256(),
            "heldout_by_round": {r["round"]: r["heldout"] for r in rounds if r.get("heldout") is not None},
            "uplink_logical_bytes_total": sum(r["uplink_logical_bytes"] for r in rounds),
            "uplink_raw_fp32_bytes_total": sum(r["uplink_raw_fp32_bytes"] for r in rounds),
        }
        init_eval = self.run_dir / "initial_eval.json"
        if init_eval.exists():
            summary["initial_heldout"] = read_json(init_eval)["heldout"]
        if status == "complete":
            final_state.save(self.run_dir / "final_adapter.safetensors")
        atomic_write_json(self.run_dir / "summary.json", json_safe(summary))
        return summary
