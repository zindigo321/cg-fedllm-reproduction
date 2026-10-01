"""Tier-C end-to-end smoke pipeline (correctness only; every number it produces is PHASE2-SMOKE).

Stages (all sharing one in-process model bundle):

  lora_ft             uncompressed federated LoRA on D2 (codec=None, sample_weighted_mean)
  tgap_local          TGAP snapshots with source_mode=local_pretrain on D1 (default)
  tgap_federated      TGAP snapshots with source_mode=federated_pretrain on D1 (minimal path)
  ae                  ResNet-3 AE trained on the tgap_local snapshots
  faf_identity        FAF with IdentityCodec, adapter_state      -> must equal lora_ft
  faf_identity_delta  FAF with IdentityCodec, adapter_delta      -> equal up to float rounding
  faf_autoencoder     FAF with AutoEncoderCodec
  cent                cent_smoke_sample_matched: one pooled client, D2 union, same local recipe
  resume              faf_identity interrupted after round 0 and resumed -> must equal faf_identity
  [controls]          ConstantMeanCodec / GaussianNoiseCodec (only when ``run_controls``)
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from cg_fedllm.compression.autoencoder import load_autoencoder
from cg_fedllm.compression.codecs import (
    AutoEncoderCodec,
    ConstantMeanCodec,
    GaussianNoiseCodec,
    IdentityCodec,
)
from cg_fedllm.compression.layout import get_layout
from cg_fedllm.config import ExperimentConfig
from cg_fedllm.data.formatting import TokenizedExample
from cg_fedllm.federated.simulator import FederatedSimulator, SimulatorSpec
from cg_fedllm.models.adapter import AdapterState
from cg_fedllm.pipeline import ModelBundle, identity_config_sha256, make_heldout_fn
from cg_fedllm.tgap.collect import collect_federated_pretrain, collect_local_pretrain, local_pretrain_clients
from cg_fedllm.tgap.snapshots import SnapshotWriter, read_index, snapshot_tensor
from cg_fedllm.tgap.train_ae import train_autoencoder
from cg_fedllm.utils.io import atomic_write_json, atomic_write_text, load_tensors, read_json


def _spec(cfg: ExperimentConfig, num_clients: int, *, representation: str | None = None, namespace: str = "fl", stop: int | None = None, rounds: int | None = None, fraction: float | None = None) -> SimulatorSpec:
    f = cfg.federated
    return SimulatorSpec(
        num_clients=num_clients,
        client_fraction=f.client_fraction if fraction is None else fraction,
        num_rounds=f.num_rounds if rounds is None else rounds,
        aggregation=f.aggregation,
        representation=representation or f.representation,
        seed=cfg.run.seed,
        namespace=namespace,
        heldout_eval_every=f.heldout_eval_every,
        stop_after_round=stop,
    )


def _round_hashes(run_dir: Path) -> list[str]:
    out = []
    t = 0
    while (run_dir / "rounds" / f"r{t:04d}" / "DONE").exists():
        out.append(read_json(run_dir / "rounds" / f"r{t:04d}" / "round.json")["global_hash"])
        t += 1
    return out


def _compare(a_dir: Path, b_dir: Path, tolerance: float | None) -> dict[str, Any]:
    a = AdapterState.load(a_dir / "final_adapter.safetensors")
    b = AdapterState.load(b_dir / "final_adapter.safetensors")
    rel = a.relative_l2_diff(b)
    bitwise = a.equal(b)
    return {
        "a": a_dir.name,
        "b": b_dir.name,
        "bitwise_equal": bitwise,
        "round_hashes_equal": _round_hashes(a_dir) == _round_hashes(b_dir),
        "relative_l2": rel,
        "max_abs_diff": a.max_abs_diff(b),
        "tolerance": tolerance,
        "pass": bitwise if tolerance is None else (bitwise or rel <= tolerance),
    }


def run_smoke(
    cfg: ExperimentConfig,
    bundle: ModelBundle,
    d1: Sequence[Sequence[TokenizedExample]],
    d2: Sequence[Sequence[TokenizedExample]],
    heldout: list[TokenizedExample],
    out_root: Path,
    *,
    gpu_tolerance: float | None,
    run_controls: bool = False,
    eval_fn: Callable[[str, AdapterState | None], dict] | None = None,
) -> dict[str, Any]:
    """Run all smoke stages; ``gpu_tolerance=None`` demands bitwise equality (CPU)."""
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    n = len(d2)
    layout = get_layout(cfg.tgap.layout)
    ident = {"identity_config_sha256": identity_config_sha256(cfg)}
    heldout_fn = make_heldout_fn(cfg, bundle, heldout)
    initial = bundle.initial_state
    timings: dict[str, float] = {}
    results: dict[str, Any] = {"label": "PHASE2-SMOKE", "num_clients": n}

    def timed(name: str, fn: Callable[[], Any]) -> Any:
        t0 = time.time()
        out = fn()
        timings[name] = round(time.time() - t0, 2)
        return out

    def fl(name: str, codec, *, representation: str | None = None, stop: int | None = None, data=d2, rounds=None, fraction=None) -> dict:
        n_clients = len(data)
        sim = FederatedSimulator(
            bundle.trainer,
            _spec(cfg, n_clients, representation=representation, stop=stop, rounds=rounds, fraction=fraction),
            data,
            out_root / name,
            codec=codec,
            layout=layout if codec is not None else None,
            identity={**ident, "stage": name.split("__")[0]},
            heldout_fn=heldout_fn,
        )
        return sim.run(initial)

    results["lora_ft"] = timed("lora_ft", lambda: fl("lora_ft", None))

    # ---- TGAP collection (both source modes share the snapshot schema) ------------------------------
    t_cfg = cfg.tgap
    local_dir, fed_dir = out_root / "tgap_local", out_root / "tgap_federated"
    local_clients = local_pretrain_clients(n, t_cfg.client_fraction, cfg.run.seed)
    w_local = SnapshotWriter(local_dir, run_id=f"{cfg.run.name}/tgap_local", source_mode="local_pretrain", representation=t_cfg.representation, layout=layout)
    results["tgap_local"] = timed(
        "tgap_local", lambda: collect_local_pretrain(bundle.trainer, initial, d1, clients=local_clients, num_time_steps=t_cfg.num_time_steps, writer=w_local)
    )
    w_fed = SnapshotWriter(fed_dir, run_id=f"{cfg.run.name}/tgap_federated", source_mode="federated_pretrain", representation=t_cfg.representation, layout=layout)
    fed_spec = _spec(cfg, n, namespace="tgap_fed", rounds=t_cfg.num_time_steps, fraction=t_cfg.client_fraction)
    results["tgap_federated"] = timed(
        "tgap_federated",
        lambda: collect_federated_pretrain(bundle.trainer, initial, d1, spec=fed_spec, run_dir=fed_dir / "fl", identity={**ident, "stage": "tgap_federated"}, writer=w_fed),
    )

    # ---- AE training on the default (local_pretrain) snapshots ----------------------------------------
    ae_cfg = cfg.autoencoder
    records = read_index(local_dir)
    pairs = [snapshot_tensor(local_dir, r, ae_cfg.representation, get_layout(ae_cfg.layout)) for r in records]
    xs, refs = [p[0] for p in pairs], [p[1] for p in pairs]
    ae_dir = out_root / "ae"
    results["ae"] = timed(
        "ae",
        lambda: train_autoencoder(
            xs, refs, records, ae_cfg, device=bundle.device, out_dir=ae_dir,
            provenance={"smoke": True, "snapshot_source": "local_pretrain", "snapshot_dir": str(local_dir), "num_snapshots": len(records)},
        ),
    )
    ae, _ = load_autoencoder(ae_dir / "autoencoder.safetensors", device=bundle.device)

    # ---- FAF ----------------------------------------------------------------------------------------
    results["faf_identity"] = timed("faf_identity", lambda: fl("faf_identity", IdentityCodec(), representation="adapter_state"))
    results["faf_identity_delta"] = timed("faf_identity_delta", lambda: fl("faf_identity_delta", IdentityCodec(), representation="adapter_delta"))
    results["faf_autoencoder"] = timed("faf_autoencoder", lambda: fl("faf_autoencoder", AutoEncoderCodec(ae, device=bundle.device), representation=ae_cfg.representation))
    if run_controls:
        mean = load_tensors(ae_dir / "train_mean.safetensors")["x"]
        results["faf_constant_mean"] = timed("faf_constant_mean", lambda: fl("faf_constant_mean", ConstantMeanCodec(mean), representation=ae_cfg.representation))
        results["faf_gaussian_noise"] = timed("faf_gaussian_noise", lambda: fl("faf_gaussian_noise", GaussianNoiseCodec(1e-4), representation="adapter_state"))

    # ---- centralized (sample-matched) and N=1 equivalence ------------------------------------------------
    pooled = [[ex for client in d2 for ex in client]]
    results["cent"] = timed("cent", lambda: fl("cent_smoke_sample_matched", None, data=pooled, rounds=1, fraction=1.0))
    direct = bundle.trainer.train(initial, pooled[0], ("fl", 0, 0)).end_state
    cent_final = AdapterState.load(out_root / "cent_smoke_sample_matched" / "final_adapter.safetensors")
    n1 = {"bitwise_equal": direct.equal(cent_final), "relative_l2": cent_final.relative_l2_diff(direct), "tolerance": gpu_tolerance}
    n1["pass"] = n1["bitwise_equal"] if gpu_tolerance is None else (n1["bitwise_equal"] or n1["relative_l2"] <= gpu_tolerance)

    # ---- resume -----------------------------------------------------------------------------------------
    def resume() -> dict:
        first = fl("resume_check", IdentityCodec(), representation="adapter_state", stop=0)
        second = fl("resume_check", IdentityCodec(), representation="adapter_state")
        return {"first": first["status"], "second": second["status"]}

    results["resume"] = timed("resume", resume)

    checks = {
        "identity_state_vs_lora_ft": _compare(out_root / "faf_identity", out_root / "lora_ft", gpu_tolerance),
        "identity_delta_vs_lora_ft": _compare(out_root / "faf_identity_delta", out_root / "lora_ft", gpu_tolerance if gpu_tolerance is not None else 1e-6),
        "resume_vs_uninterrupted": _compare(out_root / "resume_check", out_root / "faf_identity", None),
        "n1_fl_vs_local": n1,
    }
    results["checks"] = checks

    results["stage_status"] = {k: v["status"] for k, v in results.items() if isinstance(v, dict) and "status" in v}

    if eval_fn is not None:
        t0 = time.time()
        evals = {"base": eval_fn("base", None)}
        for name in ("lora_ft", "faf_autoencoder"):
            final = out_root / name / "final_adapter.safetensors"
            if final.exists():
                evals[name] = eval_fn(name, AdapterState.load(final))
            else:
                evals[name] = {"skipped": f"no final adapter (run status: {results[name]['status']})"}
        results["eval"] = evals
        timings["eval"] = round(time.time() - t0, 2)

    results["timings_s"] = timings
    results["total_time_s"] = round(sum(timings.values()), 2)
    atomic_write_json(out_root / "smoke_summary.json", results)
    atomic_write_text(out_root / "smoke_summary.md", render_markdown(results))
    return results


def render_markdown(r: dict[str, Any]) -> str:
    lines = ["# Tier-C smoke summary (PHASE2-SMOKE — correctness run, not a paper reproduction)", ""]
    lines.append(f"clients: {r['num_clients']}; total stage time: {r.get('total_time_s')} s")
    lines += ["", "## Stage status", ""] + [f"- {k}: {v}" for k, v in r.get("stage_status", {}).items()]
    lines += ["", "## Equivalence checks", "", "| check | bitwise | relative L2 | tolerance | pass |", "|---|---|---|---|---|"]
    for k, c in r.get("checks", {}).items():
        lines.append(f"| {k} | {c.get('bitwise_equal')} | {c.get('relative_l2'):.3e} | {c.get('tolerance')} | {c.get('pass')} |")
    lines += ["", "## Held-out loss (PHASE2-SMOKE)", "", "| stage | initial | final round |", "|---|---|---|"]
    for k in ("lora_ft", "faf_identity", "faf_identity_delta", "faf_autoencoder", "cent", "faf_constant_mean", "faf_gaussian_noise"):
        s = r.get(k)
        if isinstance(s, dict) and "heldout_by_round" in s:
            last = s["heldout_by_round"][max(s["heldout_by_round"], key=lambda x: int(x))] if s["heldout_by_round"] else None
            init = s.get("initial_heldout", {}).get("loss")
            lines.append(f"| {k} | {init} | {last['loss'] if last else None} |")
    lines += ["", "## Uplink bytes (logical) per run", "", "| stage | logical | raw fp32 |", "|---|---|---|"]
    for k in ("lora_ft", "faf_identity", "faf_autoencoder", "cent"):
        s = r.get(k)
        if isinstance(s, dict):
            lines.append(f"| {k} | {s.get('uplink_logical_bytes_total')} | {s.get('uplink_raw_fp32_bytes_total')} |")
    ae = r.get("ae", {})
    if ae:
        lines += ["", "## AutoEncoder (PHASE2-SMOKE)", "", f"input {ae.get('input_shape')} -> latent {ae.get('latent_shape')}; CR {ae.get('compression_ratio_elements')}"]
        for split in ("train", "val"):
            for name in ("autoencoder", "zero", "train_mean"):
                m = ae.get(split, {}).get(name, {})
                lines.append(f"- {split}/{name}: pooled rel. sq. error {m.get('pooled_rel_sq_error')}, mean innovation ratio {m.get('mean_innovation_ratio')}")
    lines += ["", "## Stage timings (s)", ""] + [f"- {k}: {v}" for k, v in r.get("timings_s", {}).items()]
    return "\n".join(lines) + "\n"
