"""Command-line entry point: ``cgfed <command> --config <yaml> [--set key.path=value ...]``.

Commands: capture-env, prepare-data, run-fl, collect-tgap, train-ae, evaluate, smoke, bench-gpu, and the
Phase-3 diagnostics calibrate-train, microbatch-diag, tgap-stats, ae-viability, ae-select, reference-codes.
Determinism settings and the optional CUDA allocator cap (``run.allocator_cap_margin_mb``) are applied before
any model or CUDA work.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from cg_fedllm.config import RESULT_LABELS, ExperimentConfig, load_config, resolve_path
from cg_fedllm.utils.io import atomic_write_json, read_json
from cg_fedllm.utils.provenance import collect_environment
from cg_fedllm.utils.seeding import configure_determinism


def _provenance() -> dict[str, Any]:
    """Git state and key package versions for lightweight result files that have no run directory."""
    from cg_fedllm.utils.provenance import git_info, package_versions

    return {"git": git_info(), "packages": package_versions()}


def _load(args) -> ExperimentConfig:
    cfg = load_config(args.config, args.set)
    configure_determinism(cfg.run.deterministic, cfg.run.num_threads)
    return cfg


def cmd_capture_env(args) -> dict:
    env = collect_environment()
    if args.out:
        atomic_write_json(Path(args.out), env)
    return env


def cmd_prepare_data(args) -> dict:
    from cg_fedllm.pipeline import prepare_manifest

    cfg = _load(args)
    manifest, status = prepare_manifest(cfg, write=args.write)
    sizes = [len(c["ids"]) for c in manifest.data["clients"]]
    return {
        **status,
        "num_clients": len(sizes),
        "min_client": min(sizes),
        "max_client": max(sizes),
        "holdout": len(manifest.holdout_ids),
    }


def cmd_run_fl(args) -> dict:
    from cg_fedllm.federated.simulator import FederatedSimulator
    from cg_fedllm.pipeline import (
        apply_vram_guard,
        build_codec,
        build_data_bundle,
        build_model_bundle,
        client_examples,
        heldout_examples,
        identity_config_sha256,
        init_run_dir,
        layout_for,
        make_heldout_fn,
        simulator_spec,
    )

    cfg = _load(args)
    cfg.require("federated", "data")
    guard = apply_vram_guard(cfg)
    bundle = build_model_bundle(cfg)
    data = build_data_bundle(cfg)
    tok = bundle.loaded.tokenizer
    clients = client_examples(cfg, data, tok, cfg.federated.client_split)
    if cfg.federated.pooled:
        clients = [[ex for c in clients for ex in c]]
    run_dir = init_run_dir(
        cfg,
        args.stage or "fl",
        {"model": bundle.loaded.info, "manifest_sha256": data.manifest_sha256, "vram_guard": guard},
    )
    codec = build_codec(cfg, bundle.device)
    sim = FederatedSimulator(
        bundle.trainer,
        simulator_spec(cfg, len(clients)),
        clients,
        run_dir,
        codec=codec,
        layout=layout_for(cfg) if codec is not None else None,
        identity={
            "identity_config_sha256": identity_config_sha256(cfg),
            "manifest_sha256": data.manifest_sha256,
        },
        heldout_fn=make_heldout_fn(cfg, bundle, heldout_examples(cfg, data, tok)),
        result_label=cfg.run.result_label,
    )
    return sim.run(bundle.initial_state)


def cmd_collect_tgap(args) -> dict:
    from cg_fedllm.compression.layout import get_layout
    from cg_fedllm.pipeline import (
        apply_vram_guard,
        build_data_bundle,
        build_model_bundle,
        client_examples,
        identity_config_sha256,
        init_run_dir,
        simulator_spec,
    )
    from cg_fedllm.tgap.collect import (
        collect_federated_pretrain,
        collect_local_pretrain,
        local_pretrain_clients,
    )
    from cg_fedllm.tgap.snapshots import SnapshotWriter

    cfg = _load(args)
    cfg.require("tgap", "data")
    guard = apply_vram_guard(cfg)
    bundle = build_model_bundle(cfg)
    data = build_data_bundle(cfg)
    clients = client_examples(cfg, data, bundle.loaded.tokenizer, cfg.tgap.client_split)
    run_dir = init_run_dir(
        cfg,
        args.stage or f"tgap_{cfg.tgap.source}",
        {"model": bundle.loaded.info, "manifest_sha256": data.manifest_sha256, "vram_guard": guard},
    )
    writer = SnapshotWriter(
        run_dir,
        run_id=f"{cfg.run.name}/{run_dir.name}",
        source_mode=cfg.tgap.source,
        representation=cfg.tgap.representation,
        layout=get_layout(cfg.tgap.layout),
    )
    if cfg.tgap.source == "local_pretrain":
        ids = local_pretrain_clients(
            len(clients), cfg.tgap.client_fraction, cfg.run.seed, cfg.tgap.local_client_selection
        )
        t0 = time.time()
        out = collect_local_pretrain(
            bundle.trainer,
            bundle.initial_state,
            clients,
            clients=ids,
            num_time_steps=cfg.tgap.num_time_steps,
            writer=writer,
        )
        out.update(
            {
                "local_client_selection": cfg.tgap.local_client_selection,
                "wall_time_s": round(time.time() - t0, 1),
                "label": cfg.run.result_label,
            }
        )
        atomic_write_json(run_dir / "collection_summary.json", out)
        return out
    cfg.require("federated")
    spec = simulator_spec(cfg, len(clients), namespace="tgap_fed")
    spec.num_rounds, spec.client_fraction = cfg.tgap.num_time_steps, cfg.tgap.client_fraction
    t0 = time.time()
    out = collect_federated_pretrain(
        bundle.trainer,
        bundle.initial_state,
        clients,
        spec=spec,
        run_dir=run_dir / "fl",
        identity={"identity_config_sha256": identity_config_sha256(cfg)},
        writer=writer,
        result_label=cfg.run.result_label,
    )
    out.update({"wall_time_s": round(time.time() - t0, 1), "label": cfg.run.result_label})
    atomic_write_json(run_dir / "collection_summary.json", out)
    return out


def cmd_train_ae(args) -> dict:
    import torch

    from cg_fedllm.compression.layout import geometry_from_dict, get_layout
    from cg_fedllm.pipeline import apply_vram_guard, init_run_dir
    from cg_fedllm.tgap.snapshots import read_index, snapshot_tensor
    from cg_fedllm.tgap.train_ae import train_autoencoder
    from cg_fedllm.utils.hashing import sha256_file

    cfg = _load(args)
    cfg.require("autoencoder")
    guard = apply_vram_guard(cfg)
    snap_dir = resolve_path(args.snapshots)
    records = read_index(snap_dir)
    layout = get_layout(cfg.autoencoder.layout)
    pairs = [snapshot_tensor(snap_dir, r, cfg.autoencoder.representation, layout) for r in records]
    index_sha = sha256_file(snap_dir / "index.jsonl")
    run_dir = init_run_dir(
        cfg,
        args.stage or "ae",
        {"snapshot_dir": str(snap_dir), "snapshot_index_sha256": index_sha, "vram_guard": guard},
    )
    return train_autoencoder(
        [p[0] for p in pairs],
        [p[1] for p in pairs],
        records,
        cfg.autoencoder,
        device=torch.device(cfg.run.device),
        out_dir=run_dir,
        provenance={
            "smoke": False,
            "snapshot_dir": str(snap_dir),
            "snapshot_index_sha256": index_sha,
            "num_snapshots": len(records),
            "source_mode": records[0]["source_mode"],
        },
        label=cfg.run.result_label,
        layout=layout,
        geom=geometry_from_dict(records[0]["geometry"]),
    )


def cmd_evaluate(args) -> dict:
    import torch

    from cg_fedllm.evaluation.reference_eval import evaluate_benchmark
    from cg_fedllm.models.adapter import AdapterState
    from cg_fedllm.models.loading import load_model
    from cg_fedllm.models.lora import attach_lora, lora_parameters, set_adapter_state
    from cg_fedllm.pipeline import apply_vram_guard, init_run_dir

    cfg = _load(args)
    cfg.require("model", "eval")
    guard = apply_vram_guard(cfg)
    device = torch.device(cfg.run.device)
    loaded = load_model(cfg.model, device)
    model = loaded.model
    adapter_info = None
    if args.adapter:
        cfg.require("lora")
        model = attach_lora(model, cfg.lora)
        state = AdapterState.load(resolve_path(args.adapter))
        set_adapter_state(lora_parameters(model), state)
        adapter_info = {"path": str(resolve_path(args.adapter)), "adapter_sha256": state.sha256()}
    run_dir = init_run_dir(
        cfg, args.stage or "eval", {"model": loaded.info, "adapter": adapter_info, "vram_guard": guard}
    )
    results = {}
    for spec in cfg.eval.benchmarks:
        r = evaluate_benchmark(
            model,
            loaded.tokenizer,
            spec,
            cfg.eval,
            device=device,
            pad_token_id=loaded.pad_token_id,
            out_dir=run_dir,
            tag=args.tag or "",
            label=cfg.run.result_label,
        )
        results[f"{spec.name}/{spec.split}"] = {
            "aggregates": {k: v for k, v in r["aggregates"].items() if k != "per_subject"},
            "scoring": r["scoring"],
            "timing_s": r["timing_s"],
        }
    atomic_write_json(run_dir / "eval_summary.json", results)
    return results


def cmd_smoke(args) -> dict:

    from cg_fedllm.evaluation.reference_eval import evaluate_benchmark
    from cg_fedllm.models.lora import set_adapter_state
    from cg_fedllm.pipeline import (
        build_data_bundle,
        build_model_bundle,
        client_examples,
        heldout_examples,
        init_run_dir,
        prepare_manifest,
    )
    from cg_fedllm.smoke import run_smoke

    cfg = _load(args)
    manifest, mstatus = prepare_manifest(cfg, write=False)
    bundle = build_model_bundle(cfg)
    data = build_data_bundle(cfg)
    tok = bundle.loaded.tokenizer
    run_dir = init_run_dir(cfg, args.stage or "smoke", {"model": bundle.loaded.info, "manifest": mstatus})
    d1 = client_examples(cfg, data, tok, "d1")
    d2 = client_examples(cfg, data, tok, "d2")
    heldout = heldout_examples(cfg, data, tok)

    eval_fn = None
    if cfg.eval is not None and cfg.eval.benchmarks:

        def eval_fn(name, state):
            if state is None:
                with bundle.peft_model.disable_adapter():
                    return _eval_all(name)
            set_adapter_state(bundle.params, state)
            return _eval_all(name)

        def _eval_all(name):
            out = {}
            for spec in cfg.eval.benchmarks:
                r = evaluate_benchmark(
                    bundle.peft_model,
                    tok,
                    spec,
                    cfg.eval,
                    device=bundle.device,
                    pad_token_id=bundle.loaded.pad_token_id,
                    out_dir=run_dir / "eval",
                    tag=name,
                    label=cfg.run.result_label,
                )
                out[f"{spec.name}/{spec.split}"] = {
                    k: v for k, v in r["aggregates"].items() if k != "per_subject"
                }
            return out

    tol = None if cfg.run.device == "cpu" else args.gpu_tolerance
    return run_smoke(
        cfg, bundle, d1, d2, heldout, run_dir, gpu_tolerance=tol, run_controls=args.controls, eval_fn=eval_fn
    )


def cmd_bench_gpu(args) -> dict:
    import yaml

    from cg_fedllm.bench import bench_config
    from cg_fedllm.config import LoRASection, ModelSection, build_dataclass
    from cg_fedllm.utils.gpu import cap_allocator_to_free_vram

    spec = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    configure_determinism(False, None)
    lora = build_dataclass(LoRASection, spec["lora"], "lora")
    vram_guard = cap_allocator_to_free_vram()
    out: dict[str, Any] = {
        "label": "LOCAL-MICROBENCH",
        "environment": collect_environment(),
        "vram_guard": vram_guard,
        "results": [],
    }
    for entry in spec["benchmarks"]:
        mcfg = build_dataclass(ModelSection, entry["model"], "model")
        for gc_opt in entry["gradient_checkpointing"]:
            out["results"] += bench_config(
                mcfg,
                lora,
                seq_len=entry["seq_len"],
                micro_batches=entry["micro_batches"],
                gradient_checkpointing=gc_opt,
                steps=spec.get("steps", 5),
                warmup=spec.get("warmup", 2),
            )
            atomic_write_json(resolve_path(spec["output"]), out)
    return out


def _schedule(num_clients: int, rounds: int, fraction: float) -> list[tuple[int, list[int]]]:
    from cg_fedllm.federated.sampling import shepherd_select_clients

    return [(t, shepherd_select_clients(num_clients, fraction, t)) for t in range(rounds)]


def cmd_calibrate_train(args) -> dict:
    """A3: realistic-sequence timing/memory of the real trainer on real Dolly client data (PHASE3-DIAGNOSTIC)."""
    import torch

    from cg_fedllm.calibration import timing_run, token_length_report
    from cg_fedllm.pipeline import (
        apply_vram_guard,
        build_data_bundle,
        build_model_bundle,
        client_examples,
        init_run_dir,
    )
    from cg_fedllm.tgap.collect import local_pretrain_clients

    cfg = _load(args)
    cfg.require("federated", "data", "tgap")
    guard = apply_vram_guard(cfg)
    t_load = time.time()
    bundle = build_model_bundle(cfg)
    load_s = time.time() - t_load
    weights_bytes = int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else None
    data = build_data_bundle(cfg)
    tok = bundle.loaded.tokenizer
    d1, d2 = client_examples(cfg, data, tok, "d1"), client_examples(cfg, data, tok, "d2")
    n = len(d2)
    local_ids = local_pretrain_clients(
        n, cfg.tgap.client_fraction, cfg.run.seed, cfg.tgap.local_client_selection
    )
    schedules = {
        "fl": ("d2", _schedule(n, cfg.federated.num_rounds, cfg.federated.client_fraction)),
        "tgap_fed": ("d1", _schedule(n, cfg.tgap.num_time_steps, cfg.tgap.client_fraction)),
        "tgap_local": ("d1", [(t, local_ids) for t in range(cfg.tgap.num_time_steps)]),
    }
    lengths = token_length_report(
        {"d1": d1, "d2": d2}, schedules, cutoff=cfg.data.cutoff_len, cfg=cfg.local_train, seed=cfg.run.seed
    )
    run_dir = init_run_dir(
        cfg,
        args.stage or "calibration",
        {"model": bundle.loaded.info, "manifest_sha256": data.manifest_sha256, "vram_guard": guard},
    )
    pool = sorted((ex for c in d1 + d2 for ex in c), key=lambda e: (-e.num_tokens, e.source_id))
    round0 = schedules["fl"][1][0][1]
    timing = timing_run(
        bundle.trainer,
        bundle.initial_state,
        d2,
        clients=round0,
        namespace="fl",
        round_index=0,
        worst_case=pool[: cfg.local_train.micro_batch_size],
        allocator_cap_bytes=guard["allocator_cap_bytes"] if guard else None,
    )
    out = {
        "label": cfg.run.result_label,
        "model_load_s": round(load_s, 1),
        "weights_allocated_bytes_after_load": weights_bytes,
        "vram_guard": guard,
        "token_lengths": lengths,
        "timing": timing,
        "config": {
            "micro_batch_size": cfg.local_train.micro_batch_size,
            "batch_size": cfg.local_train.batch_size,
            "gradient_checkpointing": cfg.model.gradient_checkpointing,
            "dtype": cfg.model.dtype,
            "quantization": cfg.model.quantization,
            "attn_implementation": cfg.model.attn_implementation,
            "deterministic": cfg.run.deterministic,
        },
    }
    atomic_write_json(run_dir / "calibration.json", out)
    return {k: out[k] for k in ("label", "model_load_s", "vram_guard")} | {
        "decision": timing["decision"],
        "totals": timing["totals"],
    }


def cmd_microbatch_diag(args) -> dict:
    """Micro-batch loss-normalisation diagnostic on one fixed real batch (PHASE3-DIAGNOSTIC)."""
    from cg_fedllm.calibration import microbatch_gradient_diagnostic
    from cg_fedllm.federated.sampling import shepherd_select_clients
    from cg_fedllm.models.adapter import AdapterState
    from cg_fedllm.pipeline import (
        apply_vram_guard,
        build_data_bundle,
        build_model_bundle,
        client_examples,
        init_run_dir,
    )
    from cg_fedllm.utils.seeding import numpy_rng

    cfg = _load(args)
    cfg.require("federated", "data")
    guard = apply_vram_guard(cfg)
    bundle = build_model_bundle(cfg)
    data = build_data_bundle(cfg)
    d2 = client_examples(cfg, data, bundle.loaded.tokenizer, "d2")
    round0 = shepherd_select_clients(len(d2), cfg.federated.client_fraction, 0)
    cid = max(round0, key=lambda c: (len(d2[c]), -c))
    perm = numpy_rng(cfg.run.seed, "fl", 0, cid, "data_order").permutation(len(d2[cid]))
    batch = [d2[cid][int(j)] for j in perm[: cfg.local_train.batch_size]]
    adapter = str(resolve_path(args.adapter)) if args.adapter else "initial"
    start = AdapterState.load(resolve_path(args.adapter)) if args.adapter else bundle.initial_state
    run_dir = init_run_dir(cfg, args.stage or "microbatch_diag", {"vram_guard": guard, "adapter": adapter})
    t0 = time.time()
    res = microbatch_gradient_diagnostic(
        bundle.trainer,
        start,
        batch,
        real_micro_batch=cfg.local_train.micro_batch_size,
        reference=cfg.local_train.micro_batch_size,
    )
    out = {
        "label": cfg.run.result_label,
        "batch": {
            "rule": "first optimizer step of FL round 0 for the round-0 client with the most D2 examples, in the trainer's data order",
            "client_id": cid,
            "source_ids": [e.source_id for e in batch],
        },
        "start_adapter": {"path_name": Path(adapter).name, "sha256": start.sha256()},
        "paper_micro_batch": 16,
        "elapsed_s": round(time.time() - t0, 1),
        **res,
    }
    atomic_write_json(run_dir / "microbatch_diag.json", out)
    return {k: out[k] for k in ("label", "batch", "elapsed_s", "real_vs_emulated_micro_batch")}


def cmd_tgap_stats(args) -> dict:
    from cg_fedllm.compression.metrics import json_safe
    from cg_fedllm.tgap.snapshots import read_index
    from cg_fedllm.tgap.stats import tgap_statistics
    from cg_fedllm.utils.hashing import sha256_file

    cfg = _load(args)
    cfg.require("autoencoder")
    snap_dir = resolve_path(args.snapshots)
    records = read_index(snap_dir)
    stats = tgap_statistics(
        snap_dir,
        records,
        split=cfg.autoencoder.split,
        val_fraction=cfg.autoencoder.val_fraction,
        split_seed=cfg.autoencoder.split_seed,
    )
    summary_path = snap_dir / "collection_summary.json"
    fields = (
        "time_index",
        "client_id",
        "num_samples",
        "start_adapter_hash",
        "end_adapter_hash",
        "file",
        "file_sha256",
        "start_file",
        "start_file_sha256",
        "l2",
    )
    out = json_safe(
        {
            "label": "DERIVED",
            "source_mode": records[0]["source_mode"],
            "snapshot_index_sha256": sha256_file(snap_dir / "index.jsonl"),
            "collection": read_json(summary_path) if summary_path.exists() else None,
            "geometry": records[0]["geometry"],
            "phi_shape": records[0]["phi_shape"],
            "layout_id": records[0]["layout_id"],
            **stats,
            "manifest": [{k: r[k] for k in fields} | {"training": r.get("extra", {})} for r in records],
            "provenance": _provenance(),
        }
    )
    if args.out:
        atomic_write_json(resolve_path(args.out), out)
    return {
        k: out[k]
        for k in (
            "num_snapshots",
            "num_time_indices",
            "unique_clients",
            "participation_histogram",
            "distinct_start_states",
        )
    }


def cmd_ae_viability(args) -> dict:
    import torch

    from cg_fedllm.compression.layout import get_layout
    from cg_fedllm.tgap.snapshots import read_index
    from cg_fedllm.tgap.viability import run_viability
    from cg_fedllm.utils.hashing import sha256_file

    cfg = _load(args)
    cfg.require("autoencoder")
    a = cfg.autoencoder
    snap_dir, ae_dir = resolve_path(args.snapshots), resolve_path(args.ae_dir)
    records = read_index(snap_dir)
    t0 = time.time()
    rep = run_viability(
        snap_dir,
        records,
        ae_dir,
        representation=a.representation,
        layout=get_layout(a.layout),
        split=a.split,
        val_fraction=a.val_fraction,
        device=torch.device(cfg.run.device),
        split_seed=a.split_seed,
    )
    keep = (
        "curve",
        "best_val",
        "train_time_s",
        "latent_shape",
        "input_shape",
        "compression_ratio_elements",
        "latent_logical_bytes_fp32",
        "raw_logical_bytes_fp32",
        "normalization_uplink_bytes",
        "parameters",
        "normalized_range",
        "config",
        "label",
    )
    out = {
        "label": cfg.run.result_label,
        "source_mode": records[0]["source_mode"],
        "snapshot_index_sha256": sha256_file(snap_dir / "index.jsonl"),
        "ae_run": {
            "dir_name": ae_dir.name,
            "checkpoint_sha256": {
                f: sha256_file(ae_dir / f)
                for f in ("autoencoder_best_val.safetensors", "autoencoder.safetensors")
            },
        },
        "ae_training": {k: v for k, v in read_json(ae_dir / "ae_metrics.json").items() if k in keep},
        "ae_run_git": read_json(ae_dir / "run_metadata.json").get("environment", {}).get("git")
        if (ae_dir / "run_metadata.json").exists()
        else None,
        "provenance": _provenance(),
        **rep,
    }
    out["elapsed_s"] = round(time.time() - t0, 1)
    if args.out:
        atomic_write_json(resolve_path(args.out), out)
    return {"gate": out["gate"], "elapsed_s": out["elapsed_s"]}


def cmd_ae_select(args) -> dict:
    from cg_fedllm.tgap.viability import select_primary

    gates, files = {}, {}
    for item in args.gate:
        mode, path = item.split("=", 1)
        gates[mode] = read_json(resolve_path(path))["gate"]
        files[mode] = path
    out = {
        "label": args.label,
        "inputs": files,
        "gates": {
            m: {"pass": g["pass"], "criteria": {k: c["pass"] for k, c in g["criteria"].items()}}
            for m, g in gates.items()
        },
        **select_primary(gates),
        "provenance": _provenance(),
    }
    if args.out:
        atomic_write_json(resolve_path(args.out), out)
    return out


def cmd_reference_codes(args) -> dict:
    """DERIVED interpretation aid: simple codes at the AE's element ratio on the validation snapshots."""
    from cg_fedllm.compression.layout import geometry_from_dict, get_layout
    from cg_fedllm.compression.metrics import json_safe
    from cg_fedllm.tgap.compressibility import reference_code_report
    from cg_fedllm.tgap.snapshots import read_index
    from cg_fedllm.utils.hashing import sha256_file

    cfg = _load(args)
    cfg.require("autoencoder")
    a = cfg.autoencoder
    snap_dir = resolve_path(args.snapshots)
    records = read_index(snap_dir)
    t0 = time.time()
    rep = reference_code_report(
        snap_dir,
        records,
        representation=a.representation,
        layout=get_layout(a.layout),
        geom=geometry_from_dict(records[0]["geometry"]),
        split=a.split,
        val_fraction=a.val_fraction,
        split_seed=a.split_seed,
    )
    out = json_safe(
        {
            "source_mode": records[0]["source_mode"],
            "snapshot_index_sha256": sha256_file(snap_dir / "index.jsonl"),
            **rep,
            "elapsed_s": round(time.time() - t0, 1),
            "provenance": _provenance(),
        }
    )
    if args.out:
        atomic_write_json(resolve_path(args.out), out)
    return {"label": out["label"], "elapsed_s": out["elapsed_s"], "train_pca_rank": out["train_pca_rank"]}


COMMANDS = {
    "capture-env": cmd_capture_env,
    "prepare-data": cmd_prepare_data,
    "run-fl": cmd_run_fl,
    "collect-tgap": cmd_collect_tgap,
    "train-ae": cmd_train_ae,
    "evaluate": cmd_evaluate,
    "smoke": cmd_smoke,
    "bench-gpu": cmd_bench_gpu,
    "calibrate-train": cmd_calibrate_train,
    "microbatch-diag": cmd_microbatch_diag,
    "tgap-stats": cmd_tgap_stats,
    "ae-viability": cmd_ae_viability,
    "ae-select": cmd_ae_select,
    "reference-codes": cmd_reference_codes,
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cgfed", description="CG-FedLLM reproduction toolkit")
    sub = p.add_subparsers(dest="command", required=True)
    for name in COMMANDS:
        sp = sub.add_parser(name)
        if name not in ("capture-env", "ae-select"):
            sp.add_argument("--config", required=True)
            sp.add_argument("--set", action="append", default=[], help="override, e.g. --set run.seed=7")
            sp.add_argument("--stage", default=None, help="run sub-directory name")
        if name == "capture-env":
            sp.add_argument("--out", default=None)
        if name == "prepare-data":
            sp.add_argument("--write", action="store_true", help="(re)write the committed manifest")
        if name in ("train-ae", "tgap-stats", "ae-viability", "reference-codes"):
            sp.add_argument("--snapshots", required=True)
        if name in ("tgap-stats", "ae-viability", "ae-select", "reference-codes"):
            sp.add_argument("--out", default=None)
        if name == "ae-viability":
            sp.add_argument("--ae-dir", required=True)
        if name == "microbatch-diag":
            sp.add_argument("--adapter", default=None, help="start adapter (default: the initial LoRA state)")
        if name == "ae-select":
            sp.add_argument("--gate", action="append", required=True, help="mode=path/to/viability.json")
            sp.add_argument("--label", required=True, choices=RESULT_LABELS)
        if name == "evaluate":
            sp.add_argument("--adapter", default=None)
            sp.add_argument("--tag", default=None)
        if name == "smoke":
            sp.add_argument("--gpu-tolerance", type=float, default=1e-6)
            sp.add_argument("--controls", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = COMMANDS[args.command](args)
    print(json.dumps(result, indent=2, default=str, ensure_ascii=False)[:20000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
