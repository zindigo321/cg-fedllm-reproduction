"""Command-line entry point: ``cgfed <command> --config <yaml> [--set key.path=value ...]``.

Commands: capture-env, prepare-data, run-fl, collect-tgap, train-ae, evaluate, smoke, bench-gpu, and the
Phase-3 diagnostics calibrate-train, microbatch-diag, tgap-stats, ae-viability, ae-select, reference-codes, and the
Phase-4 forensics validate-microbatch, forensic-stats, gradient-forensics, forensic-screen, and the Phase-5 P5-A
capacity control p5a.
Determinism settings and the optional CUDA allocator cap (``run.allocator_cap_margin_mb``) are applied before
any model or CUDA work.
"""

from __future__ import annotations

import argparse
import dataclasses
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


def cmd_validate_microbatch(args) -> dict:
    """F1: physical vs virtual micro-batches on real Dolly batches (PHASE4-DIAGNOSTIC)."""
    from cg_fedllm.federated.sampling import shepherd_select_clients
    from cg_fedllm.microbatch_validation import run_validation
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
    cfg.require("federated", "data", "local_train")
    guard = apply_vram_guard(cfg)
    bundle = build_model_bundle(cfg)
    data = build_data_bundle(cfg)
    d2 = client_examples(cfg, data, bundle.loaded.tokenizer, "d2")
    round0 = shepherd_select_clients(len(d2), cfg.federated.client_fraction, 0)
    cid = max(round0, key=lambda c: (len(d2[c]), -c))
    perm = numpy_rng(cfg.run.seed, "fl", 0, cid, "data_order").permutation(len(d2[cid]))
    bs = cfg.local_train.batch_size
    batches = [[d2[cid][int(j)] for j in perm[k * bs : (k + 1) * bs]] for k in range(len(perm) // bs)]
    if args.adapter:
        start, start_info = AdapterState.load(resolve_path(args.adapter)), Path(args.adapter).name
    else:  # warm start on the client's D1 so that B != 0 and A receives gradients
        d1 = client_examples(cfg, data, bundle.loaded.tokenizer, "d1")
        start = bundle.trainer.train(bundle.initial_state, d1[cid], ("validate_warmup", 0, cid)).end_state
        start_info = f"one local round on client {cid}'s D1 from the initial adapter"
    run_dir = init_run_dir(
        cfg,
        args.stage or "microbatch_validation",
        {"model": bundle.loaded.info, "vram_guard": guard, "start_adapter": start_info},
    )
    t0 = time.time()
    res = run_validation(
        bundle.trainer, start, batches, args.modes.split(","), dropout_batches=args.dropout_batches
    )
    out = {
        "label": cfg.run.result_label,
        "model": {
            k: bundle.loaded.info.get(k)
            for k in ("id", "revision", "dtype", "quantization", "attn_implementation")
        },
        "gradient_checkpointing": cfg.model.gradient_checkpointing,
        "deterministic": cfg.run.deterministic,
        "logical_micro_batch_size": cfg.local_train.micro_batch_size,
        "batch_size": bs,
        "cutoff_len": cfg.data.cutoff_len,
        "batch_rule": f"consecutive batches of {bs} in the trainer's FL round-0 data order of the round-0 client with the most D2 examples",
        "client_id": cid,
        "batches_available": len(batches),
        "start_adapter": {"description": start_info, "sha256": start.sha256()},
        "vram_guard": guard,
        "elapsed_s": None,
        **res,
        "provenance": _provenance(),
    }
    out["elapsed_s"] = round(time.time() - t0, 1)
    atomic_write_json(run_dir / "microbatch_validation.json", out)
    if args.out:
        atomic_write_json(resolve_path(args.out), out)
    return {k: out[k] for k in ("label", "elapsed_s", "reference_mode")} | {
        "acceptance": {
            m: v.get("vs_reference", {}).get("acceptance", {}).get("pass")
            for m, v in res["dropout_off"].items()
        }
    }


def cmd_forensic_stats(args) -> dict:
    """F2-F4 on a TGAP snapshot set: gauge diagnostics, R2 balanced-state and R3 effective-increment statistics."""
    import math
    import statistics

    import torch

    from cg_fedllm.compression.gauge import gauge_diagnostics, module_pairs, spectrum
    from cg_fedllm.compression.metrics import json_safe
    from cg_fedllm.forensics.representations import balanced_effective_delta, balanced_effective_state
    from cg_fedllm.models.adapter import AdapterState
    from cg_fedllm.tgap.snapshots import load_states, read_index
    from cg_fedllm.tgap.train_ae import split_indices
    from cg_fedllm.utils.hashing import sha256_file

    cfg = _load(args)
    cfg.require("lora", "autoencoder")
    s = cfg.lora.alpha / cfg.lora.r
    snap_dir = resolve_path(args.snapshots)
    records = read_index(snap_dir)
    _, val_idx = split_indices(
        records, cfg.autoencoder.split, cfg.autoencoder.val_fraction, cfg.autoencoder.split_seed
    )
    rows = []
    t0 = time.time()
    for i, r in enumerate(records):
        start, end = load_states(snap_dir, r)
        gd = gauge_diagnostics(end, s)
        r2, r2_info = balanced_effective_state(end, s)
        _, r3 = balanced_effective_delta(start, end, s, rank=cfg.lora.r)
        spectra = torch.tensor([m["singular_values"] for m in r3["per_module"]], dtype=torch.float64)
        top = spectra[:, :1].clamp(min=1e-300)
        rows.append(
            {
                "time_index": int(r["time_index"]),
                "client_id": int(r["client_id"]),
                "split": "val" if i in set(val_idx) else "train",
                "raw_factor_sq": gd["factor_sq_total"],
                "A_sq": gd["A_sq_total"],
                "B_sq": gd["B_sq_total"],
                "M_fro_sq": gd["M_fro_sq_total"],
                "M_nuclear": gd["M_nuclear_total"],
                "balanced_factor_sq": gd["balanced_factor_sq_total"],
                "stable_rank": gd["stable_rank"],
                "r2_rms": math.sqrt(r2.l2_sq() / r2.num_elements()),
                "r2_max_abs": max(float(t.abs().max()) for t in r2.tensors.values()),
                "r2_tied": r2_info["tied_singular_values"],
                "r2_zero_components": r2_info["numerically_zero_components"],
                "r3_retained_energy": r3["energy_weighted_retained_energy"],
                "r3_truncation_rel_fro_error": r3["truncation_rel_fro_error"],
                "r3_product_cosine": r3["rank_r_product_cosine"],
                "r3_exact_rank_distribution": r3["exact_rank_distribution"],
                "r3_module_retained": r3["module_retained_energy"],
                "r3_mean_normalised_spectrum": [float(x) for x in (spectra / top).mean(0)],
                "delta_M_energy": sum(sum(x * x for x in m["singular_values"]) for m in r3["per_module"]),
            }
        )
    val_rows = [x for x in rows if x["split"] == "val"]
    med = statistics.median(x["r3_retained_energy"] for x in val_rows)
    # gauge invariance on real data: a random well-conditioned Q per module changes the factor norms, not the spectrum
    start, end = load_states(snap_dir, records[val_idx[0]])
    g = torch.Generator().manual_seed(0)
    gauged, worst = {}, 0.0
    for ka, kb in module_pairs(end):
        q, _ = torch.linalg.qr(torch.randn(cfg.lora.r, cfg.lora.r, generator=g, dtype=torch.float64))
        q = q @ torch.diag(torch.linspace(0.5, 2.0, cfg.lora.r, dtype=torch.float64))
        a, b = end.tensors[ka].double(), end.tensors[kb].double()
        a2, b2 = torch.linalg.solve(q, a), b @ q
        gauged[ka], gauged[kb] = a2.float(), b2.float()
        s1, s2 = spectrum(a, b, s), spectrum(a2, b2, s)
        worst = max(worst, float(((s2 - s1).abs() / s1.max()).max()))
    gauged = AdapterState(gauged)
    out = json_safe(
        {
            "label": cfg.run.result_label,
            "source_mode": records[0]["source_mode"],
            "snapshot_index_sha256": sha256_file(snap_dir / "index.jsonl"),
            "lora_scaling": s,
            "geometry": records[0]["geometry"],
            "split": {
                "val_time_indices": sorted({x["time_index"] for x in val_rows}),
                "val_snapshots": len(val_rows),
            },
            "gauge_invariance_demo": {
                "snapshot": [int(records[val_idx[0]]["time_index"]), int(records[val_idx[0]]["client_id"])],
                "raw_factor_sq_before": end.l2_sq(),
                "raw_factor_sq_after": gauged.l2_sq(),
                "A_sq_before": end.l2_sq("A"),
                "A_sq_after": gauged.l2_sq("A"),
                "B_sq_before": end.l2_sq("B"),
                "B_sq_after": gauged.l2_sq("B"),
                "max_relative_singular_value_change": worst,
            },
            "r3_structural_prerequisite": {
                "rule": "median over the validation snapshots of the energy-weighted rank-8 retained energy must be >= 0.95",
                "median_val_retained_energy": med,
                "structurally_lossy": med < 0.95,
            },
            "summary": {
                split: {
                    k: {
                        "min": min(x[k] for x in rs),
                        "median": statistics.median(x[k] for x in rs),
                        "max": max(x[k] for x in rs),
                    }
                    for k in (
                        "raw_factor_sq",
                        "A_sq",
                        "B_sq",
                        "M_fro_sq",
                        "M_nuclear",
                        "balanced_factor_sq",
                        "r2_rms",
                        "r2_max_abs",
                        "r3_retained_energy",
                        "r3_truncation_rel_fro_error",
                        "r3_product_cosine",
                        "delta_M_energy",
                    )
                }
                for split, rs in (("train", [x for x in rows if x["split"] == "train"]), ("val", val_rows))
            },
            "per_snapshot": rows,
            "elapsed_s": round(time.time() - t0, 1),
            "provenance": _provenance(),
        }
    )
    if args.out:
        atomic_write_json(resolve_path(args.out), out)
    return {
        "r3_structural_prerequisite": out["r3_structural_prerequisite"],
        "gauge_invariance_demo": out["gauge_invariance_demo"],
        "elapsed_s": out["elapsed_s"],
    }


def cmd_gradient_forensics(args) -> dict:
    """F5: bounded real-gradient collection (2 D1 rounds, Phase-3 TGAP schedule/seeds) + statistics."""
    from cg_fedllm.compression.layout import get_layout
    from cg_fedllm.compression.metrics import json_safe
    from cg_fedllm.federated.simulator import FederatedSimulator
    from cg_fedllm.forensics.gradients import GradientDumper, gradient_statistics
    from cg_fedllm.pipeline import (
        apply_vram_guard,
        build_data_bundle,
        build_model_bundle,
        client_examples,
        identity_config_sha256,
        init_run_dir,
        simulator_spec,
    )
    from cg_fedllm.tgap.snapshots import SnapshotWriter, read_index

    cfg = _load(args)
    cfg.require("tgap", "data", "federated", "lora")
    guard = apply_vram_guard(cfg)
    bundle = build_model_bundle(cfg)
    data = build_data_bundle(cfg)
    clients = client_examples(cfg, data, bundle.loaded.tokenizer, cfg.tgap.client_split)
    run_dir = init_run_dir(
        cfg,
        args.stage or "gradient_forensics",
        {"model": bundle.loaded.info, "manifest_sha256": data.manifest_sha256, "vram_guard": guard},
    )
    writer = SnapshotWriter(
        run_dir,
        run_id=f"{cfg.run.name}/{run_dir.name}",
        source_mode="federated_pretrain",
        representation="adapter_state",
        layout=get_layout(cfg.tgap.layout),
    )
    spec = simulator_spec(cfg, len(clients), namespace="tgap_fed")
    spec.num_rounds, spec.client_fraction = cfg.tgap.num_time_steps, cfg.tgap.client_fraction

    def hook(t, cid, start, end, n, rec):
        writer.write(t, cid, start, end, n, {k: v for k, v in rec.items() if k != "payload"})

    t0 = time.time()
    sim = FederatedSimulator(
        bundle.trainer,
        spec,
        clients,
        run_dir / "fl",
        codec=None,
        layout=None,
        identity={"identity_config_sha256": identity_config_sha256(cfg)},
        snapshot_hook=hook,
        observer_factory=lambda t, cid: GradientDumper(run_dir / "gradients", t, cid),
        result_label=cfg.run.result_label,
    )
    fl = sim.run(bundle.initial_state)
    wall = time.time() - t0
    records = read_index(run_dir)
    check = None
    if args.reference_snapshots:
        ref = {
            (int(r["time_index"]), int(r["client_id"])): r["end_adapter_hash"]
            for r in read_index(resolve_path(args.reference_snapshots))
        }
        pairs = [((int(r["time_index"]), int(r["client_id"])), r["end_adapter_hash"]) for r in records]
        check = {
            "compared": len(pairs),
            "bitwise_equal": sum(1 for k, h in pairs if ref.get(k) == h),
            "all_equal": all(ref.get(k) == h for k, h in pairs),
        }
    stats = gradient_statistics(run_dir / "gradients", run_dir, records, cfg.lora.alpha / cfg.lora.r)
    out = json_safe(
        {
            "label": cfg.run.result_label,
            "fl_status": fl["status"],
            "rounds": spec.num_rounds,
            "collection_wall_s": round(wall, 1),
            "unchanged_optimisation_check": check,
            "schedule": [[int(r["time_index"]), int(r["client_id"]), int(r["num_samples"])] for r in records],
            **stats,
            "provenance": _provenance(),
        }
    )
    if args.out:
        atomic_write_json(resolve_path(args.out), out)
    return {
        "label": out["label"],
        "unchanged_optimisation_check": check,
        "optimizer_steps": out["optimizer_steps"],
        "collection_wall_s": out["collection_wall_s"],
    }


def _codec_predictor(codec, ctx_cls):
    def predict(x, it):
        ctx = ctx_cls(it.time_index, it.client_id, 0)
        return codec.decode(codec.encode(x, ctx), ctx)

    return predict


def cmd_forensic_screen(args) -> dict:
    """F6: one pre-registered candidate through the fixed ResNet-3 AE screen."""
    import torch

    from cg_fedllm.compression.autoencoder import load_autoencoder
    from cg_fedllm.compression.codecs import AutoEncoderCodec, CodecContext
    from cg_fedllm.compression.layout import geometry_from_dict, get_layout
    from cg_fedllm.compression.metrics import json_safe
    from cg_fedllm.compression.normalization import (
        MAXABS_QUANTILE,
        MAXABS_TARGET,
        Normalizer,
        train_abs_quantile,
    )
    from cg_fedllm.forensics.gradients import load_client_round, mean_state
    from cg_fedllm.forensics.representations import balanced_effective_delta, balanced_effective_state
    from cg_fedllm.forensics.screen import ScreenItem, evaluate, group_by_time, screen_gate
    from cg_fedllm.pipeline import apply_vram_guard, init_run_dir
    from cg_fedllm.tgap.snapshots import load_states, read_index
    from cg_fedllm.tgap.train_ae import split_indices, train_autoencoder
    from cg_fedllm.utils.hashing import sha256_file

    cfg = _load(args)
    cfg.require("lora", "autoencoder")
    guard = apply_vram_guard(cfg)
    s = cfg.lora.alpha / cfg.lora.r
    kind = args.candidate
    snap_dir = resolve_path(args.snapshots)
    records = read_index(snap_dir)
    layout = get_layout(cfg.autoencoder.layout)
    geom = geometry_from_dict(records[0]["geometry"])
    items = []
    for i, r in enumerate(records):
        start, end = load_states(snap_dir, r)
        if kind == "balanced_effective_state":
            rep, _ = balanced_effective_state(end, s)
        elif kind == "balanced_effective_delta_r8":
            rep, _ = balanced_effective_delta(start, end, s, rank=cfg.lora.r)
        elif kind == "mean_step_gradient":
            cr = load_client_round(
                resolve_path(args.gradients) / f"t{int(r['time_index']):04d}_c{int(r['client_id']):04d}"
            )
            rep = mean_state(cr["grads"])
        else:
            raise ValueError(kind)
        items.append(
            ScreenItem(i, int(r["time_index"]), int(r["client_id"]), int(r["num_samples"]), rep, start, end)
        )
    xs = [layout.forward(it.rep, geom) for it in items]
    train_idx, val_idx = split_indices(
        records, cfg.autoencoder.split, cfg.autoencoder.val_fraction, cfg.autoencoder.split_seed
    )
    q = train_abs_quantile(xs, train_idx, MAXABS_QUANTILE)
    modes = ["none"] + (["global_maxabs_train"] if q["value"] > MAXABS_TARGET else [])
    run_dir = init_run_dir(
        cfg,
        f"{args.stage or 'f6'}_{kind}",
        {"vram_guard": guard, "snapshot_dir": str(snap_dir), "candidate": kind},
    )
    mean = torch.stack([xs[i] for i in train_idx]).mean(0)
    device = torch.device(cfg.run.device)
    result: dict[str, Any] = {
        "label": cfg.run.result_label,
        "candidate": kind,
        "source_mode": records[0]["source_mode"],
        "snapshot_index_sha256": sha256_file(snap_dir / "index.jsonl"),
        "num_snapshots": len(items),
        "split": {
            "train": len(train_idx),
            "val": len(val_idx),
            "val_time_indices": sorted({items[i].time_index for i in val_idx}),
        },
        "scale_rule": {"train_abs_p99_9": q, "threshold": MAXABS_TARGET, "modes_run": modes},
        "input_stats": {
            "train_rms": float(torch.stack([xs[i] for i in train_idx]).pow(2).mean().sqrt()),
            "train_max_abs": float(max(xs[i].abs().max() for i in train_idx)),
        },
        "modes": {},
    }
    by_val = group_by_time([items[i] for i in val_idx])
    by_train = group_by_time([items[i] for i in train_idx])
    for mode in modes:
        ae_cfg = dataclasses.replace(cfg.autoencoder, normalization=mode)
        out_dir = run_dir / f"ae_{mode}"
        t0 = time.time()
        m = train_autoencoder(
            xs,
            [torch.zeros_like(x) for x in xs],
            records,
            ae_cfg,
            device=device,
            out_dir=out_dir,
            provenance={"phase4_candidate": kind, "snapshot_index_sha256": result["snapshot_index_sha256"]},
            label=cfg.run.result_label,
            layout=layout,
            geom=geom,
        )
        codecs = {}
        for name, fname in (
            ("autoencoder_best_val", "autoencoder_best_val.safetensors"),
            ("autoencoder_final", "autoencoder.safetensors"),
        ):
            ae, meta = load_autoencoder(out_dir / fname, device=device)
            codecs[name] = AutoEncoderCodec(
                ae, device=device, normalizer=Normalizer.from_dict(meta.get("normalization"))
            )
        norm = Normalizer.from_dict(m["normalization"])
        preds = {
            **{name: _codec_predictor(codec, CodecContext) for name, codec in codecs.items()},
            "zero": lambda x, it: torch.zeros_like(x),
            "train_mean": lambda x, it: mean.clone(),
            "identity": lambda x, it: x.clone(),
        }
        if not norm.is_identity:
            preds["tanh_range_ceiling"] = lambda x, it, norm=norm: norm.denormalize(
                norm.normalize(x).clamp(-1.0, 1.0)
            )
        val = evaluate(by_val, kind, s, layout, geom, preds)
        train = evaluate(by_train, kind, s, layout, geom, preds)
        result["modes"][mode] = {
            "normalization": m["normalization"],
            "ae_training": {
                k: m[k]
                for k in (
                    "curve",
                    "best_val",
                    "train_time_s",
                    "latent_shape",
                    "input_shape",
                    "compression_ratio_elements",
                    "normalized_range",
                )
                if k in m
            },
            "val": val,
            "train": train,
            "gate": screen_gate(val),
            "elapsed_s": round(time.time() - t0, 1),
        }
        result["modes"][mode]["ae_training"]["best_val"] = {
            k: v for k, v in m.get("best_val", {}).items() if k in ("iteration", "val_mse", "selection")
        }
    result["passes_screen"] = any(v["gate"]["pass"] for v in result["modes"].values())
    result["provenance"] = _provenance()
    result = json_safe(result)
    if args.out:
        atomic_write_json(resolve_path(args.out), result)
    return {
        "candidate": kind,
        "modes": modes,
        "passes_screen": result["passes_screen"],
        "gates": {m: v["gate"]["criteria"] for m, v in result["modes"].items()},
    }


def cmd_p5a(args) -> dict:
    """P5-A v2 training invocation (docs/phase5_preregistration_v2.md). Needs a passing CPU preflight record, a
    reviewed launch record and a separate written run authorisation; the command refuses overrides and any
    configuration other than the protocol's. ``--closure-only`` completes a run whose third invocation died."""
    from cg_fedllm.phase5.p5a import P5AProtocolError
    from cg_fedllm.phase5.p5a_run import production_closure, production_main

    entry = production_closure if args.closure_only else production_main
    out = entry(args.config, args.set, args.runs_root, " ".join(sys.argv), args.launch_record)
    errors = {key: out[key] for key in ("summary_publication_error", "closure_issues") if out.get(key)}
    if errors:
        raise P5AProtocolError(
            "P5-A evidence publication incomplete: " + json.dumps(errors, ensure_ascii=False)
        )
    return {k: out.get(k) for k in ("invocation", "controls", "stop", "closed", "outcomes")}


def cmd_p5a_preflight(args) -> dict:
    """P5-A v2 CPU-only input preflight (section 5.4.1): no AE, no CUDA, no training. Needs a reviewed launch record
    and a separate written authorisation before it is run on the real inputs."""
    from cg_fedllm.phase5.p5a_preflight import production_preflight

    out = production_preflight(args.config, args.set, args.runs_root, " ".join(sys.argv), args.launch_record)
    return {"preflight": out["preflight"], "pass": out["pass"],
            "representations": [{k: r.get(k) for k in ("representation", "pass", "failure")}
                                for r in out["representations"]]}  # fmt: skip


def cmd_baseline_summary(args) -> dict:
    """F7: condense the seed-1 baseline run directories into one labelled record (+ the Identity regression)."""
    import subprocess

    import yaml

    from cg_fedllm.compression.metrics import json_safe
    from cg_fedllm.federated.regression import compare_runs

    runs = {name: resolve_path(p) for name, p in (item.split("=", 1) for item in args.run)}

    def rounds_of(d: Path) -> list[dict]:
        out, t = [], 0
        while (d / "rounds" / f"r{t:04d}" / "DONE").exists():
            out.append(read_json(d / "rounds" / f"r{t:04d}" / "round.json"))
            t += 1
        return out

    summary: dict[str, Any] = {"label": args.label, "runs": {}}
    for name, d in runs.items():
        rs = rounds_of(d)
        sm = read_json(d / "summary.json")
        meta = read_json(d / "run_metadata.json") if (d / "run_metadata.json").exists() else {}
        init = read_json(d / "initial_eval.json") if (d / "initial_eval.json").exists() else None
        resolved = (
            yaml.safe_load((d / "config.resolved.yaml").read_text(encoding="utf-8"))
            if (d / "config.resolved.yaml").exists()
            else {}
        )
        pooled = bool((resolved.get("federated") or {}).get("pooled", False))
        per_round = []
        for r in rs:
            per_round.append(
                {
                    "round": r["round"],
                    "clients": r["selected_clients"],
                    "sample_counts": r["sample_counts"],
                    "client_mean_train_loss": [c.get("mean_loss") for c in r["clients"]],
                    "client_optimizer_steps": [c.get("num_optimizer_steps") for c in r["clients"]],
                    "heldout_loss": (r.get("heldout") or {}).get("loss"),
                    "global_A_sq": r["global_l2_sq"]["A"],
                    "global_B_sq": r["global_l2_sq"]["B"],
                    "effective_global": r.get("effective_global"),
                    "update_norms": r.get("update_norms"),
                    "uplink_logical_bytes": r["uplink_logical_bytes"],
                    "downlink_logical_bytes": r.get("downlink_logical_bytes"),
                    "memory": r.get("memory"),
                    "throughput": r.get("throughput"),
                    "wall_time_s": r["wall_time_s"],
                    "global_hash": r["global_hash"],
                }
            )
        up = sum(x["uplink_logical_bytes"] for x in per_round)
        down = sum((x["downlink_logical_bytes"] or 0) for x in per_round)
        per_client = rs[0]["clients"][0]["payload"]["logical_bytes"] if rs else None
        summary["runs"][name] = {
            "status": sm["status"],
            "rounds_completed": sm["rounds_completed"],
            "final_global_hash": sm["final_global_hash"],
            "initial_heldout_loss": (init or {}).get("heldout", {}).get("loss"),
            "final_heldout_loss": per_round[-1]["heldout_loss"] if per_round else None,
            "git_commit": meta.get("environment", {}).get("git", {}).get("commit"),
            "git_dirty_files": [
                f
                for f in meta.get("environment", {}).get("git", {}).get("dirty_files", [])
                if not f.startswith("?? results/")
            ],
            "git_untracked_results_dirs": [
                f
                for f in meta.get("environment", {}).get("git", {}).get("dirty_files", [])
                if f.startswith("?? results/")
            ],
            "config_sha256": meta.get("config_sha256"),
            "pooled_centralized_reference": pooled,
            "vram_guard": meta.get("vram_guard"),
            "wall_time_s": round(sum(x["wall_time_s"] for x in per_round), 1),
            "communication": {
                "uplink_per_client_bytes": per_client,
                "downlink_per_client_bytes": per_client,
                "uplink_total_bytes": up,
                "downlink_total_bytes": down,
                "two_way_total_bytes": up + down,
                "per_round_two_way_bytes": [
                    x["uplink_logical_bytes"] + (x["downlink_logical_bytes"] or 0) for x in per_round
                ],
                "note": "logical fp32 bytes of the uncompressed adapter state (12,582,912 B per client); no compression",
            }
            if not pooled
            else {
                "applicable": False,
                "note": "centralized reference (one pooled client): no client-server communication; the simulator's logical bytes are not a communication cost",
            },
            "per_round": per_round,
        }
    if args.identity_pair:
        a, b = args.identity_pair.split(",")
        reg = compare_runs(runs[a], runs[b])
        ca, cb = summary["runs"][a]["git_commit"], summary["runs"][b]["git_commit"]
        changed = None
        if ca and cb:
            res = subprocess.run(
                ["git", "diff", "--name-only", ca, cb, "--", "src/"], capture_output=True, text=True
            )
            changed = res.stdout.split() if res.returncode == 0 else None
        reg["provenance"] = {
            "commits": [ca, cb],
            "dirty_files": [summary["runs"][a]["git_dirty_files"], summary["runs"][b]["git_dirty_files"]],
            "src_files_changed_between_commits": changed,
        }
        summary["identity_regression"] = {"pair": [a, b], **{k: v for k, v in reg.items() if k != "pair"}}
    summary["provenance"] = _provenance()
    out = json_safe(summary)
    if args.out:
        atomic_write_json(resolve_path(args.out), out)
    reg = out.get("identity_regression")
    return {
        "identity_regression": None
        if reg is None
        else {k: reg[k] for k in ("pair", "rounds_compared", "checks", "validated", "num_mismatches")},
        "runs": {
            k: {
                kk: v[kk]
                for kk in (
                    "status",
                    "rounds_completed",
                    "initial_heldout_loss",
                    "final_heldout_loss",
                    "wall_time_s",
                )
            }
            for k, v in out["runs"].items()
        },
    }


def cmd_screen_reuse(args) -> dict:
    """F6 R0/R1: read the frozen Phase-3 reports against the S1-S7 screen criteria (DERIVED; nothing recomputed)."""
    from cg_fedllm.compression.metrics import json_safe
    from cg_fedllm.forensics.screen import phase3_gate_mapping
    from cg_fedllm.utils.hashing import sha256_file

    rows = {}
    for item in args.phase3:
        name, path = item.split("=", 1)
        f = resolve_path(path)
        rows[name] = {
            "source": Path(path).as_posix(),
            "source_sha256": sha256_file(f),
            **phase3_gate_mapping(read_json(f)),
        }
    out = json_safe(
        {
            "label": "DERIVED",
            "note": "Phase-3 best-val AE metrics (validation, rounds 16-19) mapped onto the Phase-4 S1-S7 criteria; R0/R1 were not retrained (reviewer R2)",
            "candidates": rows,
            "provenance": _provenance(),
        }
    )
    if args.out:
        atomic_write_json(resolve_path(args.out), out)
    return {k: {"pass": v["pass"], "criteria": v["criteria"]} for k, v in out["candidates"].items()}


def cmd_eval_cost(args) -> dict:
    """F8 planning (CPU, tokenizer only): the forward passes each benchmark of an eval config will need."""
    from transformers import AutoTokenizer

    from cg_fedllm.evaluation.reference_eval import prepare_requests
    from cg_fedllm.evaluation.scorer import batch_plan
    from cg_fedllm.models.loading import snapshot_model

    cfg = _load(args)
    cfg.require("model", "eval")
    tok = AutoTokenizer.from_pretrained(snapshot_model(cfg.model))
    out: dict[str, Any] = {
        "label": cfg.run.result_label,
        "model": {"id": cfg.model.id, "revision": cfg.model.revision},
        "benchmarks": {},
    }
    for spec in cfg.eval.benchmarks:
        requests, _, rev, _ = prepare_requests(tok, spec, cfg.eval)
        out["benchmarks"][f"{spec.name}/{spec.split}"] = {
            "spec": dataclasses.asdict(spec),
            "dataset_revision": rev,
            **batch_plan(
                requests, cfg.eval.max_batch_tokens, cfg.eval.max_batch_size, cfg.eval.max_batch_attention
            ),
        }
    if args.out:
        atomic_write_json(resolve_path(args.out), out)
    return out


def cmd_eval_projection(args) -> dict:
    """F8 decision (DERIVED): project each model's full-evaluation time from its timed sample and the cost plans."""
    full = read_json(resolve_path(args.cost_full))["benchmarks"]
    sample = read_json(resolve_path(args.cost_sample))["benchmarks"]
    models: dict[str, Any] = {}
    for item in args.timing_run:
        name, path = item.split("=", 1)
        timed = read_json(resolve_path(path) / "eval_summary.json")
        rows, total = {}, 0.0
        for bench, t in timed.items():
            fs, ss = full[bench], sample[bench]
            score = t["timing_s"]["score"] * fs["padded_tokens"] / ss["padded_tokens"]
            prepare = (
                t["timing_s"]["prepare"] * fs["questions"] / ss["questions"]
            )  # upper bound: dataset loading is a fixed cost
            rows[bench] = {
                "sample_questions": ss["questions"],
                "sample_padded_tokens": ss["padded_tokens"],
                "sample_timing_s": t["timing_s"],
                "sample_padded_tokens_per_s": ss["padded_tokens"] / t["timing_s"]["score"],
                "full_questions": fs["questions"],
                "full_padded_tokens": fs["padded_tokens"],
                "projected_full_score_s": score,
                "projected_full_prepare_s_upper": prepare,
                "projected_full_s": score + prepare,
            }
            total += score + prepare
        models[name] = {
            "timing_run": Path(path).as_posix(),
            "benchmarks": rows,
            "projected_full_minutes": total / 60,
        }
    worst = max(m["projected_full_minutes"] for m in models.values())
    out = {
        "label": "DERIVED",
        "rule": f"full MMLU test + C-Eval val per model if the projected time is <= {args.threshold_min} min for every model; otherwise a fixed seeded stratified subset (docs/phase4_preregistration.md, section 5)",
        "projection": "full_score = sample_score_s * full_padded_tokens / sample_padded_tokens; prepare scaled by question count (upper bound)",
        "threshold_minutes": args.threshold_min,
        "models": models,
        "max_projected_full_minutes": worst,
        "decision": "full" if worst <= args.threshold_min else "subset",
        "provenance": _provenance(),
    }
    if args.out:
        atomic_write_json(resolve_path(args.out), out)
    return {
        "decision": out["decision"],
        "projected_full_minutes": {k: round(v["projected_full_minutes"], 1) for k, v in models.items()},
    }


def cmd_gradient_shape(args) -> dict:
    """F5 addendum (PHASE4-FORENSIC, CPU): value-distribution shape of the recorded gradient/update families."""
    from cg_fedllm.compression.metrics import json_safe
    from cg_fedllm.forensics.gradients import family_shapes
    from cg_fedllm.tgap.snapshots import read_index
    from cg_fedllm.utils.hashing import sha256_file

    root = resolve_path(args.forensics)
    out = json_safe(
        {
            "label": "PHASE4-FORENSIC",
            "source": "F5 gradient dumps (outside Git)",
            "snapshot_index_sha256": sha256_file(root / "index.jsonl"),
            "families": family_shapes(root / "gradients", root, read_index(root)),
            "provenance": _provenance(),
        }
    )
    if args.out:
        atomic_write_json(resolve_path(args.out), out)
    return out["families"]


def cmd_eval_compare(args) -> dict:
    """F8 (DERIVED): paired per-question comparison of two `evaluate` runs on identical questions."""
    from cg_fedllm.evaluation.aggregate import paired_comparison

    (na, da), (nb, db) = ((n, resolve_path(p)) for n, p in (x.split("=", 1) for x in (args.a, args.b)))

    def correct(d: Path, stem: str) -> dict[str, bool]:
        rows = (
            json.loads(line)
            for line in (d / f"{stem}_predictions.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        return {r["qid"]: r["pred"] == r["answer"] for r in rows}

    stems = sorted(p.name[: -len("_predictions.jsonl")] for p in da.glob("*_predictions.jsonl"))
    out = {
        "label": "DERIVED",
        "pair": [na, nb],
        "runs": {na: Path(args.a.split("=", 1)[1]).as_posix(), nb: Path(args.b.split("=", 1)[1]).as_posix()},
        "note": "question-level paired comparison (question-weighted); the C-Eval headline is a subject macro average",
        "benchmarks": {stem: paired_comparison(correct(da, stem), correct(db, stem)) for stem in stems},
        "provenance": _provenance(),
    }
    if args.out:
        atomic_write_json(resolve_path(args.out), out)
    return out["benchmarks"]


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
    "validate-microbatch": cmd_validate_microbatch,
    "forensic-stats": cmd_forensic_stats,
    "gradient-forensics": cmd_gradient_forensics,
    "forensic-screen": cmd_forensic_screen,
    "baseline-summary": cmd_baseline_summary,
    "screen-reuse": cmd_screen_reuse,
    "eval-cost": cmd_eval_cost,
    "eval-projection": cmd_eval_projection,
    "gradient-shape": cmd_gradient_shape,
    "eval-compare": cmd_eval_compare,
    "p5a-preflight": cmd_p5a_preflight,
    "p5a": cmd_p5a,
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cgfed", description="CG-FedLLM reproduction toolkit")
    sub = p.add_subparsers(dest="command", required=True)
    for name in COMMANDS:
        sp = sub.add_parser(name)
        if name not in (
            "capture-env",
            "ae-select",
            "baseline-summary",
            "screen-reuse",
            "eval-projection",
            "gradient-shape",
            "eval-compare",
        ):
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
        if name in ("microbatch-diag", "validate-microbatch"):
            sp.add_argument(
                "--adapter",
                default=None,
                help="start adapter (default: the initial LoRA state / a warm start)",
            )
        if name in ("forensic-stats", "forensic-screen"):
            sp.add_argument("--snapshots", required=True)
        if name in ("forensic-stats", "gradient-forensics", "forensic-screen"):
            sp.add_argument("--out", default=None)
        if name == "gradient-forensics":
            sp.add_argument(
                "--reference-snapshots", default=None, help="Phase-3 TGAP snapshot dir for the bitwise check"
            )
        if name == "forensic-screen":
            sp.add_argument(
                "--candidate",
                required=True,
                choices=["balanced_effective_state", "balanced_effective_delta_r8", "mean_step_gradient"],
            )
            sp.add_argument("--gradients", default=None, help="F5 gradient directory (mean_step_gradient)")
        if name == "baseline-summary":
            sp.add_argument("--run", action="append", required=True, help="name=run_dir")
            sp.add_argument("--identity-pair", default=None, help="lora_ft,faf_identity")
            sp.add_argument("--label", required=True, choices=RESULT_LABELS)
            sp.add_argument("--out", default=None)
        if name == "eval-cost":
            sp.add_argument("--out", default=None)
        if name == "eval-compare":
            sp.add_argument("--a", required=True, help="name=evaluate run dir")
            sp.add_argument("--b", required=True, help="name=evaluate run dir")
            sp.add_argument("--out", default=None)
        if name == "gradient-shape":
            sp.add_argument("--forensics", required=True, help="F5 run directory (snapshots + gradients/)")
            sp.add_argument("--out", default=None)
        if name == "eval-projection":
            sp.add_argument(
                "--timing-run",
                action="append",
                required=True,
                help="model=evaluate run dir of the timed sample",
            )
            sp.add_argument("--cost-full", required=True)
            sp.add_argument("--cost-sample", required=True)
            sp.add_argument("--threshold-min", type=float, default=45.0)
            sp.add_argument("--out", default=None)
        if name == "screen-reuse":
            sp.add_argument("--phase3", action="append", required=True, help="id=path/to/phase3 a5/a8 report")
            sp.add_argument("--out", default=None)
        if name == "validate-microbatch":
            sp.add_argument(
                "--modes",
                required=True,
                help="comma list, first = reference, e.g. physical:16,virtual:1,virtual:2",
            )
            sp.add_argument("--dropout-batches", type=int, default=4)
            sp.add_argument("--out", default=None)
        if name == "ae-select":
            sp.add_argument("--gate", action="append", required=True, help="mode=path/to/viability.json")
            sp.add_argument("--label", required=True, choices=RESULT_LABELS)
        if name == "evaluate":
            sp.add_argument("--adapter", default=None)
            sp.add_argument("--tag", default=None)
        if name in ("p5a", "p5a-preflight"):
            sp.add_argument(
                "--runs-root", default=None, help="CGFED_RUNS (frozen inputs and the P5-A run root)"
            )
            sp.add_argument(
                "--launch-record",
                default=None,
                help="path of the reviewed launch record, cited by path and SHA-256 (not an authorisation check)",
            )
        if name == "p5a":
            sp.add_argument(
                "--closure-only",
                action="store_true",
                help="complete a run whose third invocation died before its summary (section 13.5)",
            )
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
