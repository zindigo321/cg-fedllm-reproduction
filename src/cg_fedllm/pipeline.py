"""Wiring from a validated config to runnable components, with provenance written for every run."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from cg_fedllm.compression.codecs import (
    AutoEncoderCodec,
    Codec,
    ConstantMeanCodec,
    GaussianNoiseCodec,
    IdentityCodec,
)
from cg_fedllm.compression.layout import get_layout
from cg_fedllm.config import ExperimentConfig, dump_config_yaml, resolve_path
from cg_fedllm.data.dolly import DollyRecord, load_source, source_cache_path
from cg_fedllm.data.formatting import TokenizedExample, tokenize_record
from cg_fedllm.data.manifest import PartitionManifest, load_manifest
from cg_fedllm.data.manifest import prepare_manifest as _prepare_manifest
from cg_fedllm.evaluation.heldout import heldout_loss
from cg_fedllm.federated.client import LocalTrainer
from cg_fedllm.federated.simulator import SimulatorSpec
from cg_fedllm.models.adapter import AdapterState
from cg_fedllm.models.loading import LoadedModel, load_model
from cg_fedllm.models.lora import attach_lora, get_adapter_state, lora_parameters, set_adapter_state
from cg_fedllm.utils.hashing import canonical_json_sha256, sha256_file
from cg_fedllm.utils.io import atomic_write_json, atomic_write_text, load_tensors
from cg_fedllm.utils.provenance import collect_environment

PRECISION_DTYPE = {"fp32": "float32", "bf16": "bfloat16"}


# --------------------------------------------------------------------------------------------------
# Run directories and provenance
# --------------------------------------------------------------------------------------------------


def identity_config_sha256(cfg: ExperimentConfig) -> str:
    """Config hash that ignores the interruption knob ``federated.stop_after_round`` (used for resume)."""
    d = copy.deepcopy(cfg.to_dict())
    if d.get("federated"):
        d["federated"]["stop_after_round"] = None
    return canonical_json_sha256(d)


def init_run_dir(cfg: ExperimentConfig, stage: str, extra_meta: dict[str, Any] | None = None) -> Path:
    run_dir = resolve_path(cfg.run.output_root) / cfg.run.name / stage
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(run_dir / "config.resolved.yaml", dump_config_yaml(cfg))
    atomic_write_json(
        run_dir / "config.sha256.json",
        {"config_sha256": cfg.sha256(), "identity_config_sha256": identity_config_sha256(cfg)},
    )
    meta = {
        "stage": stage,
        "result_label": cfg.run.result_label,
        "config_sha256": cfg.sha256(),
        "environment": collect_environment(),
        "seeds": {
            "run.seed": cfg.run.seed,
            "lora.init_seed": cfg.lora.init_seed if cfg.lora else None,
            "data.partition_seed": cfg.data.partition_seed if cfg.data else None,
            "data.d1d2_split_seed": cfg.data.d1d2_split_seed if cfg.data else None,
            "autoencoder.init_seed": cfg.autoencoder.init_seed if cfg.autoencoder else None,
        },
        "modes": {
            "representation": cfg.federated.representation if cfg.federated else None,
            "aggregation": cfg.federated.aggregation if cfg.federated else None,
            "codec": cfg.codec.type if cfg.codec else None,
            "layout": cfg.codec.layout if cfg.codec else (cfg.tgap.layout if cfg.tgap else None),
            "tgap_source": cfg.tgap.source if cfg.tgap else None,
        },
        **(extra_meta or {}),
    }
    atomic_write_json(run_dir / "run_metadata.json", meta)
    return run_dir


def update_run_metadata(run_dir: Path, **fields: Any) -> None:
    from cg_fedllm.utils.io import read_json

    path = run_dir / "run_metadata.json"
    meta = read_json(path) if path.exists() else {}
    meta.update(fields)
    atomic_write_json(path, meta)


def apply_vram_guard(cfg: ExperimentConfig) -> dict[str, Any] | None:
    """Cap the CUDA allocator before anything is loaded when ``run.allocator_cap_margin_mb`` is set."""
    if cfg.run.device != "cuda" or cfg.run.allocator_cap_margin_mb is None:
        return None
    from cg_fedllm.utils.gpu import cap_allocator_to_free_vram

    return cap_allocator_to_free_vram(margin_bytes=int(cfg.run.allocator_cap_margin_mb) * 2**20)


# --------------------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------------------


@dataclass
class ModelBundle:
    loaded: LoadedModel
    peft_model: Any
    params: dict[str, torch.nn.Parameter]
    trainer: LocalTrainer
    initial_state: AdapterState
    device: torch.device


def build_model_bundle(cfg: ExperimentConfig) -> ModelBundle:
    cfg.require("model", "lora", "local_train")
    if cfg.model.quantization == "none" and PRECISION_DTYPE[cfg.local_train.precision] != cfg.model.dtype:
        raise ValueError(
            f"local_train.precision={cfg.local_train.precision} is inconsistent with model.dtype={cfg.model.dtype}"
        )
    device = torch.device(cfg.run.device)
    loaded = load_model(cfg.model, device, with_tokenizer=cfg.model.kind == "hf")
    if cfg.model.gradient_checkpointing:
        loaded.model.config.use_cache = False
        loaded.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    peft_model = attach_lora(loaded.model, cfg.lora)
    params = lora_parameters(peft_model)
    initial = get_adapter_state(params)
    trainer = LocalTrainer(
        peft_model,
        params,
        cfg.local_train,
        pad_token_id=loaded.pad_token_id,
        padding_side=loaded.padding_side,
        device=device,
        base_seed=cfg.run.seed,
    )
    return ModelBundle(loaded, peft_model, params, trainer, initial, device)


def make_heldout_fn(cfg: ExperimentConfig, bundle: ModelBundle, heldout: list[TokenizedExample]):
    if not heldout:
        return None

    def fn(state: AdapterState) -> dict:
        set_adapter_state(bundle.params, state)
        return heldout_loss(
            bundle.peft_model,
            heldout,
            micro_batch_size=cfg.local_train.micro_batch_size,
            pad_token_id=bundle.loaded.pad_token_id,
            padding_side=bundle.loaded.padding_side,
            device=bundle.device,
            pad_to_multiple_of=cfg.local_train.pad_to_multiple_of,
        )

    return fn


# --------------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------------


@dataclass
class DataBundle:
    manifest: PartitionManifest
    manifest_sha256: str
    records: dict[int, DollyRecord]
    source_file_sha256: str


def prepare_manifest(cfg: ExperimentConfig, write: bool = False) -> tuple[PartitionManifest, dict[str, Any]]:
    """Config-level wrapper of :func:`cg_fedllm.data.manifest.prepare_manifest`."""
    cfg.require("data")
    return _prepare_manifest(cfg.data, write=write)


def build_data_bundle(cfg: ExperimentConfig) -> DataBundle:
    cfg.require("data")
    path = resolve_path(cfg.data.manifest_path)
    manifest = load_manifest(path)
    if manifest.data["source"]["sha256"] != cfg.data.source_sha256:
        raise RuntimeError("manifest source hash does not match the config's pinned source")
    records = {r.source_id: r for r in load_source(cfg.data)}
    return DataBundle(manifest, sha256_file(path), records, sha256_file(source_cache_path(cfg.data)))


def client_examples(
    cfg: ExperimentConfig, data: DataBundle, tokenizer, split: str
) -> list[list[TokenizedExample]]:
    out = []
    for cid in range(data.manifest.num_clients):
        out.append(
            [
                tokenize_record(tokenizer, data.records[i], cfg.data)
                for i in data.manifest.split_ids(cid, split)
            ]
        )
    return out


def heldout_examples(cfg: ExperimentConfig, data: DataBundle, tokenizer) -> list[TokenizedExample]:
    ids = data.manifest.holdout_ids
    if cfg.data.heldout_max_examples is not None:
        ids = ids[: cfg.data.heldout_max_examples]
    return [tokenize_record(tokenizer, data.records[i], cfg.data) for i in ids]


# --------------------------------------------------------------------------------------------------
# Simulator / codecs
# --------------------------------------------------------------------------------------------------


def simulator_spec(cfg: ExperimentConfig, num_clients: int, namespace: str = "fl") -> SimulatorSpec:
    f = cfg.federated
    return SimulatorSpec(
        num_clients=num_clients,
        client_fraction=f.client_fraction,
        num_rounds=f.num_rounds,
        aggregation=f.aggregation,
        representation=f.representation,
        seed=cfg.run.seed,
        namespace=namespace,
        heldout_eval_every=f.heldout_eval_every,
        stop_after_round=f.stop_after_round,
    )


def build_codec(cfg: ExperimentConfig, device: torch.device) -> Codec | None:
    c = cfg.codec
    if c is None or c.type == "none":
        return None
    if c.type == "identity":
        return IdentityCodec()
    if c.type == "autoencoder":
        from cg_fedllm.compression.autoencoder import load_autoencoder
        from cg_fedllm.compression.normalization import Normalizer

        ckpt = resolve_path(c.ae_checkpoint)
        ae, meta = load_autoencoder(ckpt, device=c.device or device)
        # the frozen TGAP-fitted normaliser travels with the checkpoint (Phase-2 checkpoints carry none)
        norm = Normalizer.from_dict(meta.get("normalization"))
        info = {"checkpoint": str(ckpt), "checkpoint_sha256": sha256_file(ckpt), "ae_metadata": meta}
        return AutoEncoderCodec(
            ae, device=c.device or device, latent_dtype=c.latent_dtype, info=info, normalizer=norm
        )
    if c.type == "constant_mean":
        path = resolve_path(c.mean_path)
        return ConstantMeanCodec(
            load_tensors(path)["x"], {"mean_path": str(path), "mean_sha256": sha256_file(path)}
        )
    if c.type == "gaussian_noise":
        return GaussianNoiseCodec(float(c.noise_sigma))
    raise ValueError(f"unknown codec {c.type!r}")


def layout_for(cfg: ExperimentConfig):
    return get_layout(cfg.codec.layout if cfg.codec else "layer_major_qkvo_AtB")
