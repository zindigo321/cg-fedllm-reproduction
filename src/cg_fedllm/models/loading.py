"""Model/tokenizer loading with explicit revisions, dtypes and file allow-lists.

Pretrained checkpoints are fetched with ``snapshot_download(allow_patterns=...)`` from a pinned
40-character commit SHA, so only the files listed in the config (e.g. ``model.safetensors`` but not a
repository's ``optimizer.pt`` or duplicate ``pytorch_model.bin``) are downloaded into the configured HF
cache. ``kind='tiny_llama'`` builds a seeded random LLaMA for CPU tests (nothing is downloaded).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from cg_fedllm.config import ModelSection

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


@dataclass
class LoadedModel:
    model: Any
    tokenizer: Any | None
    local_path: Path | None
    resolved_revision: str | None
    pad_token_id: int
    padding_side: str
    info: dict[str, Any]


def snapshot_model(cfg: ModelSection) -> Path:
    from huggingface_hub import snapshot_download

    if not cfg.allow_patterns:
        raise ValueError("model.allow_patterns must list the files to download (no implicit full-repo downloads)")
    local = snapshot_download(
        repo_id=cfg.id, revision=cfg.revision, allow_patterns=list(cfg.allow_patterns), local_files_only=cfg.local_files_only
    )
    path = Path(local)
    if path.name != cfg.revision:
        raise RuntimeError(f"snapshot resolved to {path.name}, expected pinned revision {cfg.revision}")
    return path


def build_tiny_llama(cfg: ModelSection):
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        hidden_size=cfg.tiny_hidden_size,
        intermediate_size=cfg.tiny_intermediate_size,
        num_hidden_layers=cfg.tiny_num_layers,
        num_attention_heads=cfg.tiny_num_heads,
        num_key_value_heads=cfg.tiny_num_heads,
        vocab_size=cfg.tiny_vocab_size,
        max_position_embeddings=cfg.tiny_max_position_embeddings,
        tie_word_embeddings=False,
    )
    config._attn_implementation = cfg.attn_implementation
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cfg.tiny_init_seed)
        model = LlamaForCausalLM(config)
    return model.to(DTYPES[cfg.dtype])


def load_model(cfg: ModelSection, device: str | torch.device, with_tokenizer: bool = True) -> LoadedModel:
    """Load the frozen base model (eval mode, ``requires_grad=False``) and, optionally, its tokenizer."""
    if cfg.kind == "tiny_llama":
        model = build_tiny_llama(cfg).to(device)
        for p in model.parameters():
            p.requires_grad_(False)
        pad = 0 if cfg.pad_token_id is None else cfg.pad_token_id
        return LoadedModel(model, None, None, None, pad, cfg.padding_side, {"kind": "tiny_llama"})

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    path = snapshot_model(cfg)
    kwargs: dict[str, Any] = {"dtype": DTYPES[cfg.dtype], "attn_implementation": cfg.attn_implementation}
    if cfg.quantization == "nf4":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=DTYPES[cfg.dtype]
        )
        kwargs["device_map"] = {"": device}
    elif cfg.quantization == "int8":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        kwargs["device_map"] = {"": device}
    model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    if cfg.quantization == "none":
        model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tokenizer = None
    pad = cfg.pad_token_id
    if with_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(path)
        if pad is None:
            pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad is None:
        raise ValueError("could not determine a pad token id; set model.pad_token_id explicitly")
    info = {
        "kind": "hf",
        "id": cfg.id,
        "revision": cfg.revision,
        "license": cfg.license,
        "local_path": str(path),
        "files": sorted(p.name for p in path.iterdir()),
        "dtype": cfg.dtype,
        "quantization": cfg.quantization,
        "attn_implementation": cfg.attn_implementation,
        "num_parameters": int(sum(p.numel() for p in model.parameters())),
    }
    return LoadedModel(model, tokenizer, path, path.name, int(pad), cfg.padding_side, info)
