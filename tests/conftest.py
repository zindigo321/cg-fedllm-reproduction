"""Shared test fixtures. CI never downloads pretrained models: model-level tests use a seeded random
tiny LLaMA on CPU and synthetic token data."""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from cg_fedllm.config import load_config
from cg_fedllm.utils.seeding import configure_determinism

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures"


def pytest_collection_modifyitems(config, items):
    has_cuda = torch.cuda.is_available()
    run_model = os.environ.get("CGFED_RUN_MODEL_TESTS") == "1"
    for item in items:
        if "gpu" in item.keywords and not has_cuda:
            item.add_marker(pytest.mark.skip(reason="requires CUDA"))
        if "model" in item.keywords and not run_model:
            item.add_marker(pytest.mark.skip(reason="set CGFED_RUN_MODEL_TESTS=1 (needs pinned models in the local HF cache)"))


@pytest.fixture(autouse=True)
def _deterministic_cpu():
    configure_determinism(True, 2)
    yield


@pytest.fixture
def tiny_cfg(tmp_path):
    return load_config(REPO / "configs" / "smoke" / "tiny_cpu.yaml", [f"run.output_root={tmp_path.as_posix()}"])


def synthetic_clients(sizes: list[int], vocab: int = 128, seed: int = 0, min_len: int = 6, max_len: int = 18) -> list[list]:
    from cg_fedllm.data.formatting import TokenizedExample

    rng = np.random.default_rng(seed)
    out, sid = [], 0
    for n in sizes:
        client = []
        for _ in range(n):
            length = int(rng.integers(min_len, max_len + 1))
            ids = tuple(int(x) for x in rng.integers(3, vocab, size=length))
            client.append(TokenizedExample(sid, ids, ids))
            sid += 1
        out.append(client)
    return out


@pytest.fixture
def tiny_hf_llama():
    """A seeded random tiny LLaMA built with transformers only (no project model code)."""
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        vocab_size=128, max_position_embeddings=512, tie_word_embeddings=False,
    )
    cfg._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = LlamaForCausalLM(cfg)
    return model.eval()


@pytest.fixture
def tiny_bundle(tiny_cfg):
    from cg_fedllm.pipeline import build_model_bundle

    return build_model_bundle(tiny_cfg)


class ToyTokenizer:
    """Deterministic character-level tokenizer (ids = 3 + Unicode code point mod 997); BOS=1, EOS=2, PAD=0.

    Single characters (e.g. "A") are single tokens; " A" is two tokens -- this lets tests exercise both the
    single-token fast path and the multi-token fallback of the scorer."""

    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0

    def _enc(self, text: str) -> list[int]:
        return [3 + (ord(c) % 997) for c in text]

    def __call__(self, text, add_special_tokens=True, truncation=False, max_length=None, padding=False, return_tensors=None):
        def one(t: str) -> list[int]:
            ids = ([self.bos_token_id] if add_special_tokens else []) + self._enc(t)
            if truncation and max_length is not None:
                ids = ids[:max_length]
            return ids

        if isinstance(text, list):
            return {"input_ids": [one(t) for t in text]}
        return {"input_ids": one(text)}


@pytest.fixture
def toy_tokenizer():
    return ToyTokenizer()


def strip_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s)
