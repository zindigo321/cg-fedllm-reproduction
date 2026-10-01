"""GPU micro-benchmark harness on the seeded tiny LLaMA (nothing downloaded)."""

from __future__ import annotations

import pytest
import torch

from cg_fedllm.config import LoRASection, ModelSection

pytestmark = pytest.mark.gpu  # skipped without CUDA (tests/conftest.py)


def test_bench_config_records_memory_throughput_and_oom():
    from cg_fedllm.bench import bench_config

    mcfg = ModelSection(kind="tiny_llama", dtype="float32", attn_implementation="eager")
    lora = LoRASection(r=8, alpha=16, dropout=0.0)
    rows = bench_config(mcfg, lora, seq_len=64, micro_batches=[1, 4], gradient_checkpointing=False, steps=2, warmup=1)
    assert [r["micro_batch"] for r in rows] == [1, 4]
    for r in rows:
        assert r["label"] == "LOCAL-MICROBENCH"
        assert r["status"] == "ok"
        assert r["peak_allocated_bytes"] >= r["weights_allocated_bytes"] > 0
        assert r["peak_reserved_bytes"] >= r["peak_allocated_bytes"]
        assert r["tokens_per_s"] > 0
    # an allocation beyond the allocator cap must surface as a recorded OOM, not a silent spill
    torch.cuda.set_per_process_memory_fraction(0.01)
    try:
        oom = bench_config(mcfg, lora, seq_len=256, micro_batches=[512], gradient_checkpointing=False, steps=1, warmup=0)
    finally:
        torch.cuda.set_per_process_memory_fraction(1.0)
    assert oom[0]["status"] == "OOM"
