"""Phase-3 calibration helpers: padded-token accounting, the pre-registered micro-batch decision rule, and the
micro-batch loss-normalisation diagnostic (exact emulation vs. the real accumulation path, CPU fp32)."""

from __future__ import annotations

import math

import torch

from cg_fedllm.calibration import (
    GiB,
    micro_batch_decision,
    microbatch_gradient_diagnostic,
    padded_tokens,
    token_length_report,
)
from cg_fedllm.data.formatting import collate
from cg_fedllm.utils.seeding import numpy_rng
from tests.conftest import synthetic_clients


def test_padded_tokens_follow_the_trainer_data_order(tiny_cfg):
    ex = synthetic_clients([9], seed=2, min_len=3, max_len=30)[0]
    cfg = tiny_cfg.local_train
    perm = numpy_rng(5, "fl", 0, 3, "data_order").permutation(len(ex))
    expected_pad = sum(
        collate([ex[int(j)] for j in perm[i : i + 2]], 0, "left", cfg.pad_to_multiple_of)["input_ids"].numel()
        for i in range(0, 9, 2)
    )
    real, pad = padded_tokens(ex, cfg, 5, ("fl", 0, 3), 2)
    assert real == sum(e.num_tokens for e in ex) and pad == expected_pad
    rep = token_length_report({"d2": [ex]}, {"fl": ("d2", [(0, [0]), (1, [0])])}, cutoff=20, cfg=cfg, seed=5)
    assert rep["splits"]["d2"]["n"] == 9 and 0 <= rep["splits"]["d2"]["fraction_at_cutoff"] <= 1
    assert rep["schedules"]["fl"]["examples"] == 18 and rep["schedules"]["fl"]["client_rounds"] == 2


def test_micro_batch_decision_rule():
    cap = int(6.69 * GiB)
    keep = micro_batch_decision([int(5.4 * GiB)], [int(5.9 * GiB), int(5.8 * GiB)], cap)
    assert keep["switch_to_micro_batch_1"] is False and keep["measurements_near_cap"] == 0
    assert (
        micro_batch_decision([int(6.31 * GiB)], [int(6.4 * GiB)], cap)["switch_to_micro_batch_1"] is True
    )  # > 6.3 GiB allocated
    near = cap - 100 * 2**20
    assert (
        micro_batch_decision([int(5 * GiB)] * 2, [near, int(5 * GiB)], cap)["switch_to_micro_batch_1"]
        is False
    )  # once is not "repeatedly"
    assert micro_batch_decision([int(5 * GiB)] * 2, [near, near], cap)["switch_to_micro_batch_1"] is True


def test_microbatch_emulation_is_exact_without_padding(tiny_bundle):
    """Equal lengths that are a multiple of 8: no padding anywhere, so the real accumulation path must equal the
    emulation, and every decomposition gives the same (global token-mean) gradient."""
    from cg_fedllm.data.formatting import TokenizedExample

    b = tiny_bundle
    warm = b.trainer.train(
        b.initial_state, synthetic_clients([8], seed=3)[0], ("warm", 0, 0)
    ).end_state  # B != 0 -> A gets gradients
    g = torch.Generator().manual_seed(9)
    batch = [
        TokenizedExample(i, tuple(int(v) for v in torch.randint(3, 128, (16,), generator=g)), ())
        for i in range(8)
    ]
    batch = [TokenizedExample(e.source_id, e.input_ids, e.input_ids) for e in batch]
    r = microbatch_gradient_diagnostic(
        b.trainer, warm, batch, decompositions=(1, 2, 4, 8), real_micro_batch=2, reference=2
    )
    real = r["real_vs_emulated_micro_batch"]
    assert real["left_padded_examples"] == 0
    assert (
        real["all"]["rel_l2_diff"] < 1e-5
        and real["A"]["rel_l2_diff"] < 1e-5
        and real["B"]["rel_l2_diff"] < 1e-5
    )
    for m in ("1", "4", "8"):
        assert (
            r["vs_token_mean_full_batch"][m]["all"]["rel_l2_diff"] < 1e-12
        )  # equal lengths: no normalisation effect


def test_microbatch_diagnostic_measures_the_normalisation_effect(tiny_bundle):
    b = tiny_bundle
    warm = b.trainer.train(b.initial_state, synthetic_clients([8], seed=3)[0], ("warm", 0, 0)).end_state
    batch = synthetic_clients([8], seed=4, min_len=3, max_len=40)[0]
    drops_before = [m.p for n, m in b.peft_model.named_modules() if n.endswith("lora_dropout.default")]
    r = microbatch_gradient_diagnostic(
        b.trainer, warm, batch, decompositions=(1, 2, 4, 8), real_micro_batch=2, reference=2
    )
    assert [
        m.p for n, m in b.peft_model.named_modules() if n.endswith("lora_dropout.default")
    ] == drops_before  # restored
    assert r["lora_dropout_disabled_for_measurement"] is True
    assert r["vs_micro_batch_2"]["2"]["all"]["rel_l2_diff"] == 0.0
    assert r["vs_token_mean_full_batch"]["8"]["all"]["rel_l2_diff"] == 0.0
    # unequal sequence lengths: per-micro-batch token means weight examples differently from the global token mean
    assert r["vs_token_mean_full_batch"]["2"]["all"]["rel_l2_diff"] > 1e-4
    assert r["per_example_weight_spread"]["8"]["max_over_min"] == max(r["label_tokens_per_example"]) / min(
        r["label_tokens_per_example"]
    )
    assert math.isclose(sum(r["label_tokens_per_example"]), sum(e.num_tokens - 1 for e in batch))
    # left padding adds a first-token label term per padded example in the real path: reported, not hidden
    real = r["real_vs_emulated_micro_batch"]
    assert real["left_padded_examples"] > 0 and real["all"]["rel_l2_diff"] > 0
    assert all(torch.isfinite(torch.tensor(v["all"]["cosine"])) for v in r["vs_micro_batch_2"].values())
