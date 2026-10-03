"""Phase-4 F1: ``virtual_paper_microbatch`` reproduces a true physical micro-batch of 16 (tiny LLaMA, CPU fp32)."""

from __future__ import annotations

import pytest

from cg_fedllm.config import ConfigError, LocalTrainSection
from cg_fedllm.microbatch_validation import accept, compare, dropout_set, one_step, trainer_with
from cg_fedllm.pipeline import build_model_bundle
from tests.conftest import synthetic_clients


@pytest.fixture
def mb16_bundle(tiny_cfg):
    cfg = tiny_cfg
    cfg.local_train.batch_size, cfg.local_train.micro_batch_size = 32, 16
    return build_model_bundle(cfg)


def _warm(bundle):
    """A start adapter with B != 0, so A receives gradients too."""
    return bundle.trainer.train(
        bundle.initial_state, synthetic_clients([16], seed=21)[0], ("warm", 0, 0)
    ).end_state


def test_virtual_microbatch_config_is_validated():
    with pytest.raises(ConfigError):
        LocalTrainSection(
            batch_size=32, micro_batch_size=16, microbatch_mode="virtual_paper_microbatch"
        ).validate("lt")
    with pytest.raises(ConfigError):
        LocalTrainSection(
            batch_size=32,
            micro_batch_size=16,
            microbatch_mode="virtual_paper_microbatch",
            physical_chunk_size=17,
        ).validate("lt")
    with pytest.raises(ConfigError):
        LocalTrainSection(batch_size=32, micro_batch_size=16, physical_chunk_size=2).validate("lt")
    LocalTrainSection(
        batch_size=32, micro_batch_size=16, microbatch_mode="virtual_paper_microbatch", physical_chunk_size=2
    ).validate("lt")


def test_virtual_matches_physical_with_dropout_off(mb16_bundle):
    b = mb16_bundle
    start = _warm(b)
    batch = synthetic_clients([32], seed=22, min_len=3, max_len=60)[0]  # unequal lengths: real padding
    with dropout_set(b.peft_model, 0.0):
        phys = one_step(b.trainer, start, batch, ("v", 0, 0))
        for chunk in (1, 2, 3, 16):
            virt = one_step(
                trainer_with(
                    b.trainer, microbatch_mode="virtual_paper_microbatch", physical_chunk_size=chunk
                ),
                start,
                batch,
                ("v", 0, 0),
            )
            c = compare(virt, phys)
            verdict = accept(c)
            assert verdict["pass"], (chunk, c, verdict)
            assert c["update_rel_l2"] < 1e-4, (chunk, c["update_rel_l2"])  # even the step itself agrees
            assert virt["result"].num_forward_chunks == 2 * -(-16 // chunk)
            assert (
                virt["result"].num_padded_tokens == phys["result"].num_padded_tokens
            )  # identical logical padding


def test_virtual_matches_physical_over_a_full_round_with_a_partial_group(mb16_bundle):
    b = mb16_bundle
    start = _warm(b)
    data = synthetic_clients([40], seed=23, min_len=3, max_len=50)[
        0
    ]  # 40 = 32 + 8: last group = one logical micro-batch of 8
    with dropout_set(b.peft_model, 0.0):
        phys = b.trainer.train(start, data, ("round", 1, 2))
        virt = trainer_with(
            b.trainer, microbatch_mode="virtual_paper_microbatch", physical_chunk_size=2
        ).train(start, data, ("round", 1, 2))
    assert phys.num_optimizer_steps == virt.num_optimizer_steps == 2
    assert all(abs(a - c) <= 1e-6 * abs(c) for a, c in zip(virt.step_losses, phys.step_losses))
    assert virt.end_state.relative_l2_diff(phys.end_state) < 1e-5


def test_dropout_on_is_reproducible_without_loss_scaling_error(mb16_bundle):
    b = mb16_bundle
    start = _warm(b)
    virt_tr = trainer_with(b.trainer, microbatch_mode="virtual_paper_microbatch", physical_chunk_size=2)
    ratios = []
    for seed in range(4):
        batch = synthetic_clients([32], seed=30 + seed, min_len=3, max_len=60)[0]
        p1, p2 = (
            one_step(b.trainer, start, batch, ("d", seed, 0)),
            one_step(b.trainer, start, batch, ("d", seed, 0)),
        )
        v1, v2 = (
            one_step(virt_tr, start, batch, ("d", seed, 0)),
            one_step(virt_tr, start, batch, ("d", seed, 0)),
        )
        assert p1["end"].equal(p2["end"]) and v1["end"].equal(v2["end"])  # each mode is reproducible
        ratios.append(v1["loss"] / p1["loss"])
    mean = sum(ratios) / len(ratios)
    assert abs(mean - 1.0) < 0.02  # different dropout masks, no systematic loss scaling
