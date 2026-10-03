"""Sampler regression, Shepherd FedAvg regression and all aggregation modes (items 2-4)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from cg_fedllm.federated.aggregation import aggregate, aggregation_weights
from cg_fedllm.federated.sampling import num_selected, shepherd_select_clients
from cg_fedllm.models.adapter import AdapterState


def _audited_shepherd_selection(num_clients: int, frac: float, round_index: int) -> set[int]:
    """Transcription of the audited algorithm (global numpy RNG, as Shepherd does)."""
    np.random.seed(round_index)
    k = max(int(frac * num_clients), 1)
    return {int(x) for x in np.random.choice(np.arange(num_clients), k, replace=False)}


@pytest.mark.parametrize(
    "num_clients,frac", [(100, 0.05), (10, 0.05), (10, 0.1), (3, 1.0), (4, 0.5), (100, 0.2)]
)
def test_sampler_matches_audited_shepherd_algorithm(num_clients, frac):
    for t in range(20):
        ours = shepherd_select_clients(num_clients, frac, t)
        assert set(ours) == _audited_shepherd_selection(num_clients, frac, t)
        assert ours == sorted(ours)  # our fixed processing order
        assert len(ours) == num_selected(num_clients, frac)


def test_selection_count_rule():
    assert num_selected(100, 0.05) == 5
    assert num_selected(10, 0.05) == 1  # int(0.5) == 0 -> max(., 1) -- the appendix's 10-client variant
    assert num_selected(3, 1.0) == 3


def _states(n: int, seed: int = 0) -> list[AdapterState]:
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        out.append(
            AdapterState(
                {
                    "layers.0.self_attn.q_proj.lora_A.weight": torch.randn(4, 16, generator=g),
                    "layers.0.self_attn.q_proj.lora_B.weight": torch.randn(16, 4, generator=g),
                    "layers.1.self_attn.v_proj.lora_A.weight": torch.randn(4, 16, generator=g),
                    "layers.1.self_attn.v_proj.lora_B.weight": torch.randn(16, 4, generator=g),
                }
            )
        )
    return out


def _audited_shepherd_fedavg(states: list[AdapterState], counts: list[int]) -> dict[str, torch.Tensor]:
    """Transcription of Shepherd's FedAvg formula: L1-normalised float32 weights, accumulated in order."""
    weights_array = F.normalize(torch.tensor(counts, dtype=torch.float32), p=1, dim=0)
    acc: dict[str, torch.Tensor] = {}
    for k, st in enumerate(states):
        single = st.tensors
        if k == 0:
            acc = {key: single[key] * (weights_array[k]) for key in single}
        else:
            acc = {key: acc[key] + single[key] * (weights_array[k]) for key in single}
    return acc


def test_sample_weighted_mean_matches_shepherd_formula_bitwise():
    states = _states(5)
    counts = [149, 47, 303, 88, 120]
    ours = aggregate(states, counts, "sample_weighted_mean")
    ref = _audited_shepherd_fedavg(states, counts)
    for k in ours.keys():
        assert torch.equal(ours.tensors[k], ref[k])


def test_weights_and_all_modes():
    counts = [1, 3]
    w = aggregation_weights(counts, "sample_weighted_mean")
    assert torch.allclose(w, torch.tensor([0.25, 0.75])) and abs(float(w.sum()) - 1) < 1e-7
    assert torch.allclose(aggregation_weights(counts, "uniform_mean"), torch.tensor([0.5, 0.5]))
    assert torch.equal(aggregation_weights(counts, "literal_sum"), torch.ones(2))
    s = _states(2, seed=1)
    for key in s[0].keys():
        a, b = s[0].tensors[key], s[1].tensors[key]
        assert torch.allclose(aggregate(s, counts, "sample_weighted_mean").tensors[key], 0.25 * a + 0.75 * b)
        assert torch.allclose(aggregate(s, counts, "uniform_mean").tensors[key], 0.5 * a + 0.5 * b)
        assert torch.allclose(aggregate(s, counts, "literal_sum").tensors[key], a + b)
    with pytest.raises(ValueError):
        aggregate(s, counts, "fedprox")
    with pytest.raises(ValueError):
        aggregation_weights([0, 1], "sample_weighted_mean")


def test_factors_are_aggregated_independently_not_in_product_space():
    s = _states(2, seed=2)
    agg = aggregate(s, [1, 1], "uniform_mean")
    a_key, b_key = "layers.0.self_attn.q_proj.lora_A.weight", "layers.0.self_attn.q_proj.lora_B.weight"
    mean_of_products = 0.5 * (
        s[0].tensors[b_key] @ s[0].tensors[a_key] + s[1].tensors[b_key] @ s[1].tensors[a_key]
    )
    product_of_means = agg.tensors[b_key] @ agg.tensors[a_key]
    assert torch.allclose(agg.tensors[a_key], 0.5 * (s[0].tensors[a_key] + s[1].tensors[a_key]))
    assert not torch.allclose(product_of_means, mean_of_products)  # documents the "LoRA subspace" semantics


def test_literal_sum_is_diagnostic_and_scales_factors():
    s = _states(1, seed=3) * 5
    out = aggregate(s, [10] * 5, "literal_sum")
    for k in out.keys():
        assert torch.allclose(out.tensors[k], 5 * s[0].tensors[k])
