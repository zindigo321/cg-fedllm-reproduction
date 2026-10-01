"""Server aggregation strategies (reviewer decision R3). A and B tensors are aggregated independently.

* ``sample_weighted_mean`` (PRIMARY, Shepherd FedAvg): w_k = n_k / sum_j n_j, computed exactly like
  Shepherd's ``normalize(torch.tensor(counts, float32), p=1, dim=0)``; result = sum_k w_k * state_k.
* ``uniform_mean``: w_k = 1 / K (same normalisation applied to a vector of ones).
* ``literal_sum``: sum_k state_k -- the paper's Algorithm-1 notation taken literally. DIAGNOSTIC ONLY:
  with K clients it scales both factors by K (so B A by K^2); never a default.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from cg_fedllm.models.adapter import AdapterState, l1_normalised, weighted_sum

AGGREGATIONS = ("sample_weighted_mean", "uniform_mean", "literal_sum")


def aggregation_weights(sample_counts: Sequence[int], strategy: str) -> torch.Tensor:
    if strategy == "sample_weighted_mean":
        if any(n <= 0 for n in sample_counts):
            raise ValueError("sample counts must be positive")
        return l1_normalised([float(n) for n in sample_counts])
    if strategy == "uniform_mean":
        return l1_normalised([1.0] * len(sample_counts))
    if strategy == "literal_sum":
        return torch.ones(len(sample_counts), dtype=torch.float32)
    raise ValueError(f"unknown aggregation {strategy!r}")


def aggregate(states: Sequence[AdapterState], sample_counts: Sequence[int], strategy: str) -> AdapterState:
    if len(states) != len(sample_counts):
        raise ValueError("states and sample_counts must have equal length")
    return weighted_sum(states, aggregation_weights(sample_counts, strategy))
