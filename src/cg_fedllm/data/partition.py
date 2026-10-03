"""Shepherd-compatible client partitioning and the per-client D1/D2 split.

Provenance
----------
BEHAVIOR RECONSTRUCTED FROM SHEPHERD. This is an independent reimplementation of the *semantics* of
``client_data_allocation.py`` in JayZhang42/FederatedGPT-Shepherd @ bcffa00e9642990ecc6210363a7f0dab91bef4dc
(Apache-2.0); no Shepherd source is copied or executed at runtime. Equivalence is established by a
regression test against oracle source-ID assignments produced by running the pinned Shepherd script
(with an added ID column) under pandas 2.3.3 / numpy 2.4.6 (see docs/provenance.md).

The Shepherd algorithm (both modes) is::

    np.random.seed(seed); random.seed(seed)
    sorted = df.sort_values('category')                  # pandas<3: numpy *quicksort* (unstable)
    holdout = per category (sorted keys): sample(n=holdout_per_category) via the global RandomState
    remaining = sorted minus holdout, re-indexed 0..N-1
    dirichlet mode: repeat until every client has >= min_require_size examples:
        for each category (order of first appearance): shuffle its positions, draw
        Dirichlet(alpha * 1_K) proportions, zero clients already holding >= N/K, renormalise, cut
    shards mode: split positions into shards_per_client*K contiguous shards, random.shuffle, pair up

IMPORTANT: pandas>=3 sorts *stably* by default, which changes the within-category order and therefore
the whole partition (and pandas 3 copy-on-write makes Shepherd's in-place shuffle raise). We therefore
reproduce the pandas<3 semantics explicitly with ``argsort(kind="quicksort")`` on an object array.

INFERENCE (reviewer decision R7): the paper states a 3:7 D1/D2 split but not its granularity. We split
*per client after partitioning*, deterministically, so every client keeps data for both stages.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from cg_fedllm.utils.seeding import numpy_rng


class PartitionError(RuntimeError):
    pass


@dataclass(frozen=True)
class ShepherdPartition:
    holdout_ids: list[int]
    remaining_ids: list[int]
    client_ids: list[list[int]]
    dirichlet_attempts: int
    numpy_version: str


def select_subset(
    source_ids: Sequence[int], categories: Sequence[str], per_category: int, seed: int
) -> list[int]:
    """Deterministic per-category subset (used only by small smoke configurations)."""
    ids = np.asarray(source_ids, dtype=np.int64)
    cats = np.asarray(categories, dtype=object)
    chosen: list[int] = []
    for cat in sorted(set(cats.tolist())):
        members = ids[cats == cat]
        if len(members) < per_category:
            raise PartitionError(f"category {cat!r} has only {len(members)} records < {per_category}")
        rng = numpy_rng(seed, "subset", cat)
        pick = np.sort(rng.choice(len(members), size=per_category, replace=False))
        chosen.extend(int(x) for x in members[pick])
    return sorted(chosen)


def shepherd_partition(
    source_ids: Sequence[int],
    categories: Sequence[str],
    *,
    num_clients: int,
    mode: str,
    holdout_per_category: int,
    dirichlet_alpha: float,
    min_require_size: int,
    shards_per_client: int,
    seed: int,
    max_attempts: int = 100_000,
) -> ShepherdPartition:
    """Reproduce Shepherd's client allocation on ``(source_ids, categories)`` (see module docstring)."""
    ids = np.asarray(source_ids, dtype=np.int64)
    cats = np.asarray(list(categories), dtype=object)
    if ids.shape != cats.shape:
        raise PartitionError("source_ids and categories must have the same length")
    rs = np.random.RandomState(seed)  # == np.random.seed(seed) followed by np.random.* calls
    py_rng = random.Random(seed)  # == random.seed(seed) followed by random.shuffle

    # pandas<3 DataFrame.sort_values(by=['category']) == nargsort(kind='quicksort') on the object column
    order = cats.argsort(kind="quicksort")
    sorted_ids, sorted_cats = ids[order], cats[order]

    # groupby('category') visits groups in sorted key order; sample(n) == RandomState.choice(len, n, False)
    holdout: list[int] = []
    for key in sorted(set(sorted_cats.tolist())):
        members = sorted_ids[sorted_cats == key]
        if len(members) < holdout_per_category:
            raise PartitionError(f"category {key!r} has fewer than {holdout_per_category} records")
        pick = rs.choice(len(members), size=holdout_per_category, replace=False)
        holdout.extend(int(x) for x in members[pick])
    holdout_set = set(holdout)
    keep = np.fromiter((int(i) not in holdout_set for i in sorted_ids), dtype=bool, count=len(sorted_ids))
    remaining_ids, remaining_cats = sorted_ids[keep], sorted_cats[keep]
    n_total = len(remaining_ids)

    attempts = 0
    if mode == "dirichlet":
        uniques = list(dict.fromkeys(remaining_cats.tolist()))  # pandas Series.unique(): first appearance
        min_size = 0
        idx_partition: list[list[int]] = []
        while min_size < min_require_size:
            attempts += 1
            if attempts > max_attempts:
                raise PartitionError(
                    f"no Dirichlet partition with min size >= {min_require_size} after {max_attempts} attempts "
                    f"(N={n_total}, K={num_clients})"
                )
            idx_partition = [[] for _ in range(num_clients)]
            for cat in uniques:
                rows_k = np.flatnonzero(remaining_cats == cat).astype(np.int64)
                rs.shuffle(rows_k)
                proportions = rs.dirichlet(np.repeat(dirichlet_alpha, num_clients))
                proportions = np.array(
                    [p * (len(idx_j) < n_total / num_clients) for p, idx_j in zip(proportions, idx_partition)]
                )
                proportions = proportions / proportions.sum()
                cuts = (np.cumsum(proportions) * len(rows_k)).astype(int)[:-1]
                idx_partition = [
                    idx_j + idx.tolist() for idx_j, idx in zip(idx_partition, np.split(rows_k, cuts))
                ]
                min_size = min(len(idx_j) for idx_j in idx_partition)
    elif mode == "shards":
        positions = np.arange(n_total, dtype=np.int64)
        shards = np.array_split(positions, int(shards_per_client * num_clients))
        py_rng.shuffle(shards)
        grouped = [shards[i : i + shards_per_client] for i in range(0, len(shards), shards_per_client)]
        idx_partition = [np.concatenate(grouped[n]).tolist() for n in range(num_clients)]
    else:
        raise PartitionError(f"unknown partition mode {mode!r}")

    client_ids = [[int(remaining_ids[p]) for p in idx] for idx in idx_partition]
    return ShepherdPartition(
        holdout_ids=holdout,
        remaining_ids=[int(x) for x in remaining_ids],
        client_ids=client_ids,
        dirichlet_attempts=attempts,
        numpy_version=np.__version__,
    )


def d1_count(n: int, d1_fraction: float) -> int:
    """Number of D1 examples for a client with ``n`` examples: floor(f*n + 0.5) (round half up)."""
    return int(math.floor(d1_fraction * n + 0.5))


def split_d1_d2(
    client_ids: Sequence[Sequence[int]], d1_fraction: float, seed: int
) -> tuple[list[list[int]], list[list[int]]]:
    """Deterministic per-client D1/D2 split preserving each client's original example order.

    For client ``c`` with ordered IDs ``L_c`` we draw ``perm = PCG64(derive_seed(seed, 'd1d2', c))``
    ``.permutation(len(L_c))`` and put the first ``d1_count`` permuted positions in D1.
    """
    d1_all: list[list[int]] = []
    d2_all: list[list[int]] = []
    for cid, ids in enumerate(client_ids):
        n = len(ids)
        chosen = set(numpy_rng(seed, "d1d2", cid).permutation(n)[: d1_count(n, d1_fraction)].tolist())
        d1_all.append([int(ids[j]) for j in range(n) if j in chosen])
        d2_all.append([int(ids[j]) for j in range(n) if j not in chosen])
    return d1_all, d2_all
