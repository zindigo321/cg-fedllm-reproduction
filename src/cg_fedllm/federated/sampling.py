"""Client sampling.

BEHAVIOR RECONSTRUCTED FROM SHEPHERD (``fed_utils/client_participation_scheduling.py`` @ bcffa00):
``np.random.seed(round_index); K = max(int(fraction * N), 1);
set(np.random.choice(np.arange(N), K, replace=False))``. We reproduce the same draw with an explicit
``RandomState(round_index)`` and return the selected clients in ascending order (Shepherd iterates a Python
``set``; our fixed ascending order only affects float summation order during aggregation).
"""

from __future__ import annotations

import numpy as np


def num_selected(num_clients: int, fraction: float) -> int:
    return max(int(fraction * num_clients), 1)


def shepherd_select_clients(num_clients: int, fraction: float, round_index: int) -> list[int]:
    rs = np.random.RandomState(round_index)
    chosen = rs.choice(np.arange(num_clients), num_selected(num_clients, fraction), replace=False)
    return sorted(int(c) for c in chosen)
