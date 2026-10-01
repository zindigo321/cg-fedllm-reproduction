"""Phase-4 forensic representations (PHASE4-FORENSIC). None of them is CG-FedLLM's paper-specified object.

* ``balanced_effective_state`` (R2): balanced canonical rank-r factors of every module's ``M = s B A``. The output
  uses LoRA key names and shapes (A_c: r x d_in, B_c: d_out x r), so the Phi layout and the AE apply unchanged;
  it is gauge invariant (any B Q, Q^-1 A gives the same factors).
* ``balanced_effective_delta_r8`` (R3): balanced factors of the best rank-r approximation of the effective
  increment ``dM = s (B_end A_end - B_start A_start)`` (exact rank <= 2r; the truncation is measured, F4).
* ``mean_step_gradient`` (R4) is built from recorded gradients (:mod:`cg_fedllm.forensics.gradients`).
"""

from __future__ import annotations

import statistics
from typing import Any

import torch

from cg_fedllm.compression.gauge import balanced_factors, effective_delta, module_pairs
from cg_fedllm.models.adapter import AdapterState


def balanced_effective_state(state: AdapterState, s: float) -> tuple[AdapterState, dict[str, Any]]:
    out: dict[str, torch.Tensor] = {}
    tied = zero = 0
    for ka, kb in module_pairs(state):
        c = balanced_factors(state.tensors[ka], state.tensors[kb], s)
        out[ka] = c["A"].to(torch.float32)
        out[kb] = c["B"].to(torch.float32)
        tied += c["tied"]
        zero += c["numerically_zero"]
    return AdapterState(out), {"tied_singular_values": tied, "numerically_zero_components": zero}


def balanced_effective_delta(start: AdapterState, end: AdapterState, s: float, rank: int) -> tuple[AdapterState, dict[str, Any]]:
    out: dict[str, torch.Tensor] = {}
    mods = []
    energy = kept = 0.0
    for ka, kb in module_pairs(end):
        e = effective_delta(end.tensors[ka], end.tensors[kb], start.tensors[ka], start.tensors[kb], s, rank)
        out[ka] = e["A"].to(torch.float32)
        out[kb] = e["B"].to(torch.float32)
        sv = e["singular_values"]
        energy += e["energy"]
        kept += float((sv[:rank] ** 2).sum())
        mods.append(
            {
                "module": ka[: -len(".lora_A.weight")],
                "exact_rank": e["exact_rank"],
                "retained_energy": e["retained_energy"],
                "truncation_rel_fro_error": e["truncation_rel_fro_error"],
                "rank_r_product_cosine": e["rank_r_product_cosine"],
                "singular_values": [float(x) for x in sv],
                "tied": e["tied"],
            }
        )
    retained = kept / energy if energy > 0 else float("nan")
    stats = {
        "rank": rank,
        "energy_weighted_retained_energy": retained,
        "truncation_rel_fro_error": (1.0 - retained) ** 0.5 if energy > 0 else float("nan"),
        "rank_r_product_cosine": retained**0.5 if energy > 0 else float("nan"),
        "exact_rank_distribution": dict(sorted(statistics_counter(m["exact_rank"] for m in mods).items())),
        "module_retained_energy": {"min": min(m["retained_energy"] for m in mods), "median": statistics.median(m["retained_energy"] for m in mods), "max": max(m["retained_energy"] for m in mods)},
        "per_module": mods,
    }
    return AdapterState(out), stats


def statistics_counter(values) -> dict[int, int]:
    out: dict[int, int] = {}
    for v in values:
        out[int(v)] = out.get(int(v), 0) + 1
    return out
