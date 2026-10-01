"""TGAP snapshot-set statistics (Phase 3, A4). Lightweight, committable summaries (DERIVED); the snapshot
payloads themselves stay outside Git.

Reported: snapshot/client counts and the participation histogram; per-snapshot and pooled distributions of the
A/B factors of the post-training state; innovation (end - start) distributions; inter-client cosine similarity
of innovations within a time index; inter-time cosine similarity of the sample-weighted mean innovation per time
index (the replayed FedAvg update); state cosine similarity; the train/validation split membership.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cg_fedllm.compression.diagnostics import factor_keys, flat
from cg_fedllm.compression.normalization import abs_quantiles
from cg_fedllm.federated.aggregation import aggregation_weights
from cg_fedllm.tgap.snapshots import load_states
from cg_fedllm.tgap.train_ae import split_indices

FACTORS = ("A", "B")
GRAM_CHUNK = 2**18


def _gram(rows: torch.Tensor) -> torch.Tensor:
    """float64 Gram matrix of float32 row vectors, accumulated in column chunks."""
    n, d = rows.shape
    g = torch.zeros(n, n, dtype=torch.float64)
    for c in range(0, d, GRAM_CHUNK):
        block = rows[:, c : c + GRAM_CHUNK].to(torch.float64)
        g += block @ block.T
    return g


def _cos(g: torch.Tensor) -> torch.Tensor:
    d = torch.sqrt(torch.clamp(torch.diagonal(g), min=0))
    denom = torch.outer(d, d)
    return torch.where(denom > 0, g / denom, torch.full_like(g, math.nan))


def _summary(vals: Sequence[float]) -> dict[str, float] | None:
    v = [x for x in vals if math.isfinite(x)]
    if not v:
        return None
    a = np.asarray(v, dtype=np.float64)
    return {"n": int(a.size), "mean": float(a.mean()), "min": float(a.min()), "p50": float(np.median(a)), "max": float(a.max())}


def _mean_offdiag(c: torch.Tensor, pairs: list[tuple[int, int]]) -> float | None:
    vals = [float(c[i, j]) for i, j in pairs if math.isfinite(float(c[i, j]))]
    return sum(vals) / len(vals) if vals else None


def tgap_statistics(snapshot_dir: Path, records: Sequence[dict], *, split: str = "temporal", val_fraction: float = 0.2, split_seed: int = 0) -> dict[str, Any]:
    n = len(records)
    times = sorted({int(r["time_index"]) for r in records})
    clients = Counter(int(r["client_id"]) for r in records)
    train_idx, val_idx = split_indices(records, split, val_fraction, split_seed)
    per: list[dict[str, Any]] = []
    rows = {f: [] for f in FACTORS}  # innovation flats (float32)
    srows = {f: [] for f in FACTORS}  # state flats (float32)
    stride = max(1, math.ceil(n * (records[0]["phi_shape"][1] * records[0]["phi_shape"][2] // 2) / 2**25))
    pooled_vals: dict[str, list[np.ndarray]] = defaultdict(list)
    keys = None
    starts: dict[str, dict[str, float]] = {}
    for r in records:
        start, end = load_states(snapshot_dir, r)
        if keys is None:
            keys = {f: factor_keys(end, f) for f in FACTORS}
        row: dict[str, Any] = {"time_index": int(r["time_index"]), "client_id": int(r["client_id"]), "num_samples": int(r["num_samples"])}
        for f in FACTORS:
            e, s = flat(end, keys[f]), flat(start, keys[f])
            u = e - s
            row[f"state_{f}_norm"] = float(e.norm())
            row[f"state_{f}_rms"] = float(e.norm() / math.sqrt(e.numel()))
            row[f"state_{f}_max_abs"] = float(e.abs().max())
            row[f"innovation_{f}_norm"] = float(u.norm())
            row[f"innovation_{f}_rel_to_state"] = float(u.norm() / e.norm()) if float(e.norm()) > 0 else math.nan
            rows[f].append(u.to(torch.float32))
            srows[f].append(e.to(torch.float32))
            pooled_vals[f"state_{f}"].append(e[::stride].to(torch.float32).numpy())
            pooled_vals[f"innovation_{f}"].append(u[::stride].to(torch.float32).numpy())
            if r["start_adapter_hash"] not in starts:
                starts[r["start_adapter_hash"]] = {}
            starts[r["start_adapter_hash"]][f] = float(s.norm())
        row["innovation_norm"] = math.hypot(row["innovation_A_norm"], row["innovation_B_norm"])
        row["state_norm"] = math.hypot(row["state_A_norm"], row["state_B_norm"])
        row["innovation_rel_to_state"] = row["innovation_norm"] / row["state_norm"] if row["state_norm"] > 0 else math.nan
        extra = r.get("extra", {})
        for k in ("num_optimizer_steps", "num_label_tokens", "mean_loss", "wall_time_s"):
            if k in extra:
                row[k] = extra[k]
        per.append(row)

    g_inn = {f: _gram(torch.stack(rows[f])) for f in FACTORS}
    g_inn["all"] = g_inn["A"] + g_inn["B"]
    g_st = {f: _gram(torch.stack(srows[f])) for f in FACTORS}
    g_st["all"] = g_st["A"] + g_st["B"]
    del rows, srows
    by_t: dict[int, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        by_t[int(r["time_index"])].append(i)
    same_t = [(i, j) for t in times for a, i in enumerate(by_t[t]) for j in by_t[t][a + 1 :]]
    same_c = [(i, j) for i in range(n) for j in range(i + 1, n) if per[i]["client_id"] == per[j]["client_id"]]
    diff_t = [(i, j) for i in range(n) for j in range(i + 1, n) if per[i]["time_index"] != per[j]["time_index"]]
    cos = {}
    for g_name, grams in (("innovation", g_inn), ("state", g_st)):
        for f in ("all", "A", "B"):
            c = _cos(grams[f])
            cos[f"{g_name}_{f}"] = {
                "inter_client_same_time_mean": _mean_offdiag(c, same_t),
                "same_client_different_time_mean": _mean_offdiag(c, same_c),
                "different_time_mean": _mean_offdiag(c, diff_t),
            }
    # sample-weighted mean innovation per time index (= the FedAvg update when all clients share a start)
    w_t = {t: aggregation_weights([per[i]["num_samples"] for i in by_t[t]], "sample_weighted_mean").to(torch.float64) for t in times}
    m = torch.zeros(len(times), n, dtype=torch.float64)
    for a, t in enumerate(times):
        for w, i in zip(w_t[t].tolist(), by_t[t]):
            m[a, i] = w
    agg_cos = {f: _cos(m @ g_inn[f] @ m.T) for f in ("all", "A", "B")}
    adjacent = [(a, a + 1) for a in range(len(times) - 1)]
    inter_time = {
        f: {
            "adjacent_time_mean": _mean_offdiag(agg_cos[f], adjacent),
            "all_pairs_mean": _mean_offdiag(agg_cos[f], [(a, b) for a in range(len(times)) for b in range(a + 1, len(times))]),
            "matrix": [[round(float(v), 6) for v in row] for row in agg_cos[f].tolist()],
        }
        for f in ("all", "A", "B")
    }
    by_time = {}
    for t in times:
        idx = by_t[t]
        by_time[str(t)] = {
            "clients": [per[i]["client_id"] for i in idx],
            "num_samples": [per[i]["num_samples"] for i in idx],
            "state_B_norm_mean": float(np.mean([per[i]["state_B_norm"] for i in idx])),
            "innovation_norm_mean": float(np.mean([per[i]["innovation_norm"] for i in idx])),
            "innovation_rel_to_state_mean": float(np.mean([per[i]["innovation_rel_to_state"] for i in idx])),
        }
    hist = Counter(clients.values())
    return {
        "num_snapshots": n,
        "num_time_indices": len(times),
        "time_indices": times,
        "unique_clients": len(clients),
        "participation_histogram": {str(k): v for k, v in sorted(hist.items())},
        "participations_per_client": {str(c): k for c, k in sorted(clients.items())},
        "distinct_start_states": len(starts),
        "num_samples": _summary([p["num_samples"] for p in per]),
        "per_snapshot_summary": {
            k: _summary([p[k] for p in per])
            for k in (
                "state_A_norm", "state_B_norm", "state_A_rms", "state_B_rms", "state_A_max_abs", "state_B_max_abs",
                "innovation_norm", "innovation_A_norm", "innovation_B_norm", "innovation_rel_to_state",
                "innovation_A_rel_to_state", "innovation_B_rel_to_state",
            )
        },
        "pooled_abs_value_quantiles": {k: abs_quantiles(v, (0.5, 0.9, 0.99)) for k, v in sorted(pooled_vals.items())},
        "energy": {
            "state_A_sq_total": float(torch.diagonal(g_st["A"]).sum()),
            "state_B_sq_total": float(torch.diagonal(g_st["B"]).sum()),
            "innovation_A_sq_total": float(torch.diagonal(g_inn["A"]).sum()),
            "innovation_B_sq_total": float(torch.diagonal(g_inn["B"]).sum()),
        },
        "cosine": cos,
        "inter_time_aggregate_innovation_cosine": inter_time,
        "by_time": by_time,
        "split": {
            "rule": f"{split}, validation = last {val_fraction:.0%} of time indices",
            "train": [[per[i]["time_index"], per[i]["client_id"]] for i in train_idx],
            "val": [[per[i]["time_index"], per[i]["client_id"]] for i in val_idx],
            "train_time_indices": sorted({per[i]["time_index"] for i in train_idx}),
            "val_time_indices": sorted({per[i]["time_index"] for i in val_idx}),
        },
        "per_snapshot": per,
    }
