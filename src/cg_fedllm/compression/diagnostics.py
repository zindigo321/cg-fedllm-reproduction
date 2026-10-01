"""Factor-aware reconstruction diagnostics (Phase 3, reviewer item A1).

A whole-tensor error hides what matters for FAF. LoRA A, mostly the shared random initialisation, carries far
more energy than B. A codec can therefore look acceptable on whole-state error while destroying B. FAF also
depends on the client's *innovation* (its update relative to the round-start state), not on the absolute state.

For one predictor (a codec, or a reference such as zero / train-mean / identity) over a set of TGAP snapshots
this module reports pooled metrics for non-overlapping groups:

  transmitted/{all,A,B}       the representation the codec actually sees (adapter_state or adapter_delta)
  state/{all,A,B}             the recovered post-training adapter (equal to ``transmitted`` for adapter_state)
  innovation/{all,A,B}        recovered state minus the client's round-start state
  product/{state,innovation}  the effective weight change B A of every module (exact, via r x r Gram matrices)
  aggregate/{update,state}    offline replay of sample-weighted FedAvg per time index (identity = true states)
  shift_control               innovation error when every reconstruction is paired with the *next* client of the
                              same time index: an input-independent decoder scores the same either way

Pooled definitions over snapshots i (float64 sums):

  mse = sum||v_hat - v||^2 / n           rel_sq_error = sum||v_hat - v||^2 / sum||v||^2
  cosine = sum<v_hat, v> / sqrt(sum||v_hat||^2 * sum||v||^2)        norm_ratio = sqrt(sum||v_hat||^2 / sum||v||^2)
  aggregate rel_l2_error = sqrt(sum_t||U_hat_t - U_t||^2 / sum_t||U_t||^2)

Absolute-error quantiles (p50/p95) use the transmitted-representation error (the state and innovation errors are
the same vector up to float rounding); they are exact unless ``error_stride > 1``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from cg_fedllm.compression.layout import Layout, LoRAGeometry
from cg_fedllm.compression.normalization import abs_quantiles
from cg_fedllm.compression.representation import to_representation
from cg_fedllm.federated.aggregation import aggregation_weights
from cg_fedllm.models.adapter import AdapterState, parse_key

FACTORS = ("A", "B")
# cos >= 0.90 with an error orthogonal to the innovation needs ||err||^2 <= (1/0.9^2 - 1) ||u||^2
COSINE_090_ERROR_BUDGET = 1.0 / 0.9**2 - 1.0


@dataclass
class SnapshotItem:
    index: int
    time_index: int
    client_id: int
    num_samples: int
    start: AdapterState
    end: AdapterState


Predictor = Callable[[torch.Tensor, SnapshotItem], torch.Tensor]  # Phi(transmitted) -> Phi(reconstruction)


def factor_keys(state: AdapterState, factor: str) -> list[str]:
    return [k for k in state.keys() if parse_key(k)[3] == factor]


def flat(state: AdapterState | Mapping[str, torch.Tensor], keys: Sequence[str]) -> torch.Tensor:
    tensors = state.tensors if isinstance(state, AdapterState) else state
    return torch.cat([tensors[k].reshape(-1).to(torch.float64) for k in keys])


def module_pairs(state: AdapterState) -> list[tuple[str, str]]:
    """(A key, B key) for every adapted module, in canonical order."""
    out = []
    for k in factor_keys(state, "A"):
        b = k[: -len("lora_A.weight")] + "lora_B.weight"
        if b not in state.tensors:
            raise ValueError(f"{k}: missing matching B factor")
        out.append((k, b))
    return out


def _nan_max(a: float, b: float) -> float:
    return a if not math.isfinite(a) else (b if not math.isfinite(b) else max(a, b))


def lowrank_inner(l1: torch.Tensor, r1: torch.Tensor, l2: torch.Tensor, r2: torch.Tensor) -> float:
    """<L1 R1, L2 R2>_F without forming the products: sum((L1^T L2) * (R1 R2^T)) (float64)."""
    return float(((l1.T @ l2) * (r1 @ r2.T)).sum())


class Pooled:
    """Running pooled sums for one (group, factor)."""

    def __init__(self, keep_errors: bool = False, error_stride: int = 1) -> None:
        self.sse = self.sig = self.hat = self.dot = 0.0
        self.n = 0
        self.max_abs = 0.0
        self.finite = True
        self.errors: list[np.ndarray] | None = [] if keep_errors else None
        self.error_stride = max(1, int(error_stride))
        self.per_rse: list[float] = []
        self.per_cos: list[float] = []

    def add(self, v: torch.Tensor, v_hat: torch.Tensor) -> None:
        d = v_hat - v
        sse, sig, hat, dot = float(d @ d), float(v @ v), float(v_hat @ v_hat), float(v @ v_hat)
        self.add_sums(sse, sig, hat, dot, v.numel())
        self.max_abs = _nan_max(self.max_abs, float(d.abs().max()))
        if self.errors is not None:
            self.errors.append(d[:: self.error_stride].to(torch.float32).numpy())

    def add_sums(self, sse: float, sig: float, hat: float, dot: float, n: int) -> None:
        self.finite = self.finite and all(math.isfinite(v) for v in (sse, hat, dot))
        self.sse += sse
        self.sig += sig
        self.hat += hat
        self.dot += dot
        self.n += int(n)
        self.per_rse.append(sse / sig if sig > 0 else math.nan)
        self.per_cos.append(dot / math.sqrt(sig * hat) if sig > 0 and hat > 0 else math.nan)

    @staticmethod
    def merged(parts: Iterable[Pooled]) -> Pooled:
        out = Pooled()
        for p in parts:
            out.sse += p.sse
            out.sig += p.sig
            out.hat += p.hat
            out.dot += p.dot
            out.n += p.n
            out.finite = out.finite and p.finite
            out.max_abs = _nan_max(out.max_abs, p.max_abs)
        return out

    def result(self) -> dict[str, Any]:
        sig, hat = self.sig, self.hat
        out: dict[str, Any] = {
            "n": self.n,
            "finite": self.finite,
            "signal_sq": sig,
            "recon_sq": hat,
            "sse": self.sse,
            "mse": self.sse / self.n if self.n else math.nan,
            "rel_sq_error": self.sse / sig if sig > 0 else math.nan,
            "cosine": self.dot / math.sqrt(sig * hat) if sig > 0 and hat > 0 else math.nan,
            "norm_ratio": math.sqrt(hat / sig) if sig > 0 else math.nan,
            "max_abs_error": self.max_abs,
        }
        for name, vals in (("rel_sq_error", self.per_rse), ("cosine", self.per_cos)):
            finite = [v for v in vals if math.isfinite(v)]
            if finite:
                out[f"per_snapshot_{name}"] = {"min": min(finite), "median": float(np.median(finite)), "max": max(finite)}
        return out


def _group_result(parts: dict[str, Pooled], errors: dict[str, list[np.ndarray]] | None = None, exact: bool = True) -> dict[str, Any]:
    out = {f: parts[f].result() for f in FACTORS}
    out["all"] = Pooled.merged(parts[f] for f in FACTORS).result()
    if errors is not None:
        qs = (0.5, 0.95)
        for f in FACTORS:
            q = abs_quantiles(errors[f], qs)
            out[f].update({"abs_error_p50": q.get("p50"), "abs_error_p95": q.get("p95"), "quantiles_exact": exact and q.get("exact", True)})
        q = abs_quantiles(errors["A"] + errors["B"], qs)
        out["all"].update({"abs_error_p50": q.get("p50"), "abs_error_p95": q.get("p95"), "quantiles_exact": exact and q.get("exact", True)})
    return out


def _agg_result(sums: dict[str, list[float]]) -> dict[str, Any]:
    """``sums[factor] = [sse, sig, hat, dot]`` pooled over time indices; ``all`` = A and B together."""
    rows = {f: sums[f] for f in FACTORS}
    rows["all"] = [sum(sums[f][i] for f in FACTORS) for i in range(4)]
    out: dict[str, Any] = {}
    for f, (sse, sig, hat, dot) in rows.items():
        out[f] = {
            "rel_l2_error": math.sqrt(sse / sig) if sig > 0 else math.nan,
            "cosine": dot / math.sqrt(sig * hat) if sig > 0 and hat > 0 else math.nan,
            "norm_ratio": math.sqrt(hat / sig) if sig > 0 else math.nan,
            "signal_sq": sig,
        }
    return out


class PredictorDiagnostics:
    """Accumulates every A1 metric for one predictor over snapshots grouped by time index."""

    def __init__(self, representation: str, *, error_stride: int = 1) -> None:
        if representation not in ("adapter_state", "adapter_delta"):
            raise ValueError(f"unknown representation {representation!r}")
        self.rep = representation
        self.error_stride = error_stride
        self.groups = {g: {f: Pooled(keep_errors=(g == "transmitted"), error_stride=error_stride) for f in FACTORS} for g in ("transmitted", "state", "innovation")}
        self.product = {g: Pooled() for g in ("state", "innovation")}
        self.agg = {g: {f: [0.0, 0.0, 0.0, 0.0] for f in FACTORS} for g in ("update", "state")}
        self.shift = {"matched": Pooled(), "shifted": Pooled()}
        self.num_snapshots = 0
        self.time_indices: list[int] = []
        self.finite_outputs = True

    def add_time(self, items: Sequence[SnapshotItem], truths: Sequence[dict[str, Any]], predictor: Predictor, layout: Layout, geom: LoRAGeometry, xs: Sequence[torch.Tensor]) -> None:
        """``truths[i]`` holds the float64 flats of item i (``rep``/``end``/``start`` per factor) and its key lists."""
        weights = aggregation_weights([it.num_samples for it in items], "sample_weighted_mean").to(torch.float64)
        agg_u = {f: None for f in FACTORS}
        agg_uh = {f: None for f in FACTORS}
        agg_e = {f: None for f in FACTORS}
        agg_eh = {f: None for f in FACTORS}
        innov_hat: list[torch.Tensor] = []
        innov_true: list[torch.Tensor] = []
        for w, it, tr, x in zip(weights.tolist(), items, truths, xs):
            x_hat = predictor(x, it)
            if x_hat.shape != x.shape:
                raise ValueError(f"predictor returned shape {tuple(x_hat.shape)} for input {tuple(x.shape)}")
            self.finite_outputs = self.finite_outputs and bool(torch.isfinite(x_hat).all())
            rep_hat = layout.inverse(x_hat.to(torch.float32), geom)
            uh_cat, u_cat = [], []
            e_hat_t: dict[str, torch.Tensor] = {}
            for f in FACTORS:
                keys = tr["keys"][f]
                r_hat = flat(rep_hat, keys)
                e_hat = r_hat if self.rep == "adapter_state" else tr["start"][f] + r_hat
                u = tr["end"][f] - tr["start"][f]
                u_hat = e_hat - tr["start"][f]
                self.groups["transmitted"][f].add(tr["rep"][f], r_hat)
                self.groups["state"][f].add(tr["end"][f], e_hat)
                self.groups["innovation"][f].add(u, u_hat)
                for store, val in ((agg_u, u), (agg_uh, u_hat), (agg_e, tr["end"][f]), (agg_eh, e_hat)):
                    store[f] = val * w if store[f] is None else store[f] + val * w
                uh_cat.append(u_hat)
                u_cat.append(u)
                if self.rep == "adapter_state":
                    e_hat_t.update({k: rep_hat.tensors[k].to(torch.float64) for k in keys})
                else:
                    e_hat_t.update({k: it.start.tensors[k].to(torch.float64) + rep_hat.tensors[k].to(torch.float64) for k in keys})
            self._add_product(it, e_hat_t)
            innov_hat.append(torch.cat(uh_cat))
            innov_true.append(torch.cat(u_cat))
            self.num_snapshots += 1
        for f in FACTORS:
            for g, (true, hat) in (("update", (agg_u[f], agg_uh[f])), ("state", (agg_e[f], agg_eh[f]))):
                d = hat - true
                s = self.agg[g][f]
                s[0] += float(d @ d)
                s[1] += float(true @ true)
                s[2] += float(hat @ hat)
                s[3] += float(true @ hat)
        k = len(items)
        if k >= 2:  # pair each reconstruction with the true innovation of the next client of the same time index
            for i in range(k):
                self.shift["matched"].add(innov_true[i], innov_hat[i])
                self.shift["shifted"].add(innov_true[i], innov_hat[(i + 1) % k])
        self.time_indices.append(int(items[0].time_index))

    def _add_product(self, it: SnapshotItem, e_hat: Mapping[str, torch.Tensor]) -> None:
        sums = {g: [0.0, 0.0, 0.0, 0.0] for g in ("state", "innovation")}
        n = 0
        for ka, kb in module_pairs(it.end):
            a, b = it.end.tensors[ka].double(), it.end.tensors[kb].double()
            a_s, b_s = it.start.tensors[ka].double(), it.start.tensors[kb].double()
            a_h, b_h = e_hat[ka], e_hat[kb]
            # error E = B_h A_h - B A = [B_h - B | B] [A_h ; A_h - A]
            le, re_ = torch.cat([b_h - b, b], 1), torch.cat([a_h, a_h - a], 0)
            sse = lowrank_inner(le, re_, le, re_)
            sums["state"][0] += sse
            sums["state"][1] += lowrank_inner(b, a, b, a)
            sums["state"][2] += lowrank_inner(b_h, a_h, b_h, a_h)
            sums["state"][3] += lowrank_inner(b, a, b_h, a_h)
            # innovation u = B A - B_s A_s = [B - B_s | B_s][A ; A - A_s]; u_hat likewise with the reconstruction
            lu, ru = torch.cat([b - b_s, b_s], 1), torch.cat([a, a - a_s], 0)
            luh, ruh = torch.cat([b_h - b_s, b_s], 1), torch.cat([a_h, a_h - a_s], 0)
            sums["innovation"][0] += sse
            sums["innovation"][1] += lowrank_inner(lu, ru, lu, ru)
            sums["innovation"][2] += lowrank_inner(luh, ruh, luh, ruh)
            sums["innovation"][3] += lowrank_inner(lu, ru, luh, ruh)
            n += b.shape[0] * a.shape[1]
        for g in ("state", "innovation"):
            self.product[g].add_sums(*sums[g], n)

    def result(self) -> dict[str, Any]:
        tx_errors = {f: self.groups["transmitted"][f].errors or [] for f in FACTORS}
        out: dict[str, Any] = {
            "representation": self.rep,
            "num_snapshots": self.num_snapshots,
            "time_indices": sorted(self.time_indices),
            "finite_outputs": self.finite_outputs,
            "transmitted": _group_result(self.groups["transmitted"], tx_errors, exact=self.error_stride == 1),
            "state": _group_result(self.groups["state"]),
            "innovation": _group_result(self.groups["innovation"]),
            "product": {g: self.product[g].result() for g in ("state", "innovation")},
            "aggregate": {g: _agg_result(self.agg[g]) for g in ("update", "state")},
        }
        tx_all = out["transmitted"]["all"]
        # paper-style SNR (DERIVED in Phase 1): signal energy of ONE snapshot over the per-element MSE, here the mean
        # per-snapshot signal energy over the pooled per-element MSE (reported, never gated: it scales with n)
        mean_sig = tx_all["signal_sq"] / self.num_snapshots if self.num_snapshots else math.nan
        out["snr_paper"] = mean_sig / tx_all["mse"] if tx_all["mse"] and tx_all["mse"] > 0 else math.inf
        out["snr_standard_db"] = 10 * math.log10(tx_all["signal_sq"] / tx_all["sse"]) if tx_all["sse"] > 0 and tx_all["signal_sq"] > 0 else math.inf
        m, s = self.shift["matched"].result(), self.shift["shifted"].result()
        out["shift_control"] = {
            "innovation_rel_sq_error_matched": m["rel_sq_error"],
            "innovation_rel_sq_error_shifted": s["rel_sq_error"],
            "innovation_cosine_matched": m["cosine"],
            "innovation_cosine_shifted": s["cosine"],
            "pairs": len(self.shift["matched"].per_rse),
        }
        st, inn = out["state"]["all"], out["innovation"]["all"]
        rho = inn["signal_sq"] / st["signal_sq"] if st["signal_sq"] > 0 else math.nan
        out["innovation_energy_fraction"] = rho
        out["state_rel_sq_error_budget_for_innovation_cosine_0_9"] = COSINE_090_ERROR_BUDGET * rho
        return out


def prepare_truth(item: SnapshotItem, representation: str) -> dict[str, Any]:
    rep = to_representation(item.end, item.start, representation)
    keys = {f: factor_keys(item.end, f) for f in FACTORS}
    return {
        "keys": keys,
        "rep": {f: flat(rep, keys[f]) for f in FACTORS},
        "end": {f: flat(item.end, keys[f]) for f in FACTORS},
        "start": {f: flat(item.start, keys[f]) for f in FACTORS},
    }


def evaluate_predictors(
    groups: Iterable[Sequence[SnapshotItem]],
    representation: str,
    layout: Layout,
    geom: LoRAGeometry,
    predictors: Mapping[str, Predictor],
    *,
    error_stride: int = 1,
) -> dict[str, dict[str, Any]]:
    """Evaluate every predictor on snapshots supplied one time index at a time (each group = one time index)."""
    acc = {name: PredictorDiagnostics(representation, error_stride=error_stride) for name in predictors}
    for items in groups:
        items = sorted(items, key=lambda it: it.client_id)
        if len({it.time_index for it in items}) != 1:
            raise ValueError("every group must contain the snapshots of exactly one time index")
        truths = [prepare_truth(it, representation) for it in items]
        xs = [layout.forward(to_representation(it.end, it.start, representation), geom) for it in items]
        for name, pred in predictors.items():
            acc[name].add_time(items, truths, pred, layout, geom, xs)
    return {name: a.result() for name, a in acc.items()}
