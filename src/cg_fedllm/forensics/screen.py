"""Phase-4 F6 fixed compressibility screen (PHASE4-FORENSIC). Definitions: docs/phase4_preregistration.md, sec. 4.

Every candidate supplies, per snapshot: the transmitted factors (LoRA-shaped, so the Phi layout and the ResNet AE
apply unchanged) and an effective-product target, as low-rank factorisations per module (float64, no d x d):

  R2 balanced_effective_state     target M = B_c A_c (= s B A)           update-relevant: M - M_start (innovation)
  R3 balanced_effective_delta_r8  target = transmitted rank-8 product     update-relevant: the product itself
                                  (also reported vs the exact dM)
  R4 mean_step_gradient           target = s (g_B A_s + B_s g_A)          update-relevant: the product itself
                                  (first-order effective change at the client's start state)
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from cg_fedllm.compression.diagnostics import Pooled, lowrank_inner
from cg_fedllm.compression.gauge import module_pairs
from cg_fedllm.compression.layout import Layout, LoRAGeometry
from cg_fedllm.federated.aggregation import aggregation_weights
from cg_fedllm.models.adapter import AdapterState, parse_key

GATE = {"rep_cosine_min": 0.90, "rep_rel_sq_error_max": 0.50, "product_cosine_min": 0.90, "product_rel_fro_error_max": 0.50, "train_mean_factor": 0.8}
KINDS = ("balanced_effective_state", "balanced_effective_delta_r8", "mean_step_gradient")


@dataclass
class ScreenItem:
    index: int
    time_index: int
    client_id: int
    num_samples: int
    rep: AdapterState  # transmitted factors
    start: AdapterState  # client start state (raw LoRA factors)
    end: AdapterState  # client end state (raw LoRA factors)


def _d(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to("cpu", torch.float64)


def product_factors(kind: str, rep: AdapterState, item: ScreenItem, s: float) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Per module (A key): (L, R) with the effective product P = L @ R, for transmitted factors ``rep``."""
    out = {}
    for ka, kb in module_pairs(rep):
        a, b = _d(rep.tensors[ka]), _d(rep.tensors[kb])
        if kind in ("balanced_effective_state", "balanced_effective_delta_r8"):
            out[ka] = (b, a)
        elif kind == "mean_step_gradient":
            a_s, b_s = _d(item.start.tensors[ka]), _d(item.start.tensors[kb])
            out[ka] = (s * torch.cat([b, b_s], 1), torch.cat([a_s, a], 0))
        else:
            raise ValueError(kind)
    return out


def start_product(item: ScreenItem, s: float) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    return {ka: (s * _d(item.start.tensors[kb]), _d(item.start.tensors[ka])) for ka, kb in module_pairs(item.start)}


def exact_delta_product(item: ScreenItem, s: float) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    return {
        ka: (s * torch.cat([_d(item.end.tensors[kb]), -_d(item.start.tensors[kb])], 1), torch.cat([_d(item.end.tensors[ka]), _d(item.start.tensors[ka])], 0))
        for ka, kb in module_pairs(item.end)
    }


def _minus(p: dict, q: dict) -> dict:
    return {k: (torch.cat([p[k][0], -q[k][0]], 1), torch.cat([p[k][1], q[k][1]], 0)) for k in p}


def _sums(p_hat: dict, p: dict) -> tuple[float, float, float, float]:
    """(sse, sig, hat, dot) between two per-module low-rank products."""
    sse = sig = hat = dot = 0.0
    for k in p:
        lh, rh = p_hat[k]
        lt, rt = p[k]
        sig += lowrank_inner(lt, rt, lt, rt)
        hat += lowrank_inner(lh, rh, lh, rh)
        dot += lowrank_inner(lh, rh, lt, rt)
        le, re_ = torch.cat([lh, -lt], 1), torch.cat([rh, rt], 0)
        sse += lowrank_inner(le, re_, le, re_)
    return sse, sig, hat, dot


def _weighted_sum(products: Sequence[dict], weights: Sequence[float]) -> dict:
    keys = products[0].keys()
    return {k: (torch.cat([w * p[k][0] for p, w in zip(products, weights)], 1), torch.cat([p[k][1] for p in products], 0)) for k in keys}


def _summ(sse: float, sig: float, hat: float, dot: float) -> dict[str, float]:
    sse, hat = max(sse, 0.0), max(hat, 0.0)  # squared norms; Gram-form rounding can give -1e-35 for an exact match
    return {
        "rel_sq_error": sse / sig if sig > 0 else math.nan,
        "rel_fro_error": math.sqrt(sse / sig) if sig > 0 else math.nan,
        "cosine": dot / math.sqrt(sig * hat) if sig > 0 and hat > 0 else math.nan,
        "norm_ratio": math.sqrt(hat / sig) if sig > 0 else math.nan,
        "signal_sq": sig,
    }


class _Acc:
    def __init__(self) -> None:
        self.v = [0.0, 0.0, 0.0, 0.0]

    def add(self, t: tuple[float, float, float, float]) -> None:
        self.v = [a + b for a, b in zip(self.v, t)]

    def result(self) -> dict[str, float]:
        return _summ(*self.v)


def evaluate(groups: Sequence[Sequence[ScreenItem]], kind: str, s: float, layout: Layout, geom: LoRAGeometry, predictors: dict[str, Callable[[torch.Tensor, ScreenItem], torch.Tensor]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, pred in predictors.items():
        rep = {f: Pooled() for f in ("A", "B")}
        prod, upd, upd_shift, exact = _Acc(), _Acc(), _Acc(), _Acc()
        agg, agg_upd, agg_grad = _Acc(), _Acc(), {"A": [0.0] * 4, "B": [0.0] * 4}
        finite = True
        for items in groups:
            items = sorted(items, key=lambda it: it.client_id)
            hats, targets, upd_hats, upd_targets = [], [], [], []
            rep_hats = []
            for it in items:
                x = layout.forward(it.rep, geom)
                x_hat = pred(x, it)
                finite = finite and bool(torch.isfinite(x_hat).all())
                r_hat = layout.inverse(x_hat.to(torch.float32), geom)
                rep_hats.append(r_hat)
                for f in ("A", "B"):
                    keys = [k for k in it.rep.keys() if parse_key(k)[3] == f]
                    rep[f].add(torch.cat([_d(it.rep.tensors[k]).reshape(-1) for k in keys]), torch.cat([_d(r_hat.tensors[k]).reshape(-1) for k in keys]))
                p_hat, p = product_factors(kind, r_hat, it, s), product_factors(kind, it.rep, it, s)
                prod.add(_sums(p_hat, p))
                if kind == "balanced_effective_delta_r8":
                    exact.add(_sums(p_hat, exact_delta_product(it, s)))
                if kind == "balanced_effective_state":
                    ms = start_product(it, s)
                    u_hat, u = _minus(p_hat, ms), _minus(p, ms)
                else:
                    u_hat, u = p_hat, p
                upd.add(_sums(u_hat, u))
                hats.append(p_hat)
                targets.append(p)
                upd_hats.append(u_hat)
                upd_targets.append(u)
            k = len(items)
            if k >= 2:
                for i in range(k):
                    upd_shift.add(_sums(upd_hats[(i + 1) % k], upd_targets[i]))
            w = aggregation_weights([it.num_samples for it in items], "sample_weighted_mean").tolist()
            agg.add(_sums(_weighted_sum(hats, w), _weighted_sum(targets, w)))
            if kind == "balanced_effective_state":
                agg_upd.add(_sums(_weighted_sum(upd_hats, w), _weighted_sum(upd_targets, w)))
            if kind == "mean_step_gradient":
                for f in ("A", "B"):
                    keys = [kk for kk in items[0].rep.keys() if parse_key(kk)[3] == f]
                    g = sum(wi * torch.cat([_d(it.rep.tensors[kk]).reshape(-1) for kk in keys]) for wi, it in zip(w, items))
                    gh = sum(wi * torch.cat([_d(rh.tensors[kk]).reshape(-1) for kk in keys]) for wi, rh in zip(w, rep_hats))
                    dd = gh - g
                    agg_grad[f] = [a + b for a, b in zip(agg_grad[f], (float(dd @ dd), float(g @ g), float(gh @ gh), float(g @ gh)))]
        rep_all = Pooled.merged(rep.values()).result()
        res: dict[str, Any] = {
            "finite_outputs": finite,
            "representation": {"all": rep_all, "A": rep["A"].result(), "B": rep["B"].result()},
            "product": prod.result(),
            "update_relevant": upd.result(),
            "shift_control": {"matched_rel_sq_error": upd.result()["rel_sq_error"], "shifted_rel_sq_error": upd_shift.result()["rel_sq_error"]},
            "aggregate_product": agg.result(),
        }
        if kind == "balanced_effective_delta_r8":
            res["product_vs_exact_delta"] = exact.result()
        if kind == "balanced_effective_state":
            res["aggregate_innovation"] = agg_upd.result()
        if kind == "mean_step_gradient":
            tot = [agg_grad["A"][i] + agg_grad["B"][i] for i in range(4)]
            res["aggregate_gradient_space"] = {"all": _summ(*tot), "A": _summ(*agg_grad["A"]), "B": _summ(*agg_grad["B"])}
        out[name] = res
    return out


def _margin(matched: float, shifted: float) -> float:
    """S6 has no pre-registered margin; (shifted - matched) / shifted is reported so a negligible pass is visible."""
    return (shifted - matched) / shifted if shifted > 0 else math.nan


def screen_gate(val: dict[str, Any], candidate: str = "autoencoder_best_val") -> dict[str, Any]:
    ae, tm = val[candidate], val["train_mean"]
    r, p, u = ae["representation"]["all"], ae["product"], ae["update_relevant"]
    finite = bool(ae["finite_outputs"]) and all(isinstance(v, float) and math.isfinite(v) for v in (r["cosine"], r["rel_sq_error"], p["cosine"], p["rel_fro_error"], u["rel_sq_error"]))
    sh = ae["shift_control"]
    crit = {
        "S1_finite_outputs": finite,
        "S2_representation_cosine_ge_0_90": r["cosine"] >= GATE["rep_cosine_min"],
        "S3_representation_rel_sq_error_le_0_50": r["rel_sq_error"] <= GATE["rep_rel_sq_error_max"],
        "S4_product_cosine_ge_0_90": p["cosine"] >= GATE["product_cosine_min"],
        "S5_product_rel_fro_error_le_0_50": p["rel_fro_error"] <= GATE["product_rel_fro_error_max"],
        "S6_input_dependent": sh["matched_rel_sq_error"] < sh["shifted_rel_sq_error"],
        "S7_materially_better_than_train_mean": (
            r["rel_sq_error"] <= GATE["train_mean_factor"] * tm["representation"]["all"]["rel_sq_error"]
            and u["rel_sq_error"] <= GATE["train_mean_factor"] * tm["update_relevant"]["rel_sq_error"]
        ),
    }
    crit = {k: bool(v) and (finite or k == "S1_finite_outputs") for k, v in crit.items()}
    return {
        "candidate": candidate,
        "criteria": crit,
        "values": {
            "representation_cosine": r["cosine"], "representation_rel_sq_error": r["rel_sq_error"], "product_cosine": p["cosine"],
            "product_rel_fro_error": p["rel_fro_error"], "update_rel_sq_error": u["rel_sq_error"], "shift_matched": sh["matched_rel_sq_error"],
            "shift_shifted": sh["shifted_rel_sq_error"], "shift_relative_margin": _margin(sh["matched_rel_sq_error"], sh["shifted_rel_sq_error"]),
            "train_mean_representation_rel_sq_error": tm["representation"]["all"]["rel_sq_error"],
            "train_mean_update_rel_sq_error": tm["update_relevant"]["rel_sq_error"],
        },
        "pass": all(crit.values()),
        "thresholds": GATE,
    }


def phase3_gate_mapping(p3: dict[str, Any]) -> dict[str, Any]:
    """R0/R1 reuse: read a frozen Phase-3 A5/A8 report against S1-S7 (DERIVED; nothing is retrained or re-evaluated).

    S4/S5 use the transmitted object's own effective product (state: s B A; delta: the product innovation
    s B A - s B_s A_s); the update-relevant product is the Phase-3 product innovation. S6 reuses the Phase-3 shift
    control, which pairs factor-space innovations, whereas the F6 version pairs product-space quantities.
    """
    pr = p3["val"]["predictors"]
    ae, tm = pr["autoencoder_best_val"], pr["train_mean"]
    prod = ae["product"]["state" if p3["representation"] == "adapter_state" else "innovation"]
    r, u, sh = ae["transmitted"]["all"], ae["product"]["innovation"], ae["shift_control"]
    finite = bool(ae["finite_outputs"])
    crit = {
        "S1_finite_outputs": finite,
        "S2_representation_cosine_ge_0_90": r["cosine"] >= GATE["rep_cosine_min"],
        "S3_representation_rel_sq_error_le_0_50": r["rel_sq_error"] <= GATE["rep_rel_sq_error_max"],
        "S4_product_cosine_ge_0_90": prod["cosine"] >= GATE["product_cosine_min"],
        "S5_product_rel_fro_error_le_0_50": math.sqrt(prod["rel_sq_error"]) <= GATE["product_rel_fro_error_max"],
        "S6_input_dependent": sh["innovation_rel_sq_error_matched"] < sh["innovation_rel_sq_error_shifted"],
        "S7_materially_better_than_train_mean": (
            r["rel_sq_error"] <= GATE["train_mean_factor"] * tm["transmitted"]["all"]["rel_sq_error"]
            and u["rel_sq_error"] <= GATE["train_mean_factor"] * tm["product"]["innovation"]["rel_sq_error"]
        ),
    }
    crit = {k: bool(v) and (finite or k == "S1_finite_outputs") for k, v in crit.items()}
    return {
        "representation": p3["representation"],
        "source_mode": p3["source_mode"],
        "normalization": p3["normalization"] if isinstance(p3["normalization"], str) else p3["normalization"].get("mode"),
        "phase3_label": p3["label"],
        "phase3_gate_pass": p3["gate"]["pass"],
        "criteria": crit,
        "values": {
            "representation_cosine": r["cosine"], "representation_rel_sq_error": r["rel_sq_error"], "product_cosine": prod["cosine"],
            "product_rel_fro_error": math.sqrt(prod["rel_sq_error"]), "update_rel_sq_error": u["rel_sq_error"], "update_cosine": u["cosine"],
            "shift_matched": sh["innovation_rel_sq_error_matched"], "shift_shifted": sh["innovation_rel_sq_error_shifted"],
            "shift_relative_margin": _margin(sh["innovation_rel_sq_error_matched"], sh["innovation_rel_sq_error_shifted"]),
            "train_mean_representation_rel_sq_error": tm["transmitted"]["all"]["rel_sq_error"], "train_mean_update_rel_sq_error": tm["product"]["innovation"]["rel_sq_error"],
        },
        "pass": all(crit.values()),
        "thresholds": GATE,
    }


def group_by_time(items: Sequence[ScreenItem]) -> list[list[ScreenItem]]:
    by_t: dict[int, list[ScreenItem]] = defaultdict(list)
    for it in items:
        by_t[it.time_index].append(it)
    return [by_t[t] for t in sorted(by_t)]
