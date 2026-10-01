"""Bounded real-gradient forensics (Phase 4, F5; PHASE4-FORENSIC, not recovered truth).

A short federated D1 run (the Phase-3 TGAP schedule, seeds and semantics) records, for every optimizer step of
every client round, the accumulated gradient before clipping, the gradient after clipping, and the parameter
change of the step, plus the start/end states (TGAP snapshot schema). Payloads stay outside Git.

Summary representations per client round (docs/phase4_preregistration.md, section 3):
  last_step_gradient  -- pre-clip gradient of the last optimizer step
  mean_step_gradient  -- mean over steps of the pre-clip gradients (screen candidate R4)
  optimizer_step_delta-- the per-step parameter changes (statistics over steps)
  local_epoch_delta   -- end - start
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from cg_fedllm.compression.diagnostics import lowrank_inner
from cg_fedllm.compression.gauge import module_pairs
from cg_fedllm.federated.client import StepObserver
from cg_fedllm.models.adapter import AdapterState, parse_key
from cg_fedllm.utils.io import atomic_write_json, read_json

PAPER_N = 8_388_608  # LLaMA-7B, r = 8, q/k/v/o, 32 layers
PAPER_G_SQ = 14.29


def _state_from_params(params: dict[str, torch.nn.Parameter], attr: str) -> AdapterState:
    if attr == "grad":
        return AdapterState({k: (p.grad.detach().to("cpu", torch.float32).clone() if p.grad is not None else torch.zeros(p.shape)) for k, p in params.items()})
    return AdapterState({k: p.detach().to("cpu", torch.float32).clone() for k, p in params.items()})


class GradientDumper(StepObserver):
    """Writes every step of one client round to ``root/t{t:04d}_c{cid:04d}/``."""

    def __init__(self, root: Path, t: int, cid: int) -> None:
        self.dir = Path(root) / f"t{t:04d}_c{cid:04d}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta: dict[str, Any] = {"time_index": t, "client_id": cid, "steps": []}

    @property
    def wants_step_delta(self) -> bool:
        return True

    def on_gradients(self, step: int, params, loss: float) -> None:
        _state_from_params(params, "grad").save(self.dir / f"step{step:03d}_grad.safetensors", {"role": "grad_before_clip"})
        self.meta["steps"].append({"step": step, "loss": loss})

    def on_clipped(self, step: int, params, total_norm: float | None) -> None:
        _state_from_params(params, "grad").save(self.dir / f"step{step:03d}_grad_clipped.safetensors", {"role": "grad_after_clip"})
        self.meta["steps"][-1]["pre_clip_total_norm"] = total_norm

    def on_step(self, step: int, params, before: dict[str, torch.Tensor]) -> None:
        AdapterState({k: (p.detach() - before[k]).to("cpu", torch.float32) for k, p in params.items()}).save(self.dir / f"step{step:03d}_delta.safetensors", {"role": "step_delta"})
        atomic_write_json(self.dir / "meta.json", self.meta)


def load_client_round(d: Path) -> dict[str, Any]:
    meta = read_json(d / "meta.json")
    n = len(meta["steps"])
    return {
        "meta": meta,
        "grads": [AdapterState.load(d / f"step{k:03d}_grad.safetensors") for k in range(n)],
        "clipped": [AdapterState.load(d / f"step{k:03d}_grad_clipped.safetensors") for k in range(n)],
        "deltas": [AdapterState.load(d / f"step{k:03d}_delta.safetensors") for k in range(n)],
    }


def mean_state(states: list[AdapterState]) -> AdapterState:
    keys = states[0].keys()
    return AdapterState({k: sum(st.tensors[k].double() for st in states).div(len(states)).to(torch.float32) for k in keys})


def _flat(st: AdapterState, factor: str | None = None) -> torch.Tensor:
    return torch.cat([st.tensors[k].reshape(-1).double() for k in st.keys() if factor is None or parse_key(k)[3] == factor])


def _first_order_sq(g: AdapterState, at: AdapterState, s: float) -> float:
    """||s (g_B A + B g_A)||_F^2 summed over modules, at the state ``at`` (rank <= 2r per module)."""
    tot = 0.0
    for ka, kb in module_pairs(at):
        left = torch.cat([g.tensors[kb].double(), at.tensors[kb].double()], 1) * s
        right = torch.cat([at.tensors[ka].double(), g.tensors[ka].double()], 0)
        tot += lowrank_inner(left, right, left, right)
    return tot


def _first_order_vs_exact(start: AdapterState, end: AdapterState, s: float) -> dict[str, float]:
    """First-order effective change s (dB A_s + B_s dA) vs the exact dM = s (B_e A_e - B_s A_s)."""
    sse = sig = hat = dot = 0.0
    for ka, kb in module_pairs(end):
        a_s, b_s = start.tensors[ka].double(), start.tensors[kb].double()
        a_e, b_e = end.tensors[ka].double(), end.tensors[kb].double()
        l1, r1 = torch.cat([b_e - b_s, b_s], 1) * s, torch.cat([a_s, a_e - a_s], 0)  # first order
        l2, r2 = torch.cat([b_e, -b_s], 1) * s, torch.cat([a_e, a_s], 0)  # exact
        hat += lowrank_inner(l1, r1, l1, r1)
        sig += lowrank_inner(l2, r2, l2, r2)
        dot += lowrank_inner(l1, r1, l2, r2)
        le, re_ = torch.cat([l1, -l2], 1), torch.cat([r1, r2], 0)
        sse += lowrank_inner(le, re_, le, re_)
    return {
        "cosine": dot / math.sqrt(sig * hat) if sig > 0 and hat > 0 else math.nan,
        "rel_fro_error": math.sqrt(sse / sig) if sig > 0 else math.nan,
        "exact_fro": math.sqrt(sig),
        "first_order_fro": math.sqrt(hat),
    }


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    na, nb = float(a.norm()), float(b.norm())
    return float(a @ b) / (na * nb) if na > 0 and nb > 0 else math.nan


def _family_stats(samples: list[dict[str, Any]], s: float, first_order: bool) -> dict[str, Any]:
    """samples: [{"t", "cid", "step", "state" (AdapterState), "at" (state the gradient/delta applies to)}]."""
    if not samples:
        return {}
    n_el = samples[0]["state"].num_elements()
    sq = [s_["state"].l2_sq() for s_ in samples]
    a_sq = sum(s_["state"].l2_sq("A") for s_ in samples)
    b_sq = sum(s_["state"].l2_sq("B") for s_ in samples)
    flats = [_flat(s_["state"]) for s_ in samples]
    by_t: dict[int, list[int]] = defaultdict(list)
    for i, s_ in enumerate(samples):
        by_t[s_["t"]].append(i)
    inter_client = [_cos(flats[i], flats[j]) for idx in by_t.values() for a, i in enumerate(idx) for j in idx[a + 1 :] if samples[i]["cid"] != samples[j]["cid"]]
    times = sorted(by_t)
    means = {t: torch.stack([flats[i] for i in by_t[t]]).mean(0) for t in times}
    inter_time = [_cos(means[times[a]], means[times[a + 1]]) for a in range(len(times) - 1)]
    out: dict[str, Any] = {
        "samples": len(samples),
        "elements_per_sample": n_el,
        "per_element_rms": math.sqrt(sum(sq) / (len(sq) * n_el)),
        "total_sq_norm": {"mean": sum(sq) / len(sq), "min": min(sq), "max": max(sq)},
        "A_over_B_energy": a_sq / b_sq if b_sq > 0 else math.inf,
        "inter_client_cosine_mean": sum(inter_client) / len(inter_client) if inter_client else None,
        "inter_time_cosine_of_means": inter_time,
        "max_abs": max(float(f.abs().max()) for f in flats),
        "rms_over_max_abs": math.sqrt(sum(sq) / (len(sq) * n_el)) / max(float(f.abs().max()) for f in flats),
        "paper_comparison": {
            "paper": f"||G||^2 = {PAPER_G_SQ} over n = {PAPER_N} (LLaMA-7B, r 8, q/k/v/o; alpha unknown) -> per-element RMS {math.sqrt(PAPER_G_SQ / PAPER_N):.3e}",
            "ours": f"Qwen1.5-1.8B, r 8, q/k/v/o, n = {n_el}, s = alpha/r = {s}",
            "per_element_rms_ratio_ours_over_paper": math.sqrt(sum(sq) / (len(sq) * n_el)) / math.sqrt(PAPER_G_SQ / PAPER_N),
            "gauge_invariant": False,
        },
    }
    if first_order:
        fo = [math.sqrt(_first_order_sq(s_["state"], s_["at"], s)) for s_ in samples]
        out["first_order_effective_fro"] = {"mean": sum(fo) / len(fo), "min": min(fo), "max": max(fo)}
    return out


def distribution_shape(xs: list[torch.Tensor], drop_zeros: bool, quantile_sample: int = 2_000_000, seed: int = 0) -> dict[str, Any]:
    """Value distribution of a family: std, excess kurtosis, max and the fraction beyond 0.01 over all values;
    |x| quantiles over a seeded subsample (torch.quantile has an input-size limit)."""
    x = torch.cat([t.reshape(-1).double() for t in xs])
    n_zero = int((x == 0).sum())
    if drop_zeros:
        x = x[x != 0]
    a = x.abs()
    idx = torch.randperm(a.numel(), generator=torch.Generator().manual_seed(seed))[:quantile_sample]
    probs = [0.5, 0.9, 0.99, 0.999]
    q = torch.quantile(a[idx], torch.tensor(probs, dtype=torch.float64))
    m, sd = x.mean(), x.std()
    return {
        "values": a.numel(),
        "exact_zeros": n_zero,
        "zeros_excluded": drop_zeros,
        "std": float(sd),
        "excess_kurtosis": float(((x - m) ** 4).mean() / sd**4 - 3),
        "max_abs": float(a.max()),
        "fraction_abs_gt_0_01": float((a > 0.01).double().mean()),
        "abs_quantiles": {f"p{100 * p:g}": float(v) for p, v in zip(probs, q)},
        "quantile_subsample": int(idx.numel()),
    }


def family_shapes(grad_root: Path, snapshot_root: Path, records: list[dict]) -> dict[str, Any]:
    """F5 addendum: value-distribution shape of the recorded families (gradients drop exact zeros: round-0 A)."""
    from cg_fedllm.tgap.snapshots import load_states

    fam: dict[str, list[torch.Tensor]] = {"pre_clip_step_gradient": [], "mean_step_gradient": [], "optimizer_step_delta": [], "local_epoch_delta": []}
    for r in records:
        t, cid = int(r["time_index"]), int(r["client_id"])
        cr = load_client_round(Path(grad_root) / f"t{t:04d}_c{cid:04d}")
        start, end = load_states(snapshot_root, r)
        fam["pre_clip_step_gradient"] += [_flat(x) for x in cr["grads"]]
        fam["mean_step_gradient"].append(_flat(mean_state(cr["grads"])))
        fam["optimizer_step_delta"] += [_flat(x) for x in cr["deltas"]]
        fam["local_epoch_delta"].append(_flat(end.sub(start)))
    return {name: distribution_shape(xs, drop_zeros=name.endswith("gradient")) for name, xs in fam.items()}


def gradient_statistics(grad_root: Path, snapshot_root: Path, records: list[dict], s: float) -> dict[str, Any]:
    from cg_fedllm.tgap.snapshots import load_states

    fam: dict[str, list[dict[str, Any]]] = {"last_step_gradient": [], "mean_step_gradient": [], "optimizer_step_delta": [], "local_epoch_delta": [], "pre_clip_step_gradient": []}
    clip, steps_per_round, fo_exact = [], [], []
    consecutive_step_cos = []
    for r in records:
        t, cid = int(r["time_index"]), int(r["client_id"])
        start, end = load_states(snapshot_root, r)
        cr = load_client_round(Path(grad_root) / f"t{t:04d}_c{cid:04d}")
        steps_per_round.append(len(cr["grads"]))
        clip += [st["pre_clip_total_norm"] for st in cr["meta"]["steps"]]
        state = start
        for k, (g, d) in enumerate(zip(cr["grads"], cr["deltas"])):
            fam["pre_clip_step_gradient"].append({"t": t, "cid": cid, "step": k, "state": g, "at": state})
            fam["optimizer_step_delta"].append({"t": t, "cid": cid, "step": k, "state": d, "at": state})
            if k:
                consecutive_step_cos.append(_cos(_flat(cr["deltas"][k - 1]), _flat(d)))
            state = state.add(d)
        if not state.equal(end):
            # summing the recorded step deltas in float32 must reproduce the snapshot end state (to rounding)
            if state.relative_l2_diff(end) > 1e-6:
                raise RuntimeError(f"t{t} c{cid}: recorded step deltas do not reproduce the end state")
        fam["last_step_gradient"].append({"t": t, "cid": cid, "step": len(cr["grads"]) - 1, "state": cr["grads"][-1], "at": start})
        fam["mean_step_gradient"].append({"t": t, "cid": cid, "step": -1, "state": mean_state(cr["grads"]), "at": start})
        fam["local_epoch_delta"].append({"t": t, "cid": cid, "step": -1, "state": end.sub(start), "at": start})
        fo_exact.append(_first_order_vs_exact(start, end, s))
    out = {
        "client_rounds": len(records),
        "optimizer_steps": {"total": sum(steps_per_round), "per_client_round": steps_per_round},
        "pre_clip_total_norm": {"mean": sum(clip) / len(clip), "min": min(clip), "max": max(clip), "fraction_clipped": sum(1 for c in clip if c > 1.0) / len(clip)},
        "consecutive_step_delta_cosine_mean": sum(consecutive_step_cos) / len(consecutive_step_cos) if consecutive_step_cos else None,
        "local_epoch_delta_first_order_vs_exact": {
            "cosine_mean": sum(x["cosine"] for x in fo_exact) / len(fo_exact),
            "rel_fro_error_mean": sum(x["rel_fro_error"] for x in fo_exact) / len(fo_exact),
            "rel_fro_error_max": max(x["rel_fro_error"] for x in fo_exact),
        },
        "families": {name: _family_stats(samples, s, first_order=True) for name, samples in fam.items()},
    }
    return out
