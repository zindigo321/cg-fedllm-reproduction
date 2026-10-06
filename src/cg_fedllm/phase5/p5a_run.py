"""P5-A v2 training invocations (protocol sections 7-13). Training-only: no validation payload is opened.

One training invocation k (at most 3):

1. checks the run root, the CPU preflight record and the identity of earlier invocations, then publishes
   ``ledger/invocation_<k>.json`` (also the no-clobber capability check);
2. recovers the evidence of controls left ``STARTED`` or ``ENDED`` by a process death (no retraining, no payload);
3. for each representation (R2, R3, R4) with a ``NOT_STARTED`` control: hash-verified construction of the training
   population, exact-max, range and RMS checks, and agreement with the most recent preflight record (section 5.4.1);
4. for each control in plan order: anomalous-state, ``sig = 0`` and C0 checks, the time and byte start rules, the
   start record (published *before* ``ae.to(device)``), exactly one training, final-checkpoint evaluation, C1-C4,
   and publication in the order end record, checkpoint, provenance set, record;
5. at run closure, terminal records for every control without one, and the section-11 outcomes;
6. ``ledger/summary_<k>.json``.

No control outcome prunes, reorders or changes another control. A started control is never trained again.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from cg_fedllm.compression.autoencoder import ResNetAutoEncoder, config_from_section
from cg_fedllm.compression.codecs import AutoEncoderCodec, CodecContext
from cg_fedllm.compression.layout import geometry_from_dict, get_layout
from cg_fedllm.compression.normalization import Normalizer, range_report
from cg_fedllm.phase5 import p5a
from cg_fedllm.phase5 import p5a_state as st
from cg_fedllm.phase5.p5a import AE_SEED, CURVE_EVERY, ITERATIONS, MAX_BATCH, P5AProtocolError
from cg_fedllm.phase5.p5a_artifacts import (
    TMP_SUFFIX,
    ArtifactStore,
    ByteCeilingError,
    PublicationError,
    json_bytes,
)
from cg_fedllm.phase5.p5a_inputs import (
    FROZEN,
    ControlSetup,
    FrozenInputs,
    Population,
    RepresentationSetup,
    evidence,
    prepare_representation,
)
from cg_fedllm.phase5.p5a_metrics import (
    ae_decision,
    c0_feasibility,
    predictor_metrics,
    subset_mean,
    tanh_range_ceiling,
    zero_prediction,
)
from cg_fedllm.phase5.p5a_preflight import (
    PREFLIGHT_SCHEMA,
    cpu_environment,
    existing_preflights,
    latest_preflight,
    validate_against_preflight,
)
from cg_fedllm.utils.seeding import derive_seed, numpy_rng

PRODUCTION_PROFILE = "PRODUCTION"
IDENTITY_KEYS = (
    "execution_commit",
    "config_sha256",
    "launch_record",
)  # must match invocation 1 (section 13.1)


@dataclass(frozen=True)
class Profile:
    """Production uses :data:`PRODUCTION`. Synthetic tests may shorten training and supply synthetic frozen inputs;
    such a profile is never accepted by the production entry points and its records are marked as not a result."""

    name: str
    iterations: int
    frozen: Mapping[str, FrozenInputs]
    require_cuda: bool

    @property
    def production(self) -> bool:
        return self.name == PRODUCTION_PROFILE

    def __post_init__(self) -> None:
        if self.name == PRODUCTION_PROFILE and (
            self.iterations != ITERATIONS or dict(self.frozen) != dict(FROZEN) or not self.require_cuda
        ):
            raise P5AProtocolError("the production profile cannot be altered")


PRODUCTION = Profile(PRODUCTION_PROFILE, ITERATIONS, FROZEN, True)


def synthetic_profile(frozen: Mapping[str, FrozenInputs], iterations: int) -> Profile:
    """Test-only profile: few iterations on synthetic inputs, CPU allowed. Not reachable from the CLI."""
    if not 1 <= iterations <= 50:
        raise ValueError("synthetic profiles run 1-50 iterations")
    return Profile("SYNTHETIC-TEST", iterations, dict(frozen), False)


def _utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


class TrainingStop(Exception):
    """A started training ended without a finished evaluation; ``status`` is its section-10 status."""

    def __init__(
        self, status: str, detail: str, curve: list, iterations_done: int, operator: bool = False
    ) -> None:
        super().__init__(detail)
        self.status, self.detail, self.curve, self.iterations_done = status, detail, curve, iterations_done
        self.operator_interrupt = operator


class GpuClock:
    """Monotonic per-training clock with the 1,800 s hard limit; ``sync`` is a CUDA synchronize."""

    def __init__(self, clock: Callable[[], float], sync: Callable[[], None], limit_s: float) -> None:
        self.clock, self.sync, self.limit = clock, sync, limit_s
        self.t0: float | None = None
        self.elapsed: float | None = None

    def start(self) -> None:
        self.sync()
        self.t0 = self.clock()

    def now(self) -> float:
        return self.clock() - self.t0

    def over(self) -> bool:
        return self.now() >= self.limit

    def stop(self) -> float | None:
        if self.t0 is None:
            return None
        try:
            self.sync()
        finally:
            self.elapsed = self.clock() - self.t0
        return self.elapsed


def batch_schedule(k: int, iterations: int, seed: int = AE_SEED) -> list[list[int]]:
    """Section 8.3: ``bs = min(4, k)``; the first ``bs`` entries of a permutation from ``numpy_rng(seed,
    "ae_batches")``, with a new permutation when fewer than ``bs`` remain (``train_autoencoder``, ``cbfedab``)."""
    rng = numpy_rng(seed, "ae_batches")
    bs = min(MAX_BATCH, k)
    perm: list[int] = []
    out = []
    for _ in range(iterations):
        if len(perm) < bs:
            perm = [int(j) for j in rng.permutation(k)]
        out.append(perm[:bs])
        perm = perm[bs:]
    return out


def build_ae(cfg_section: Any) -> ResNetAutoEncoder:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(derive_seed(AE_SEED, "ae_init"))
        return ResNetAutoEncoder(config_from_section(cfg_section))


def train_and_evaluate(
    ae: ResNetAutoEncoder,
    xs: Sequence[torch.Tensor],
    norm: Normalizer,
    *,
    iterations: int,
    device: torch.device,
    gpu: GpuClock,
) -> dict[str, Any]:
    """Train on the unique selected snapshots and evaluate the final checkpoint (sections 8.3-8.4).

    The GPU clock starts immediately before ``ae.to(device)`` and stops after the evaluation; it is stopped on every
    exit path. Raises :class:`TrainingStop` for a non-finite loss, the time limit, a crash or an operator interrupt.
    """
    stack = torch.stack([norm.normalize(x) for x in xs])  # [k, 1, d, W], normalised space, CPU
    schedule = batch_schedule(len(xs), iterations)
    curve: list[dict[str, Any]] = []
    it = 0
    gpu.start()
    try:
        ae.to(device)
        opt = torch.optim.Adam(ae.parameters(), lr=2e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
        for it in range(1, iterations + 1):
            if gpu.over():
                raise TrainingStop(
                    p5a.INCOMPLETE_RESOURCE, f"1,800 s limit before iteration {it}", curve, it - 1
                )
            ae.train()
            x = stack[schedule[it - 1]].to(device)
            x_hat, _ = ae(x)
            loss = F.mse_loss(x_hat, x)
            if not torch.isfinite(loss):
                raise TrainingStop(
                    p5a.INCOMPLETE_NON_FINITE, f"non-finite loss at iteration {it}", curve, it - 1
                )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if it == 1 or it % CURVE_EVERY == 0 or it == iterations:
                curve.append({"iteration": it, "train_mse_batch": float(loss.item())})
        if gpu.over():
            raise TrainingStop(p5a.INCOMPLETE_RESOURCE, "1,800 s limit before evaluation", curve, iterations)
        # section 8.4: final checkpoint, eval mode (BatchNorm running statistics), no gradients, one snapshot each
        ae.eval()
        codec = AutoEncoderCodec(ae, device=device, normalizer=norm)
        recons = []
        for i, x in enumerate(xs):
            ctx = CodecContext(0, i, 0)
            recons.append(codec.decode(codec.encode(x, ctx), ctx))
        state = {k: v.detach().to("cpu").clone() for k, v in ae.state_dict().items()}
    except TrainingStop:
        gpu.stop()
        raise
    except KeyboardInterrupt as exc:  # operator interrupt: the training counts; the invocation stops
        gpu.stop()
        raise TrainingStop(p5a.INCOMPLETE_INTERRUPTED, "KeyboardInterrupt", curve, it, operator=True) from exc
    except Exception as exc:  # crash, OOM, deterministic-backend refusal: the training counts, never retried
        gpu.stop()
        raise TrainingStop(p5a.INCOMPLETE_INTERRUPTED, f"{type(exc).__name__}: {exc}", curve, it) from exc
    gpu.stop()
    if gpu.elapsed >= gpu.limit:
        raise TrainingStop(
            p5a.INCOMPLETE_RESOURCE, "per-training limit reached during evaluation", curve, iterations
        )
    return {"curve": curve, "reconstructions": recons, "state": state, "iterations_done": iterations}


def checkpoint_bytes(
    state: Mapping[str, torch.Tensor], ae: ResNetAutoEncoder, metadata: Mapping[str, Any]
) -> bytes:
    """The ``save_autoencoder`` format (``cg_fedllm.resnet_ae/v1``, readable by ``load_autoencoder``) as bytes."""
    from safetensors.torch import save as st_save

    meta = {
        "format": "cg_fedllm.resnet_ae/v1",
        "config": json.dumps(dataclasses.asdict(ae.cfg)),
        "metadata": json.dumps(dict(metadata), sort_keys=True, allow_nan=False),
    }
    return st_save({k: v.detach().to("cpu").contiguous() for k, v in state.items()}, metadata=meta)


def control_c0(setup: ControlSetup, norm: Normalizer) -> dict[str, Any]:
    """Section 8.2: the control's own gate applied to the Tanh-range ceiling, before training."""
    ceiling = predictor_metrics(setup.xs, [tanh_range_ceiling(x, norm) for x in setup.xs])
    mean = None
    if setup.control == "fixed_subset4":
        m = subset_mean(setup.xs)
        mean = predictor_metrics(setup.xs, [m] * len(setup.xs))
    return {
        "tanh_range_ceiling": ceiling,
        "subset_mean": mean,
        **c0_feasibility(setup.control, ceiling, mean),
    }


def evaluate_final(setup: ControlSetup, norm: Normalizer, recons: Sequence[torch.Tensor]) -> dict[str, Any]:
    """Sections 8.1 and 9: AE plus reporting-only predictors; section 10 decision on the AE only."""
    xs = setup.xs
    finite = all(bool(torch.isfinite(r).all()) for r in recons)
    ae = predictor_metrics(xs, list(recons))
    preds: dict[str, Any] = {
        "autoencoder": ae,
        "zero": predictor_metrics(xs, [zero_prediction(x) for x in xs]),
        "tanh_range_ceiling": predictor_metrics(xs, [tanh_range_ceiling(x, norm) for x in xs]),
    }
    mean = None
    if setup.control == "fixed_subset4":
        m = subset_mean(xs)
        mean = preds["subset_mean"] = predictor_metrics(xs, [m] * len(xs))
    return {"predictors": preds, "decision": ae_decision(setup.control, finite, ae, mean)}


def strict(obj: Any, path: str = "", found: list[str] | None = None) -> Any:
    """Replace any remaining non-finite float by ``null`` (records are strict JSON, section 9.3); metric fields are
    already ``null`` with reasons, so this only guards non-metric fields and lists them in ``found``."""
    import math

    if isinstance(obj, float) and not math.isfinite(obj):
        if found is not None:
            found.append(path)
        return None
    if isinstance(obj, dict):
        return {k: strict(v, f"{path}/{k}", found) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [strict(v, f"{path}/{i}", found) for i, v in enumerate(obj)]
    return obj


def record_bytes(doc: Mapping[str, Any]) -> bytes:
    found: list[str] = []
    clean = strict(dict(doc), "", found)
    if found:
        clean["non_finite_fields_written_as_null"] = found
    return json_bytes(clean)


@dataclass
class RunContext:
    """Invocation-wide facts written into every record. ``identity`` holds ``execution_commit``,
    ``config_sha256``, ``launch_record`` (a citation of the human launch record, never proof of authorisation) and
    ``command``."""

    profile: Profile
    store: ArtifactStore
    ae_section: Any
    device: torch.device
    identity: dict[str, Any]
    clock: Callable[[], float] = time.monotonic
    sync: Callable[[], None] = lambda: None
    trainer: Callable[..., dict[str, Any]] = train_and_evaluate
    run_files: Mapping[str, bytes] = field(default_factory=dict)  # the provenance set of section 13.2
    invocation: int = 0
    setup_wall_s: float = 0.0
    measured: dict[tuple[str, str], float | None] = field(default_factory=dict)
    # v2.1 A5 P1: re-reads the CPU preprocessing environment just before the inputs are rebuilt (production only)
    cpu_env: Callable[[], dict[str, Any]] | None = None
    entry: dict[str, Any] | None = None  # v2.1 A2: how invocation 1 found the root (recorded in its record)

    def reserved(self, exclude: Sequence[str] = (), checkpoint_for: tuple[str, str] | None = None) -> int:
        return st.reserved_caps(
            self.store,
            invocation=max(self.invocation, 1),
            exclude=set(exclude),
            checkpoint_for=checkpoint_for,
        )

    def publish(self, rel: str, data: bytes, checkpoint_for: tuple[str, str] | None = None) -> dict[str, Any]:
        return self.store.publish(rel, data, reserved_other=self.reserved([rel], checkpoint_for))


def _header(ctx: RunContext) -> dict[str, Any]:
    return {
        "label": p5a.P5A_LABEL,
        "profile": ctx.profile.name,
        "is_result": ctx.profile.production,
        "protocol": p5a.protocol_identity(),
        "invocation": ctx.invocation,
        "identity": ctx.identity,
    }


def _base_record(ctx: RunContext, rs: RepresentationSetup, cs: ControlSetup) -> dict[str, Any]:
    return {
        **_header(ctx),
        "representation": rs.rid,
        "designation": p5a.DESIGNATION[rs.rid],
        "kind": rs.frozen.kind,
        "control": cs.control,
        "plan_position": p5a.PLAN.index((rs.rid, cs.control)) + 1,
        "input": {
            "root": rs.frozen.root,
            "index_sha256": rs.population["index_sha256"],
            "n_train": rs.population["n_train"],
        },
        "selected_records": cs.selected,
        "normalization": {"normalizer": rs.normalizer.to_dict(), "scale_check": rs.scale_check},
        "rms_check": rs.rms_check,
        "seeds": {
            "ae_seed": AE_SEED,
            "ae_init": derive_seed(AE_SEED, "ae_init"),
            "ae_batches": derive_seed(AE_SEED, "ae_batches"),
        },
        "iterations_planned": ctx.profile.iterations,
        "thresholds": p5a.THRESHOLDS[cs.control],
        "standard": p5a.STANDARDS[cs.control],
    }


def _range(rs: RepresentationSetup, cs: ControlSetup) -> dict[str, Any] | None:
    """Section 5.5: normalised-range statistics, reporting only (never a criterion)."""
    try:
        geom = geometry_from_dict(dict(rs.frozen.geometry))
        return range_report(cs.xs, list(range(len(cs.xs))), rs.normalizer, get_layout(p5a.LAYOUT_ID), geom)
    except ValueError:
        return None


class InvocationStop(Exception):
    """The invocation stops (section 13.5). ``closes`` is True for a byte- or time-ceiling stop (run closure);
    ``cause`` is the v2.1 A4.1 ``resource_stop_cause`` of a byte-policy refusal (never a control status)."""

    def __init__(self, reason: str, detail: str, closes: bool, cause: str | None = None) -> None:
        super().__init__(detail)
        self.reason, self.detail, self.closes, self.cause = reason, detail, closes, cause

    def to_dict(self) -> dict[str, Any]:
        return {"reason": self.reason, "detail": self.detail, "resource_stop_cause": self.cause}


@dataclass
class Outcome:
    """What one control attempt produced in this invocation."""

    status: str | None  # terminal status published, or None (still NOT_STARTED)
    pre_start: dict[str, Any] | None = None  # a pre-start status kept for the summary (section 12)
    criteria: dict[str, Any] | None = None


def _pre_start(status: str, detail: str, **extra: Any) -> Outcome:
    return Outcome(None, {"status": status, "detail": detail, **extra})


def _publish_failure(ctx: RunContext, rid: str, control: str, doc: dict[str, Any]) -> bool:
    """Best effort: the failure record has its own reserved cap; it never deletes anything."""
    try:
        ctx.publish(p5a.failure_path(rid, control), record_bytes(doc))
        return True
    except PublicationError:
        return False


def run_control(ctx: RunContext, rs: RepresentationSetup, cs: ControlSetup) -> Outcome:
    """One NOT_STARTED control through section 10. Pre-start outcomes keep it NOT_STARTED (section 12)."""
    rid, control = rs.rid, cs.control
    anomalous = st.anomalous_files(ctx.store, rid, control)
    if anomalous:
        return _pre_start(p5a.INCOMPLETE_PROVENANCE, "anomalous state: files exist for a NOT_STARTED control",
                          files=anomalous)  # fmt: skip
    if cs.provenance_error:
        return _pre_start(p5a.INCOMPLETE_PROVENANCE, cs.provenance_error)
    c0 = control_c0(cs, rs.normalizer)
    c0_rec = {"pass": c0["pass"], "criteria": c0["criteria"], "required": c0["required"]}
    if not c0["pass"]:
        return _pre_start(
            p5a.NOT_INFORMATIVE, "C0 failed: the Tanh-range ceiling does not meet the gate", c0=c0_rec
        )
    res = st.resource_state(ctx.store)
    if not res["may_start"]:
        raise InvocationStop(
            "time_ceiling", "no full 1,800 s allocation or training left (section 12)", closes=True
        )
    fits = ctx.store.check(p5a.start_path(rid, control), b"", ctx.reserved([p5a.start_path(rid, control)],
                                                                           checkpoint_for=(rid, control)))  # fmt: skip
    if not fits["fits"]:
        raise InvocationStop("byte_ceiling", f"reservation refused the start of {rid} {control}: {fits}",
                             closes=True, cause=fits["resource_stop_cause"])  # fmt: skip

    rec = _base_record(ctx, rs, cs)
    rec["c0"] = c0_rec
    ae = build_ae(ctx.ae_section)
    ae.check_input_hw(*cs.xs[0].shape[-2:])
    start = {**_header(ctx), "representation": rid, "control": control, "started_utc": _utc(),
             "resource_state_before": res, "retained_bytes_before_this_file": ctx.store.footprint(),
             "rule": "a started training counts and is never retried (section 12)"}  # fmt: skip
    try:
        ctx.publish(p5a.start_path(rid, control), json_bytes(start), checkpoint_for=(rid, control))
    except ByteCeilingError as exc:  # nothing started: a byte-ceiling stop before the start (section 13.3)
        cause = exc.accounting.get("resource_stop_cause")
        raise InvocationStop("byte_ceiling", str(exc), closes=True, cause=cause) from exc
    except PublicationError as exc:  # nothing started: the control stays NOT_STARTED
        raise InvocationStop("publication_failure", f"start record: {exc}", closes=False) from exc
    # ---- START BOUNDARY: from here on the training counts ----
    gpu = GpuClock(ctx.clock, ctx.sync, p5a.TRAINING_LIMIT_S)
    operator = False
    try:
        out = ctx.trainer(
            ae, cs.xs, rs.normalizer, iterations=ctx.profile.iterations, device=ctx.device, gpu=gpu
        )
        status, curve, done, detail = None, out["curve"], out["iterations_done"], None
    except TrainingStop as stop:
        out, status, curve, done, detail = None, stop.status, stop.curve, stop.iterations_done, stop.detail
        operator = stop.operator_interrupt
    timing = st.timing(gpu.elapsed)
    ctx.measured[(rid, control)] = gpu.elapsed
    rec.update(trained=True, iterations_done=done, curve=curve, timing=timing)
    if detail:
        rec["detail"] = detail
    end = {**_header(ctx), "representation": rid, "control": control, "ended_utc": _utc(),
           "training_end": "completed" if out is not None else status, "iterations_done": done,
           "timing": timing, "retained_bytes_before_this_file": ctx.store.footprint()}  # fmt: skip
    files: list[tuple[str, bytes, bool]] = [(p5a.end_path(rid, control), json_bytes(end), False)]
    if out is not None:
        ev = evaluate_final(cs, rs.normalizer, out["reconstructions"])
        status = p5a.FIT_PASS if ev["decision"]["pass"] else p5a.FIT_FAIL
        rec["evaluation"] = ev
        rec["latent_shape"] = list(ae.latent_shape(*cs.xs[0].shape[-2:]))
        rec["normalized_range"] = _range(rs, cs)
        meta = {"representation": rid, "control": control, "checkpoint": "final", "iteration": done,
                "normalization": rs.normalizer.to_dict(), "label": p5a.P5A_LABEL, "profile": ctx.profile.name,
                "protocol_commit": p5a.P5A_PROTOCOL_COMMIT}  # fmt: skip
        files.append((p5a.checkpoint_path(rid, control), checkpoint_bytes(out["state"], ae, meta), True))
    d = p5a.control_dir(rid, control)
    files += [(f"{d}/{name}", data, False) for name, data in ctx.run_files.items()]
    rec["status"] = status
    criteria = rec.get("evaluation", {}).get("decision")
    outcome = _publish_control(ctx, rid, control, rec, files)
    if operator and outcome.status is not None:
        raise InvocationStop("operator_interrupt", "operator interrupt during training", closes=False)
    return Outcome(outcome.status, criteria=criteria) if outcome.status else outcome


def _publish_control(
    ctx: RunContext, rid: str, control: str, rec: dict[str, Any], files: list[tuple[str, bytes, bool]]
) -> Outcome:
    """Section 13.4 order: end record, checkpoint, provenance set, record. A later failure keeps every finalized
    file, yields ``INCOMPLETE (PUBLICATION)`` with a failure record, and stops the invocation."""
    done: list[dict[str, Any]] = []
    for rel, data, is_ckpt in files:
        try:
            done.append(ctx.publish(rel, data, checkpoint_for=(rid, control) if is_ckpt else None))
        except PublicationError as exc:
            return _publication_failed(ctx, rid, control, rec, done, rel, data, exc)
    rec["retained_bytes_before_this_file"] = ctx.store.footprint()
    rec["resource_state"] = st.resource_state(ctx.store)
    rec["published_files"] = done
    rel = p5a.record_path(rid, control)
    try:
        ctx.publish(rel, record_bytes(rec))
    except PublicationError as exc:
        return _publication_failed(ctx, rid, control, rec, done, rel, b"", exc)
    return Outcome(rec["status"])


def started_status(training_status: str | None) -> str:
    """v2.1 A4.2 for a started control whose file could not be finalized: a training-stop status (S1) is kept;
    a completed training and evaluation (``None`` or a gate status) becomes ``INCOMPLETE (PUBLICATION)`` (S3),
    whatever the cause. A byte-policy refusal never makes a started control ``INCOMPLETE (RESOURCE STOP)``."""
    if training_status in p5a.TRAINING_STOP_STATUSES:
        return training_status
    return p5a.INCOMPLETE_PUBLICATION


def _publication_failed(
    ctx: RunContext,
    rid: str,
    control: str,
    rec: dict[str, Any],
    done: list,
    rel: str,
    data: bytes,
    exc: Exception,
) -> Outcome:
    """A file of a started control could not be finalized (v2.1 A4.3). Finalized files are kept (never deleted).
    The control's status follows v2.1 A4.2 (:func:`started_status`); the reason for stopping is recorded separately:
    a byte-policy refusal (with its ``resource_stop_cause``) is a byte-ceiling stop that closes the run, any other
    failure stops the invocation. A computed gate result is kept as reported-only evidence."""
    ceiling = isinstance(exc, ByteCeilingError)
    cause = exc.accounting.get("resource_stop_cause") if ceiling else None
    gate = rec.get("status") if rec.get("status") in (p5a.FIT_PASS, p5a.FIT_FAIL) else None
    status = started_status(rec.get("status"))
    fail = {**_header(ctx), "representation": rid, "control": control, "status": status,
            "training_end": "completed" if gate else rec.get("status"), "reported_only_gate_result": gate,
            "stop_reason": "byte_ceiling" if ceiling else "publication_failure", "resource_stop_cause": cause,
            "finalized_before_failure": done, "failed_file": {"path": rel, "bytes": len(data)},
            "error": f"{type(exc).__name__}: {exc}", "accounting": getattr(exc, "accounting", None),
            "retained_bytes_before_this_file": ctx.store.footprint()}  # fmt: skip
    _publish_failure(ctx, rid, control, fail)
    if rel != p5a.record_path(rid, control) and not ctx.store.exists(p5a.record_path(rid, control)):
        slim = {**rec, "status": status, "reported_only_gate_result": gate, "resource_stop_cause": cause,
                "publication_failure": fail["failed_file"], "finalized_before_failure": done,
                "retained_bytes_before_this_file": ctx.store.footprint()}  # fmt: skip
        try:
            ctx.publish(p5a.record_path(rid, control), record_bytes(slim))
        except PublicationError:
            pass  # stays ENDED (or STARTED); a later invocation recovers it (section 12, v2.1 A4.4)
    raise InvocationStop("byte_ceiling" if ceiling else "publication_failure",
                         f"{rid} {control}: {rel}: {exc}", closes=ceiling, cause=cause)  # fmt: skip


# ---- invocations, recovery and closure (sections 12-13) -------------------------------------------------------
def _invocations(store: ArtifactStore) -> list[int]:
    return [k for k in range(1, p5a.MAX_INVOCATIONS + 1) if store.exists(p5a.invocation_path(k))]


def is_closed(store: ArtifactStore) -> bool:
    return any(
        (store.read_json(p5a.summary_path(k)) or {}).get("closed") for k in range(1, p5a.MAX_INVOCATIONS + 1)
    )


_TOKEN = r"[0-9a-f]{12}"  # ArtifactStore.token
_FIRST_RECORD_TMP = re.compile(rf"^ledger/\.invocation_1\.json\.{_TOKEN}{re.escape(TMP_SUFFIX)}$")
_PREFLIGHT_TMP = re.compile(rf"^preflight/\.preflight_[1-3]\.json\.{_TOKEN}{re.escape(TMP_SUFFIX)}$")


def _strict_json(path: Path) -> Any:
    def reject(token: str) -> None:
        raise ValueError(f"non-finite JSON constant {token}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


def first_slot_entry(store: ArtifactStore) -> dict[str, Any]:
    """v2.1 A2: may a root without a finalized ``ledger/invocation_1.json`` take the invocation-1 slot?

    Allowed: consecutive, well-formed preflight records; the directories ``preflight/`` and ``ledger/`` (possibly
    empty); temporaries of an interrupted invocation-1 or preflight publication, recognized by name only. Refused
    (``P5AProtocolError``, nothing published, renamed or deleted): any other finalized file, any control directory,
    any other temporary (e.g. a start-record temporary: an uncertain start), any unknown path, a malformed preflight
    record. Returns the entry evidence recorded in the invocation-1 record."""
    done = existing_preflights(store)
    problems: list[str] = []
    if not done or done != list(range(1, len(done) + 1)):
        problems.append(f"preflight records are missing or not consecutive: {done}")
    for k in done:
        try:
            doc = _strict_json(store.path(p5a.preflight_path(k)))
        except (OSError, ValueError) as exc:
            problems.append(f"{p5a.preflight_path(k)} is not strict JSON: {exc}")
            continue
        if not (isinstance(doc, dict) and doc.get("schema") == PREFLIGHT_SCHEMA and doc.get("preflight") == k
                and doc.get("trainings_started") == 0):  # fmt: skip
            problems.append(f"{p5a.preflight_path(k)} is not a P5-A v2 preflight record of number {k}")
    records = {p5a.preflight_path(k) for k in done}
    temporaries = []
    for p in sorted(store.root.rglob("*")):
        rel = p.relative_to(store.root).as_posix()
        if p.is_dir() and not p.is_symlink():
            if rel not in ("preflight", "ledger"):
                kind = "control directory (training may have started)" if "/" not in rel else "directory"
                problems.append(f"{kind}: {rel}")
        elif p.is_file() and not p.is_symlink():
            if rel in records:
                continue
            if _FIRST_RECORD_TMP.match(rel) or _PREFLIGHT_TMP.match(rel):
                temporaries.append({"path": rel, "bytes": p.stat().st_size})
            elif rel.endswith(TMP_SUFFIX):
                problems.append(f"temporary of a training-stage artifact (uncertain start or later): {rel}")
            else:
                problems.append(f"finalized or unknown file: {rel}")
        else:
            problems.append(f"unknown path type: {rel}")
    if problems:
        raise P5AProtocolError(
            "the invocation-1 slot is refused (v2.1 A2: not a fresh run nor an abandoned first publication): "
            + "; ".join(problems[:10])
        )
    ledger = store.root / "ledger"
    return {
        "state": "abandoned_first_publication" if ledger.is_dir() else "fresh",
        "ledger_existed": ledger.is_dir(),
        "retained_temporaries": temporaries,
        "rule": "v2.1 A2: no finalized training-stage file and no evidence that training started",
    }


def check_same_run(ctx: RunContext) -> None:
    """Section 13.1: a later invocation (or the closure-only step) must match invocation 1's provenance."""
    first = ctx.store.read_json(p5a.invocation_path(1)) or {}
    old = first.get("identity") or {}
    if first.get("protocol") != p5a.protocol_identity():
        raise P5AProtocolError(
            "invocation 1 names a different protocol identity (v2 commit and v2.1 addendum)"
        )
    if old.get("config_sha256") != ctx.identity.get("config_sha256"):
        raise P5AProtocolError("invocation 1 names a different configuration")
    same_commit = old.get("execution_commit") == ctx.identity.get("execution_commit")
    same_launch = old.get("launch_record") == ctx.identity.get("launch_record")
    if same_commit and not same_launch:
        raise P5AProtocolError("invocation 1 cites a different launch record (section 13.1)")
    if not same_commit and same_launch:
        raise P5AProtocolError("a different implementation commit needs a cited launch-record update (13.1)")


def next_invocation(ctx: RunContext) -> int:
    """Section 13.1 entry rules; raises before anything is published."""
    store = ctx.store
    if store.root.name == p5a.P5A_V1_RUN_NAME:
        raise P5AProtocolError("the v1 run root is never used (v2 section 13.1)")
    if ctx.profile.production and store.root.name != p5a.P5A_RUN_NAME:
        raise P5AProtocolError(f"the production run root is {p5a.P5A_RUN_NAME}/")
    if not store.root.exists() or not existing_preflights(store):
        raise P5AProtocolError("no preflight record: run the CPU preflight first (section 5.4.1)")
    if is_closed(store):
        raise P5AProtocolError("the run is closed; no invocation follows closure (section 13.5)")
    done = _invocations(store)
    if done != list(range(1, len(done) + 1)):
        raise P5AProtocolError(f"invocation records are not consecutive: {done}")
    if len(done) >= p5a.MAX_INVOCATIONS:
        raise P5AProtocolError(
            "at most 3 training invocations; use the closure-only step if summary_3 is missing"
        )
    if not done:  # v2.1 A2 replaces the v2 section-13.1 directory rule
        ctx.entry = first_slot_entry(store)
        return 1
    check_same_run(ctx)
    return len(done) + 1


def _terminal(
    ctx: RunContext, rid: str, control: str, status: str, reason: str, **extra: Any
) -> dict[str, Any]:
    return {**_header(ctx), "representation": rid, "designation": p5a.DESIGNATION[rid], "control": control,
            "plan_position": p5a.PLAN.index((rid, control)) + 1, "status": status, "terminal_reason": reason,
            "thresholds": p5a.THRESHOLDS[control], "standard": p5a.STANDARDS[control], **extra,
            "retained_bytes_before_this_file": ctx.store.footprint()}  # fmt: skip


def _checkpoint_ref(store: ArtifactStore, rid: str, control: str) -> dict[str, Any] | None:
    rel = p5a.checkpoint_path(rid, control)
    return {"path": rel, "bytes": store.size(rel)} if store.exists(rel) else None


def ended_status(training_end: Any) -> str:
    """v2.1 A4.4: the status of an ``ENDED`` control without a record, from its finalized end record: a completed
    training -> ``INCOMPLETE (PUBLICATION)``; a recorded training-stop status is kept; an end record written by a
    section-12 recovery (``unknown_process_death``) or an unreadable one -> ``INCOMPLETE (INTERRUPTED)``."""
    if training_end == "completed":
        return p5a.INCOMPLETE_PUBLICATION
    if training_end in p5a.TRAINING_STOP_STATUSES:
        return training_end
    return p5a.INCOMPLETE_INTERRUPTED


def recover(ctx: RunContext) -> list[dict[str, Any]]:
    """Section 12 evidence recovery: no retraining, no re-evaluation, no payload read. ``STARTED`` -> end record
    (unknown timing, charged 1,800 s) and ``INCOMPLETE (INTERRUPTED)``; ``ENDED`` -> :func:`ended_status`
    (v2.1 A4.4). These statuses also apply at a resource closure: a trained control is never relabelled."""
    out = []
    for rid, control in p5a.PLAN:
        state = st.control_state(ctx.store, rid, control)
        if state == st.STARTED:
            timing = st.timing(ctx.measured.get((rid, control)))  # measured only if this process observed it
            end = {**_header(ctx), "representation": rid, "control": control, "ended_utc": _utc(),
                   "training_end": "unknown_process_death", "iterations_done": None, "timing": timing,
                   "recovered_by_invocation": ctx.invocation,
                   "retained_bytes_before_this_file": ctx.store.footprint()}  # fmt: skip
            ctx.publish(p5a.end_path(rid, control), record_bytes(end))
            status = p5a.INCOMPLETE_INTERRUPTED
            rec = _terminal(ctx, rid, control, status, "recovered: process death after the start record",
                            trained=True, timing=timing)  # fmt: skip
            ctx.publish(p5a.record_path(rid, control), record_bytes(rec))
            out.append({"representation": rid, "control": control, "from": state, "status": status})
        elif state == st.ENDED:
            end = ctx.store.read_json(p5a.end_path(rid, control)) or {}
            status = ended_status(end.get("training_end"))
            fail = (
                ctx.store.read_json(p5a.failure_path(rid, control)) or {}
            )  # cited evidence only (v2.1 A4.4)
            rec = _terminal(ctx, rid, control, status, "recovered: no control record after the end record",
                            trained=True, timing=end.get("timing"), training_end=end.get("training_end"),
                            iterations_done=end.get("iterations_done"),
                            checkpoint=_checkpoint_ref(ctx.store, rid, control),
                            reported_only_gate_result=fail.get("reported_only_gate_result"),
                            resource_stop_cause=fail.get("resource_stop_cause"))  # fmt: skip
            ctx.publish(p5a.record_path(rid, control), record_bytes(rec))
            out.append({"representation": rid, "control": control, "from": state, "status": status})
    return out


def _earlier_pre_start(store: ArtifactStore, invocation: int) -> dict[tuple[str, str], dict[str, Any]]:
    pre: dict[tuple[str, str], dict[str, Any]] = {}
    for k in range(1, invocation):
        for row in (store.read_json(p5a.summary_path(k)) or {}).get("controls", []):
            if row.get("pre_start"):
                pre[(row["representation"], row["control"])] = row["pre_start"]
    return pre


def close_run(
    ctx: RunContext,
    resource_closure: bool,
    pre: Mapping[tuple[str, str], dict[str, Any]],
    stop: InvocationStop | None = None,
) -> list[dict[str, Any]]:
    """Section 13.5: a terminal record for every control without one. Started controls get their section-12
    recovery status; the closure statuses (resource stop, last pre-start status, not started) apply only to
    controls that never started (sections 7 and 13.4: "not trained"; v2.1 A4.2). A resource closure cites the
    stop reason and its ``resource_stop_cause``. An occupied path is left untouched."""
    issues = []
    recover(ctx)
    for rid, control in p5a.PLAN:
        if st.control_state(ctx.store, rid, control) == st.RECORDED:
            continue
        last = pre.get((rid, control))
        if resource_closure:
            status, reason = p5a.INCOMPLETE_RESOURCE, "a resource ceiling closed the run before this control"
        elif last:
            status, reason = last["status"], f"last recorded pre-start status: {last.get('detail')}"
        else:
            status, reason = p5a.INCOMPLETE_NOT_STARTED, "the run was closed before this control started"
        extra = {"run_closed_by": stop.to_dict()} if resource_closure and stop else {}
        rec = _terminal(ctx, rid, control, status, reason, trained=False, pre_start=last, **extra)
        try:
            ctx.publish(p5a.record_path(rid, control), record_bytes(rec))
        except PublicationError as exc:
            path = p5a.record_path(rid, control)
            issues.append({"representation": rid, "control": control, "status": status, "path": path,
                           "occupied_bytes": ctx.store.size(path), "error": str(exc)})  # fmt: skip
    return issues


def outcomes(store: ArtifactStore, issues: Sequence[Mapping[str, Any]] = ()) -> list[dict[str, Any]]:
    """Section 11.2/11.4 from the terminal records, primary representations first; never aggregated."""
    fallback = {(i["representation"], i["control"]): i["status"] for i in issues}
    rows = []
    for rid in sorted(p5a.REPRESENTATIONS, key=lambda r: (p5a.DESIGNATION[r] != "primary", r)):
        statuses, criteria, reported = {}, {}, {}
        for c in p5a.CONTROLS:
            rec = store.read_json(p5a.record_path(rid, c)) or {}
            statuses[c] = rec.get("status") or fallback.get((rid, c)) or p5a.INCOMPLETE_NOT_STARTED
            if statuses[c] in (p5a.FIT_PASS, p5a.FIT_FAIL):
                criteria[c] = (rec.get("evaluation") or {}).get("decision")
            if rec.get("reported_only_gate_result"):  # v2.1 A4.5: shown, never a gate result
                reported[c] = rec["reported_only_gate_result"]
        row = p5a.outcome_record(rid, statuses, criteria)
        row["reported_only_gate_results"] = {
            "controls": reported,
            "note": "computed before a publication failure; not gate results and not used by the outcome",
        }
        rows.append(row)
    return rows


def _summary(ctx: RunContext, rows: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    return {
        **_header(ctx),
        "controls": rows,
        "resource_state": st.resource_state(ctx.store),
        "setup_wall_s": ctx.setup_wall_s,
        "retained_bytes_before_this_file": ctx.store.footprint(),
        "note": "representations are never aggregated; there is no combined P5-A pass (section 11.4)",
        **extra,
    }


def run_invocation(
    ctx: RunContext,
    load_population: Callable[[FrozenInputs], Population],
) -> dict[str, Any]:
    """One training invocation (section 13). Returns its summary (also published as ``summary_<k>``)."""
    k = next_invocation(ctx)
    ctx.invocation = k
    preflight = latest_preflight(ctx.store)
    inv = {**_header(ctx), "started_utc": _utc(), "preflight_records": existing_preflights(ctx.store),
           "entry": ctx.entry if k == 1 else None,
           "retained_bytes_before_this_file": ctx.store.footprint()}  # fmt: skip
    try:  # also the filesystem capability check (section 13.1); a concurrent claimant loses at the link (v2.1 A2)
        ctx.publish(p5a.invocation_path(k), json_bytes(inv))
    except (
        ByteCeilingError
    ) as exc:  # section 13.3: a run-level artifact is not published; the invocation stops
        cause = exc.accounting.get("resource_stop_cause")
        raise P5AProtocolError(f"invocation record refused by the byte policy ({cause}); nothing was published, "
                               f"this invocation does not count: {exc.accounting}") from exc  # fmt: skip
    except PublicationError as exc:
        raise P5AProtocolError(f"capability check failed; nothing was published: {exc}") from exc

    recovered = recover(ctx)
    pre: dict[tuple[str, str], dict[str, Any]] = {}
    stop: InvocationStop | None = None
    for rid in p5a.REPRESENTATIONS:
        if stop:
            break
        todo = [c for c in p5a.CONTROLS if st.control_state(ctx.store, rid, c) == st.NOT_STARTED]
        if not todo:
            continue
        t0 = time.perf_counter()
        try:
            if ctx.cpu_env is not None:  # v2.1 A5 P1: the environment just before the inputs are rebuilt
                ctx.identity["cpu_environment"] = ctx.cpu_env()
            frozen = ctx.profile.frozen[rid]
            rs = prepare_representation(frozen, load_population(frozen))
            why = validate_against_preflight(preflight, ctx.identity, rid, evidence(rs))
            if why:
                raise P5AProtocolError(why)
        except P5AProtocolError as exc:
            for c in todo:
                pre[(rid, c)] = {"status": p5a.INCOMPLETE_PROVENANCE, "detail": str(exc)}
            continue
        finally:
            ctx.setup_wall_s += time.perf_counter() - t0
        for c in todo:
            try:
                res = run_control(ctx, rs, rs.controls[c])
            except InvocationStop as s:
                stop = s
                break
            if res.pre_start:
                pre[(rid, c)] = res.pre_start
    # section 13.5: closed by a ceiling stop, by the end of the third invocation, or when every control is recorded
    closed = bool(stop and stop.closes) or k == p5a.MAX_INVOCATIONS
    closed = closed or all(st.control_state(ctx.store, r, c) == st.RECORDED for r, c in p5a.PLAN)
    return finish_invocation(ctx, pre, recovered, stop, closed)


def finish_invocation(
    ctx: RunContext,
    pre: Mapping[tuple[str, str], dict[str, Any]],
    recovered: list[dict[str, Any]],
    stop: InvocationStop | None,
    closed: bool,
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    if closed:
        merged = {**_earlier_pre_start(ctx.store, ctx.invocation), **pre}
        issues = close_run(ctx, resource_closure=bool(stop and stop.closes), pre=merged, stop=stop)
    rows = []
    for rid, c in p5a.PLAN:
        rec = ctx.store.read_json(p5a.record_path(rid, c)) or {}
        rows.append({"representation": rid, "designation": p5a.DESIGNATION[rid], "control": c,
                     "state": st.control_state(ctx.store, rid, c), "status": rec.get("status"),
                     "pre_start": pre.get((rid, c))})  # fmt: skip
    extra: dict[str, Any] = {
        "recovered": recovered,
        "stop": stop.to_dict() if stop else None,
        "closed": closed,
    }
    if closed:
        extra["outcomes"] = outcomes(ctx.store, issues)
        extra["closure_issues"] = issues
    summary = _summary(ctx, rows, **extra)
    try:
        ctx.publish(p5a.summary_path(ctx.invocation), record_bytes(summary))
    except PublicationError as exc:
        summary["summary_publication_error"] = str(exc)
    return summary


def closure_only(ctx: RunContext) -> dict[str, Any]:
    """Section 13.5: completes a run whose third invocation died before ``summary_3``. Reads no payload, builds no
    AE, starts no control and publishes no invocation record; runs at most once."""
    store = ctx.store
    if not store.exists(p5a.invocation_path(p5a.MAX_INVOCATIONS)) or store.exists(
        p5a.summary_path(p5a.MAX_INVOCATIONS)
    ):
        raise P5AProtocolError("closure-only needs invocation_3 without summary_3 (section 13.5)")
    check_same_run(ctx)  # recovery requires the run's own provenance
    ctx.invocation = p5a.MAX_INVOCATIONS
    recovered = recover(ctx)
    return finish_invocation(ctx, {}, recovered, None, closed=True)


def _production_context(config_path: str, overrides: Sequence[str], runs_root: str | None, command: str,
                        launch_record: str | None, *, cuda: bool) -> tuple[RunContext, Path]:  # fmt: skip
    """Shared by the training and closure-only entry points: locked config, no overrides, v2 run root only."""
    from cg_fedllm.config import dump_config_yaml, load_config, resolve_path
    from cg_fedllm.pipeline import apply_vram_guard, identity_config_sha256
    from cg_fedllm.utils.hashing import sha256_file
    from cg_fedllm.utils.provenance import collect_environment
    from cg_fedllm.utils.seeding import configure_determinism

    if overrides:
        raise P5AProtocolError(f"P5-A accepts no --set overrides, got {list(overrides)}")
    cfg = load_config(config_path)
    p5a.check_config(cfg)
    runs = Path(runs_root) if runs_root else resolve_path("${CGFED_RUNS}")
    if not runs.is_absolute() or "${" in str(runs) or not runs.is_dir():
        raise P5AProtocolError(f"CGFED_RUNS must name an existing absolute directory, got {runs}")
    if (runs / p5a.P5A_V1_RUN_NAME).exists():
        raise P5AProtocolError(f"the v1 run root {p5a.P5A_V1_RUN_NAME}/ exists; v1 is never executed")
    launch = None
    if (
        launch_record
    ):  # a citation of the reviewed launch record; it is not, and cannot be, an authorisation check
        lp = Path(launch_record)
        if not lp.is_file():
            raise P5AProtocolError(f"launch record not found: {lp}")
        launch = {"path": lp.as_posix(), "sha256": sha256_file(lp)}
    meta: dict[str, Any] = {"stage": "p5a_v2", "result_label": p5a.P5A_LABEL, "config_sha256": cfg.sha256()}
    device = torch.device("cpu")
    # the locked configuration's settings, exactly as the CPU preflight applies them (v2.1 A5 P1); never disabled
    meta["determinism"] = configure_determinism(cfg.run.deterministic, cfg.run.num_threads)
    if cuda:
        if not torch.cuda.is_available():
            raise P5AProtocolError("P5-A training requires the local CUDA GPU (section 8.3)")
        meta["vram_guard"] = apply_vram_guard(cfg)
        device = torch.device("cuda")
    meta["cpu_environment"] = cpu_environment()
    env = collect_environment()
    meta.update(environment=env, seeds={"autoencoder.init_seed": AE_SEED})
    run_files = {
        "config.resolved.yaml": dump_config_yaml(cfg).encode("utf-8"),
        "config.sha256.json": json_bytes(
            {"config_sha256": cfg.sha256(), "identity_config_sha256": identity_config_sha256(cfg)}
        ),
        "run_metadata.json": record_bytes(meta),
    }
    identity = {"execution_commit": env["git"].get("commit"), "execution_git": env["git"],
                "config_path": config_path, "config_sha256": cfg.sha256(), "launch_record": launch,
                "command": command, "cpu_environment": meta["cpu_environment"]}  # fmt: skip
    ctx = RunContext(
        profile=PRODUCTION,
        store=ArtifactStore(runs / p5a.P5A_RUN_NAME),
        ae_section=cfg.autoencoder,
        device=device,
        identity=identity,
        sync=(lambda: torch.cuda.synchronize(device)) if cuda else (lambda: None),
        run_files=run_files,
        cpu_env=cpu_environment,
    )
    return ctx, runs


def production_main(
    config_path: str,
    overrides: Sequence[str],
    runs_root: str | None,
    command: str,
    launch_record: str | None = None,
) -> dict[str, Any]:
    """``cgfed p5a``: one training invocation. Requires CUDA, the locked config, no overrides and a passing CPU
    preflight record; writes only under ``<CGFED_RUNS>/phase5_p5a_v2_seed1/``. A real invocation needs the
    reviewed launch record and a separate written run authorisation, which no software check can replace."""
    from cg_fedllm.phase5.p5a_inputs import build_population

    ctx, runs = _production_context(config_path, overrides, runs_root, command, launch_record, cuda=True)
    return run_invocation(ctx, lambda frozen: build_population(frozen, runs))


def production_closure(
    config_path: str,
    overrides: Sequence[str],
    runs_root: str | None,
    command: str,
    launch_record: str | None = None,
) -> dict[str, Any]:
    """``cgfed p5a --closure-only``: section 13.5 completion after a death of invocation 3 (no payload, no AE).
    Requires a fresh process with CUDA hidden so shared imports and environment capture cannot initialise it.
    """
    if torch.cuda.is_initialized() or torch.cuda.is_available():
        raise P5AProtocolError(
            "closure-only requires a fresh CPU process with CUDA_VISIBLE_DEVICES=-1; "
            "no CUDA context may already be initialised"
        )
    ctx, _ = _production_context(config_path, overrides, runs_root, command, launch_record, cuda=False)
    return closure_only(ctx)


def own_temporaries(store: ArtifactStore) -> list[str]:
    """Surviving temporaries of this invocation's token (diagnostic only; they are counted, never presented)."""
    return [f for f in store.files() if f.endswith(f".{store.token}{TMP_SUFFIX}")]
