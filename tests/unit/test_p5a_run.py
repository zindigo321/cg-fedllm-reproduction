"""P5-A v2 training invocations on synthetic inputs: real CPU ResNet-3 training/evaluation, gates, accounting,
recovery, closure and publication. Nothing here touches CGFED_RUNS, CUDA or a real payload."""

from __future__ import annotations

import dataclasses
import json

import pytest
import torch

from cg_fedllm.compression.autoencoder import load_autoencoder
from cg_fedllm.config import load_config
from cg_fedllm.phase5 import p5a
from cg_fedllm.phase5 import p5a_run as run
from cg_fedllm.phase5 import p5a_state as st
from cg_fedllm.phase5.p5a_artifacts import TMP_SUFFIX, ArtifactStore, PublicationError, json_bytes
from cg_fedllm.phase5.p5a_inputs import FROZEN, prepare_representation
from tests.unit.p5a_fixtures import (
    CONFIG,
    REPO,
    FakeClock,
    all_pops,
    context,
    exact,
    fake_trainer,
    frozen_for,
    loader,
    population,
    preflight,
    xs_for,
)

H = W = 128


def _read(store, rel):
    return json.loads(store.path(rel).read_text(encoding="utf-8"))


def _ready(tmp_path, pops=None, **kw):
    pops = pops or all_pops()
    ctx = context(tmp_path / p5a.P5A_RUN_NAME, pops, **kw)
    preflight(ctx, pops)
    return ctx, pops


def test_production_config_and_profile_are_locked():
    cfg = load_config(CONFIG)
    p5a.check_config(cfg)
    assert run.PRODUCTION.iterations == 3_000 and run.PRODUCTION.require_cuda
    for path, value in (("autoencoder.iterations", 50), ("autoencoder.learning_rate", 1e-3),
                        ("autoencoder.init_seed", 2), ("autoencoder.batch_size", 1),
                        ("autoencoder.normalization", "global_maxabs_train"),
                        ("autoencoder.checkpoint_policy", "final_and_best_val"),
                        ("run.result_label", "PHASE4-FORENSIC"), ("run.name", "phase5_p5a_seed1")):  # fmt: skip
        with pytest.raises(p5a.P5AProtocolError, match=path):
            p5a.check_config(load_config(CONFIG, [f"{path}={value}"]))
    with pytest.raises(p5a.P5AProtocolError):
        run.Profile(run.PRODUCTION_PROFILE, 10, FROZEN, True)
    with pytest.raises(ValueError):
        run.synthetic_profile(FROZEN, 3_000)
    with pytest.raises(p5a.P5AProtocolError, match="overrides"):
        run.production_main(str(CONFIG), ["autoencoder.iterations=10"], None, "x")
    # no production entry point accepts a profile, a trainer or frozen-value overrides
    import inspect

    for fn in (run.production_main, run.production_closure):
        assert set(inspect.signature(fn).parameters) == {"config_path", "overrides", "runs_root", "command",
                                                          "launch_record"}  # fmt: skip


def test_batch_schedule_matches_the_inherited_loop():
    from cg_fedllm.utils.seeding import numpy_rng

    assert run.batch_schedule(1, 5) == [[0]] * 5  # bs = 1: never cloned to four
    rng = numpy_rng(1, "ae_batches")
    assert run.batch_schedule(4, 3) == [[int(j) for j in rng.permutation(4)] for _ in range(3)]
    rng = numpy_rng(1, "ae_batches")
    p1, p2, p3 = ([int(j) for j in rng.permutation(6)] for _ in range(3))
    assert run.batch_schedule(6, 3) == [p1[:4], p2[:4], p3[:4]]  # redraw when fewer than 4 remain


def _norm(xs):
    return prepare_representation(frozen_for("R4", xs), population("R4", xs)).normalizer


def test_real_resnet_training_and_final_eval_on_cpu():
    xs = xs_for(0)
    norm = _norm(xs)
    ae, ref = run.build_ae(load_config(CONFIG).autoencoder), run.build_ae(load_config(CONFIG).autoencoder)
    assert all(
        torch.equal(a, b) for a, b in zip(ae.state_dict().values(), ref.state_dict().values())
    )  # seeded
    for k in (1, 4):
        ae = run.build_ae(load_config(CONFIG).autoencoder)
        gpu = run.GpuClock(FakeClock(), lambda: None, 1_800.0)
        out = run.train_and_evaluate(ae, xs[:k], norm, iterations=3, device=torch.device("cpu"), gpu=gpu)
        assert [c["iteration"] for c in out["curve"]] == [1, 3] and out["iterations_done"] == 3
        assert (
            not ae.training and len(out["reconstructions"]) == k
        )  # eval mode; one reconstruction per snapshot
        bn = [n for n in out["state"] if n.endswith("running_mean")]
        assert bn and any(float(out["state"][n].abs().sum()) > 0 for n in bn)  # BatchNorm buffers kept
        with torch.no_grad():  # the evaluation is the final AE in eval mode, de-normalised: recompute it
            y = ae.decode(ae.encode(norm.normalize(xs[0]).unsqueeze(0)))[0]
        assert torch.allclose(out["reconstructions"][0], norm.denormalize(y), rtol=0, atol=1e-7)
        assert gpu.elapsed is not None and gpu.elapsed > 0


def test_nonfinite_loss_timeout_crash_and_operator_interrupt():
    xs = xs_for(1)[:1]
    norm = _norm(xs_for(1))
    sec = load_config(CONFIG).autoencoder
    with pytest.raises(run.TrainingStop) as e:
        run.train_and_evaluate(run.build_ae(sec), [torch.full((1, H, W), float("nan"))], norm, iterations=3,
                               device=torch.device("cpu"), gpu=run.GpuClock(FakeClock(), lambda: None, 1_800.0))  # fmt: skip
    assert e.value.status == p5a.INCOMPLETE_NON_FINITE and e.value.iterations_done == 0
    slow = run.GpuClock(FakeClock(step=700.0), lambda: None, 1_800.0)
    with pytest.raises(run.TrainingStop) as e:
        run.train_and_evaluate(
            run.build_ae(sec), xs, norm, iterations=10, device=torch.device("cpu"), gpu=slow
        )
    assert e.value.status == p5a.INCOMPLETE_RESOURCE and slow.elapsed >= 1_800.0  # measured, not clamped

    class Boom(torch.nn.Module):
        def __init__(self, exc):
            super().__init__()
            self.exc = exc

        def to(self, *_a, **_k):
            raise self.exc

    for exc, operator in ((torch.OutOfMemoryError("synthetic OOM"), False), (KeyboardInterrupt(), True)):
        gpu = run.GpuClock(FakeClock(), lambda: None, 1_800.0)
        with pytest.raises(run.TrainingStop) as e:
            run.train_and_evaluate(Boom(exc), xs, norm, iterations=3, device=torch.device("cpu"), gpu=gpu)
        assert e.value.status == p5a.INCOMPLETE_INTERRUPTED and e.value.operator_interrupt is operator
        assert gpu.elapsed is not None  # the clock started before ae.to(device) and was stopped


def test_real_evaluation_overrun_path_publishes_no_checkpoint(tmp_path):
    """The real trainer with a clock that is fast during training and slow only once evaluation starts: every
    pre-evaluation check passes, the measured time at the end of evaluation is >= 1,800 s (section 12)."""

    class EvalSlow:
        def __init__(self):
            self.t, self.slow = 0.0, False

        def __call__(self):  # one read per check; slow only after the evaluation codec is built
            self.t += 2_000.0 if self.slow else 1.0
            return self.t

    clock = EvalSlow()
    real_eval = run.AutoEncoderCodec.__init__

    def mark_eval(self, *a, **k):  # the codec is built only for the section-8.4 evaluation
        clock.slow = True
        real_eval(self, *a, **k)

    import unittest.mock as mock

    ctx, pops = _ready(tmp_path, clock=clock)
    assert ctx.trainer is run.train_and_evaluate  # the real trainer, including its timeout and overrun checks
    with mock.patch.object(run.AutoEncoderCodec, "__init__", mark_eval):
        summary = run.run_invocation(ctx, loader(pops))
    rec = _read(ctx.store, p5a.record_path("R2", "single_snapshot"))
    assert rec["status"] == p5a.INCOMPLETE_RESOURCE, (rec["status"], rec.get("timing"), rec.get("detail"))
    assert "during evaluation" in rec["detail"]
    assert rec["timing"]["gpu_wall_s_measured"] >= 1_800.0 and rec["iterations_done"] == 2
    assert not ctx.store.exists(p5a.checkpoint_path("R2", "single_snapshot"))
    assert summary["controls"][1]["status"] is not None  # a per-training overrun does not stop the invocation


def test_evaluation_overrun_is_a_resource_stop_without_checkpoint():
    xs = xs_for(2)[:1]
    clock = FakeClock(step=1.0)
    sec = load_config(CONFIG).autoencoder

    class Late(run.GpuClock):
        def stop(self):  # the evaluation itself pushes the measured time past the limit
            self.clock.t += 2_000.0
            return super().stop()

    gpu = Late(clock, lambda: None, 1_800.0)
    with pytest.raises(run.TrainingStop) as e:
        run.train_and_evaluate(run.build_ae(sec), xs, _norm(xs_for(2)), iterations=2, device=torch.device("cpu"),
                               gpu=gpu)  # fmt: skip
    assert e.value.status == p5a.INCOMPLETE_RESOURCE and "during evaluation" in e.value.detail


def test_end_to_end_invocation_trains_the_fixed_plan_in_order(tmp_path):
    order = []
    real = run.train_and_evaluate

    def spy(ae, xs, norm, **kw):
        order.append(len(xs))
        return real(ae, xs, norm, **kw)

    ctx, pops = _ready(tmp_path, trainer=spy)
    summary = run.run_invocation(ctx, loader(pops))
    assert order == [1, 4, 1, 4, 1, 4]  # R2 s, R2 f, R3 s, R3 f, R4 s, R4 f
    assert summary["closed"] and summary["invocation"] == 1
    store = ctx.store
    for rid, control in p5a.PLAN:
        rec = _read(store, p5a.record_path(rid, control))
        assert rec["status"] in (p5a.FIT_PASS, p5a.FIT_FAIL) and rec["is_result"] is False
        assert (
            rec["protocol"]["commit"] == p5a.P5A_PROTOCOL_COMMIT
            and rec["designation"] == p5a.DESIGNATION[rid]
        )
        assert rec["plan_position"] == p5a.PLAN.index((rid, control)) + 1
        assert rec["thresholds"] == p5a.THRESHOLDS[control]
        assert rec["normalization"]["normalizer"]["mode"] == "global_exact_maxabs_train"
        assert rec["timing"]["timing_status"] == "measured" and rec["timing"]["charge_reason"] == "measured"
        assert ("C4" in rec["evaluation"]["decision"]["criteria"]) == (control == "fixed_subset4")
        ae, meta = load_autoencoder(store.path(p5a.checkpoint_path(rid, control)))
        assert meta["checkpoint"] == "final" and meta["iteration"] == 2 and not ae.training
        assert _read(store, p5a.end_path(rid, control))["training_end"] == "completed"
        assert st.control_state(store, rid, control) == st.RECORDED
    # both controls of a representation share the one frozen scale
    for rid in p5a.REPRESENTATIONS:
        scales = {_read(store, p5a.record_path(rid, c))["normalization"]["normalizer"]["scale_global"]
                  for c in p5a.CONTROLS}  # fmt: skip
        assert scales == {ctx.profile.frozen[rid].scale}
    outs = summary["outcomes"]
    assert [o["representation"] for o in outs] == ["R2", "R3", "R4"] and outs[2]["designation"] == "secondary"
    for o in outs:
        assert o["outcome"] == p5a.representation_outcome(o["controls"]["single_snapshot"],
                                                          o["controls"]["fixed_subset4"])  # fmt: skip
    assert not any("combined" in k or k == "overall" for k in summary)
    files = store.files()
    assert p5a.summary_path(1) in files and not [f for f in files if f.endswith(".p5a-tmp")]
    on_disk = _read(store, p5a.summary_path(1))
    assert on_disk["retained_bytes_before_this_file"] + store.size(p5a.summary_path(1)) == store.footprint()


def test_single_snapshot_failure_does_not_skip_fixed_subset4(tmp_path):
    def recon(xs):
        return [torch.zeros_like(x) for x in xs] if len(xs) == 1 else exact(xs)

    ctx, pops = _ready(tmp_path, trainer=fake_trainer(recon))
    outs = {o["representation"]: o for o in run.run_invocation(ctx, loader(pops))["outcomes"]}
    for rid in p5a.REPRESENTATIONS:
        assert outs[rid]["controls"] == {"single_snapshot": p5a.FIT_FAIL, "fixed_subset4": p5a.FIT_PASS}
        assert outs[rid]["outcome"] == p5a.NOT_DEMONSTRATED


def test_strict_single_snapshot_gate_is_wired_into_the_runner(tmp_path):
    def recon(
        xs,
    ):  # RSE 0.0225 for every snapshot: passes the subset standard, fails the memorisation standard
        return [x * 0.85 for x in xs]

    ctx, pops = _ready(tmp_path, trainer=fake_trainer(recon))
    run.run_invocation(ctx, loader(pops))
    single = _read(ctx.store, p5a.record_path("R2", "single_snapshot"))
    subset = _read(ctx.store, p5a.record_path("R2", "fixed_subset4"))
    assert single["status"] == p5a.FIT_FAIL and subset["status"] == p5a.FIT_PASS
    c2 = single["evaluation"]["decision"]["criteria"]["C2"]["per_snapshot"][0]
    assert c2["threshold"] == 0.01 and c2["value"] == pytest.approx(0.0225, rel=1e-6) and c2["pass"] is False


def test_c0_is_an_integrity_check_and_identical_snapshots_are_not_trained(tmp_path):
    pops = all_pops()
    same = xs_for(9)[0]
    pops["R4"] = [same.clone() for _ in range(5)]  # four identical selected snapshots: subset-mean SSE = 0
    calls = []

    def trainer(ae, xs, *a, **k):
        calls.append(len(xs))
        return fake_trainer(exact)(ae, xs, *a, **k)

    ctx = context(tmp_path / p5a.P5A_RUN_NAME, pops, trainer=trainer)
    preflight(ctx, pops)
    summary = run.run_invocation(ctx, loader(pops))
    # a pre-start status keeps the control NOT_STARTED and goes into the summary; the run stays open (12, 13.5)
    row = next(
        r for r in summary["controls"] if (r["representation"], r["control"]) == ("R4", "fixed_subset4")
    )
    assert row["state"] == st.NOT_STARTED and row["pre_start"]["status"] == p5a.NOT_INFORMATIVE
    assert not summary["closed"] and "outcomes" not in summary  # outcomes only at closure
    assert not ctx.store.exists(p5a.start_path("R4", "fixed_subset4"))
    assert calls == [1, 4, 1, 4, 1]  # only R4 fixed_subset4 is skipped; nothing else is pruned
    # re-entry: C0 fails again (inputs unchanged); the third invocation closes the run with the last status
    for k in (2, 3):
        summary = run.run_invocation(dataclasses.replace(ctx, invocation=0), loader(pops))
        assert summary["invocation"] == k
    assert calls == [1, 4, 1, 4, 1] and summary["closed"]
    rec = _read(ctx.store, p5a.record_path("R4", "fixed_subset4"))
    assert rec["status"] == p5a.NOT_INFORMATIVE and rec["trained"] is False
    assert summary["outcomes"][2]["controls"]["fixed_subset4"] == p5a.NOT_INFORMATIVE
    with pytest.raises(p5a.P5AProtocolError, match="closed"):
        run.run_invocation(dataclasses.replace(ctx, invocation=0), loader(pops))


def test_zero_signal_snapshot_is_incomplete_provenance_and_not_trained(tmp_path):
    pops = all_pops()
    pops["R4"][2] = torch.zeros(1, H, W)  # the single_snapshot position for N = 5
    ctx = context(tmp_path / p5a.P5A_RUN_NAME, pops, trainer=fake_trainer(exact))
    preflight(ctx, pops)
    summary = run.run_invocation(ctx, loader(pops))
    row = next(
        r for r in summary["controls"] if (r["representation"], r["control"]) == ("R4", "single_snapshot")
    )
    assert row["pre_start"]["status"] == p5a.INCOMPLETE_PROVENANCE and "sig = 0" in row["pre_start"]["detail"]
    assert not ctx.store.exists(p5a.start_path("R4", "single_snapshot"))
    assert _read(ctx.store, p5a.record_path("R4", "fixed_subset4"))["status"] == p5a.FIT_PASS


def _again(ctx):
    return dataclasses.replace(ctx, invocation=0, measured={})


def test_crash_counts_is_never_retried_and_the_invocation_continues(tmp_path):
    def crash(ae, xs, norm, *, iterations, device, gpu):
        gpu.start()
        gpu.stop()
        raise run.TrainingStop(p5a.INCOMPLETE_INTERRUPTED, "RuntimeError: boom", [], 0)

    ctx, pops = _ready(tmp_path, trainer=crash)
    summary = run.run_invocation(ctx, loader(pops))
    assert summary["closed"] and summary["stop"] is None  # a crash does not stop the invocation
    for rid, control in p5a.PLAN:
        rec = _read(ctx.store, p5a.record_path(rid, control))
        assert rec["status"] == p5a.INCOMPLETE_INTERRUPTED and rec["trained"] is True
        assert not ctx.store.exists(p5a.checkpoint_path(rid, control))  # INCOMPLETE: no checkpoint
        assert rec["timing"]["timing_status"] == "measured"
    assert {o["outcome"] for o in summary["outcomes"]} == {p5a.INCONCLUSIVE}
    with pytest.raises(p5a.P5AProtocolError, match="closed"):
        run.run_invocation(_again(ctx), loader(pops))


def test_operator_interrupt_publishes_records_and_stops_the_invocation(tmp_path):
    def interrupted(ae, xs, norm, *, iterations, device, gpu):
        gpu.start()
        gpu.stop()
        raise run.TrainingStop(p5a.INCOMPLETE_INTERRUPTED, "KeyboardInterrupt", [], 1, operator=True)

    ctx, pops = _ready(tmp_path, trainer=interrupted)
    summary = run.run_invocation(ctx, loader(pops))
    assert summary["stop"]["reason"] == "operator_interrupt" and not summary["closed"]
    assert _read(ctx.store, p5a.record_path("R2", "single_snapshot"))["status"] == p5a.INCOMPLETE_INTERRUPTED
    assert (
        st.control_state(ctx.store, "R2", "fixed_subset4") == st.NOT_STARTED
    )  # not reached: stays NOT_STARTED


def _hard_kill(root, mode):
    """Run one synthetic invocation in a child process and end it with ``os._exit`` at a P5-A boundary."""
    import os
    import subprocess
    import sys

    from tests.unit.p5a_hard_kill import KILL_CODE

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "HF_HUB_OFFLINE": "1"}
    proc = subprocess.run([sys.executable, "-m", "tests.unit.p5a_hard_kill", str(root), mode],
                          cwd=REPO, env=env, capture_output=True, text=True, timeout=600)  # fmt: skip
    assert proc.returncode == KILL_CODE, proc.stderr[-2000:]


def test_hard_kill_after_the_start_record_is_recovered_without_retraining(tmp_path):
    root = tmp_path / p5a.P5A_RUN_NAME
    _hard_kill(root, "after_start")  # os._exit inside ae.to(device): no finally, except or cleanup runs
    store = ArtifactStore(root)
    assert st.control_state(store, "R2", "single_snapshot") == st.STARTED
    assert not store.exists(p5a.end_path("R2", "single_snapshot"))
    assert not store.exists(p5a.summary_path(1)) and st.resource_state(store)["trainings_started"] == 1
    pops = all_pops()
    trained = []
    ctx2 = context(root, pops, trainer=fake_trainer(lambda xs: trained.append(len(xs)) or exact(xs)))
    summary = run.run_invocation(ctx2, loader(pops))
    assert summary["stop"] is None, summary["stop"]
    assert summary["closed"], [
        (r["control"], r["state"], r["status"], r["pre_start"]) for r in summary["controls"]
    ]
    assert summary["invocation"] == 2 and summary["recovered"][0]["from"] == st.STARTED
    end = _read(store, p5a.end_path("R2", "single_snapshot"))
    assert end["timing"] == {"gpu_wall_s_measured": None, "timing_status": "unknown_process_death",
                             "gpu_wall_s_charged": 1_800.0,
                             "charge_reason": "conservative_full_allocation_after_unobserved_process_death"}  # fmt: skip
    assert _read(store, p5a.record_path("R2", "single_snapshot"))["status"] == p5a.INCOMPLETE_INTERRUPTED
    assert trained == [4, 1, 4, 1, 4]  # the started control is never trained again
    assert st.charged_s(store, "R2", "single_snapshot") == 1_800.0


def test_hard_kill_while_writing_the_start_record_counts_no_training(tmp_path):
    root = tmp_path / p5a.P5A_RUN_NAME
    _hard_kill(root, "during_start_write")  # the temporary is written and fsynced, the link never happens
    store = ArtifactStore(root)
    temps = [f for f in store.files() if f.endswith(TMP_SUFFIX)]
    assert len(temps) == 1 and temps[0].startswith("ledger/.start_R2_single_snapshot.json.")  # left behind
    assert st.control_state(store, "R2", "single_snapshot") == st.NOT_STARTED
    assert st.resource_state(store)["trainings_started"] == 0
    tmp_path_ = root / temps[0]
    tmp_bytes = tmp_path_.stat().st_size
    assert tmp_bytes > 0 and store.footprint() >= tmp_bytes  # the surviving temporary is counted
    # re-entry: the root now holds ledger/invocation_1.json, so this is invocation 2; the control never started
    pops = all_pops()
    summary = run.run_invocation(context(root, pops, trainer=fake_trainer(exact)), loader(pops))
    assert summary["invocation"] == 2 and summary["recovered"] == []
    assert _read(store, p5a.record_path("R2", "single_snapshot"))["status"] == p5a.FIT_PASS
    assert tmp_path_.stat().st_size == tmp_bytes  # never deleted: it belongs to the dead invocation
    assert summary["closed"]


def test_hard_kill_after_the_end_record_keeps_completion_and_is_recovered_as_publication(tmp_path):
    root = tmp_path / p5a.P5A_RUN_NAME
    _hard_kill(root, "after_end")  # dies after the end record, while linking the checkpoint
    store = ArtifactStore(root)
    assert st.control_state(store, "R2", "single_snapshot") == st.ENDED
    assert _read(store, p5a.end_path("R2", "single_snapshot"))["training_end"] == "completed"
    assert not store.exists(p5a.checkpoint_path("R2", "single_snapshot"))
    assert [f for f in store.files() if f.endswith(TMP_SUFFIX)]  # the checkpoint temporary survived the kill
    pops = all_pops()
    summary = run.run_invocation(context(root, pops, trainer=fake_trainer(exact)), loader(pops))
    rec = _read(store, p5a.record_path("R2", "single_snapshot"))
    assert (
        rec["status"] == p5a.INCOMPLETE_PUBLICATION and rec["checkpoint"] is None and rec["trained"] is True
    )
    assert summary["recovered"][0] == {"representation": "R2", "control": "single_snapshot", "from": st.ENDED,
                                       "status": p5a.INCOMPLETE_PUBLICATION}  # fmt: skip


def test_an_ended_control_without_record_is_recovered_as_publication(tmp_path):
    ctx, pops = _ready(tmp_path, trainer=fake_trainer(exact))
    real = ctx.store.linker

    def fail_record(src, dst):
        if dst.name == "p5a_record.json":
            raise OSError("record write failed")
        real(src, dst)

    ctx.store.linker = fail_record
    summary = run.run_invocation(ctx, loader(pops))
    assert summary["stop"]["reason"] == "publication_failure" and not summary["closed"]
    assert st.control_state(ctx.store, "R2", "single_snapshot") == st.ENDED
    assert ctx.store.exists(p5a.checkpoint_path("R2", "single_snapshot"))  # finalized checkpoint kept
    fail = _read(ctx.store, p5a.failure_path("R2", "single_snapshot"))
    assert fail["status"] == p5a.INCOMPLETE_PUBLICATION and fail["reported_only_gate_result"] == p5a.FIT_PASS
    ctx2 = _again(ctx)
    ctx2.store = ArtifactStore(ctx.store.root)
    summary = run.run_invocation(ctx2, loader(pops))
    rec = _read(ctx2.store, p5a.record_path("R2", "single_snapshot"))
    assert rec["status"] == p5a.INCOMPLETE_PUBLICATION and rec["checkpoint"]["bytes"] > 0
    assert summary["closed"] and summary["outcomes"][0]["outcome"] == p5a.INCONCLUSIVE


def test_aggregate_time_reservation_and_six_training_ceiling(tmp_path):
    store = ArtifactStore(tmp_path / "run")
    for rid, control in p5a.PLAN[:5]:
        store.publish(p5a.start_path(rid, control), b"{}", reserved_other=0)
        store.publish(
            p5a.end_path(rid, control), json_bytes({"timing": st.timing(1_800.0)}), reserved_other=0
        )
    state = st.resource_state(store)
    assert state["trainings_started"] == 5 and state["gpu_wall_s_charged_total"] == 9_000.0
    assert state["may_start"]  # 9,000 + 1,800 = 10,800: allowed (inclusive)
    store.publish(p5a.start_path(*p5a.PLAN[5]), b"{}", reserved_other=0)
    assert not st.resource_state(store)["may_start"]  # six trainings started
    store2 = ArtifactStore(tmp_path / "run2")
    for i, (rid, control) in enumerate(p5a.PLAN[:5]):
        store2.publish(p5a.start_path(rid, control), b"{}", reserved_other=0)
        t = 1_800.5 if i == 0 else 1_800.0
        store2.publish(p5a.end_path(rid, control), json_bytes({"timing": st.timing(t)}), reserved_other=0)
    assert not st.resource_state(store2)["may_start"]  # 9,000.5 + 1,800 > 10,800


def test_per_training_limit_and_time_ceiling_with_a_fake_clock(tmp_path):
    # every clock read advances 700 s: a training is stopped at the before-evaluation check (2,100 s) and measures
    # 2,800 s at its final stop. Charged: 2,800 per training. Starts: 0 + 1,800, 2,800 + 1,800, 5,600 + 1,800 and
    # 8,400 + 1,800 <= 10,800 are allowed; 11,200 + 1,800 > 10,800 refuses the fifth (a time-ceiling stop).
    ctx, pops = _ready(tmp_path, clock=FakeClock(step=700.0))
    summary = run.run_invocation(ctx, loader(pops))
    assert summary["stop"]["reason"] == "time_ceiling" and summary["closed"]
    for rid, control in p5a.PLAN[:4]:
        rec = _read(ctx.store, p5a.record_path(rid, control))
        assert rec["status"] == p5a.INCOMPLETE_RESOURCE and rec["trained"] is True
        assert (
            rec["timing"]["gpu_wall_s_measured"] == 2_800.0 and rec["timing"]["gpu_wall_s_charged"] == 2_800.0
        )
        assert not ctx.store.exists(
            p5a.checkpoint_path(rid, control)
        )  # no checkpoint for an INCOMPLETE training
    for rid, control in p5a.PLAN[4:]:
        rec = _read(ctx.store, p5a.record_path(rid, control))
        assert rec["status"] == p5a.INCOMPLETE_RESOURCE and rec["trained"] is False
        assert not ctx.store.exists(p5a.start_path(rid, control))
    state = st.resource_state(ctx.store)
    assert (
        state["trainings_started"] == 4 and state["gpu_wall_s_charged_total"] == 11_200.0
    )  # measured, unclamped


def test_byte_reservation_refusal_closes_the_run_before_training(tmp_path):
    ctx, pops = _ready(tmp_path)
    # at the first training start the reservation is: summaries 1-3, invocation records 2-3, the evidence of all six
    # controls, and this control's checkpoint (section 13.3); the start record's own cap is part of the evidence
    at_start = 3 * 200_000 + 2 * 50_000 + 6 * (20_000 + 20_000 + 100_000 + 150_000 + 500_000) + 2_100_000
    assert at_start == 7_540_000
    ctx.store.limit = (
        ctx.store.footprint() + at_start - 1
    )  # the invocation record fits; the first start does not
    calls = []
    ctx.trainer = lambda *a, **k: calls.append(1)
    summary = run.run_invocation(ctx, loader(pops))
    assert summary["stop"]["reason"] == "byte_ceiling" and summary["closed"] and calls == []
    for rid, control in p5a.PLAN:
        assert _read(ctx.store, p5a.record_path(rid, control))["status"] == p5a.INCOMPLETE_RESOURCE
        assert not ctx.store.exists(p5a.start_path(rid, control))
    assert ctx.store.footprint() <= ctx.store.limit and ctx.store.exists(p5a.summary_path(1))


def test_checkpoint_is_not_published_unless_remaining_evidence_is_reserved(tmp_path):
    ctx, pops = _ready(tmp_path, trainer=fake_trainer(exact))
    real_check = ctx.store.check

    def tight(rel, data, reserved_other):  # shrink the ceiling exactly when the first checkpoint is checked
        acc = real_check(rel, data, reserved_other)
        if rel.endswith(".safetensors") and not ctx.store.files_ckpt_seen:
            ctx.store.files_ckpt_seen = True
            ctx.store.limit = acc["reservation_total"] - 1
            acc = real_check(rel, data, reserved_other)
        return acc

    ctx.store.files_ckpt_seen = False
    ctx.store.check = tight
    summary = run.run_invocation(ctx, loader(pops))
    assert summary["stop"] == {"reason": "byte_ceiling", "resource_stop_cause": p5a.STOP_RESERVATION,
                               "detail": summary["stop"]["detail"]} and summary["closed"]  # fmt: skip
    assert not ctx.store.exists(p5a.checkpoint_path("R2", "single_snapshot"))
    assert ctx.store.exists(p5a.end_path("R2", "single_snapshot"))  # training completion is kept
    rec = _read(ctx.store, p5a.record_path("R2", "single_snapshot"))
    assert rec["status"] == p5a.INCOMPLETE_PUBLICATION and rec["reported_only_gate_result"] == p5a.FIT_PASS
    assert rec["resource_stop_cause"] == p5a.STOP_RESERVATION
    later = _read(
        ctx.store, p5a.record_path("R4", "fixed_subset4")
    )  # never started: the frozen v2 closure rule
    assert later["status"] == p5a.INCOMPLETE_RESOURCE and later["trained"] is False
    assert later["run_closed_by"]["resource_stop_cause"] == p5a.STOP_RESERVATION
    r2 = summary["outcomes"][0]
    assert r2["outcome"] == p5a.INCONCLUSIVE and r2["reported_only_gate_results"]["controls"] == {
        "single_snapshot": p5a.FIT_PASS}  # fmt: skip


def _refuse_checkpoint_over_cap(store):
    """Make the first checkpoint exceed its own 2,100,000-byte cap (v2.1 A4.1 ``per_file_cap``)."""
    real_check = store.check

    def over(rel, data, reserved_other):
        if rel.endswith(".safetensors"):
            data = data + b"\0" * (2_100_001 - len(data))
        return real_check(rel, data, reserved_other)

    store.check = over


def test_over_cap_checkpoint_of_a_completed_training_is_publication_not_resource_stop(tmp_path):
    """Engineering choice 2, resolved by v2.1 A4.2/A4.3: the same S3 status for a per-file cap refusal as for a
    reservation refusal; the cause is recorded separately and only controls that never started get RESOURCE STOP."""
    ctx, pops = _ready(
        tmp_path, trainer=fake_trainer(lambda xs: [x * 0.0 for x in xs])
    )  # a computed FIT FAIL
    _refuse_checkpoint_over_cap(ctx.store)
    summary = run.run_invocation(ctx, loader(pops))
    assert (
        summary["stop"]["reason"] == "byte_ceiling"
        and summary["stop"]["resource_stop_cause"] == "per_file_cap"
    )
    assert summary["closed"]
    rec = _read(ctx.store, p5a.record_path("R2", "single_snapshot"))
    assert rec["status"] == p5a.INCOMPLETE_PUBLICATION != p5a.INCOMPLETE_RESOURCE
    assert rec["reported_only_gate_result"] == p5a.FIT_FAIL and rec["resource_stop_cause"] == "per_file_cap"
    assert rec["evaluation"]["decision"]["pass"] is False  # the observed failure evidence is kept
    fail = _read(ctx.store, p5a.failure_path("R2", "single_snapshot"))
    assert fail["status"] == p5a.INCOMPLETE_PUBLICATION and fail["accounting"]["within_cap"] is False
    assert fail["failed_file"]["path"] == p5a.checkpoint_path("R2", "single_snapshot")
    assert _read(ctx.store, p5a.end_path("R2", "single_snapshot"))["training_end"] == "completed"
    for rid, control in p5a.PLAN[1:]:
        later = _read(ctx.store, p5a.record_path(rid, control))
        assert later["status"] == p5a.INCOMPLETE_RESOURCE and later["trained"] is False
        assert not ctx.store.exists(p5a.start_path(rid, control))
    # a reported-only FIT FAIL is not a gate result: the outcome stays INCONCLUSIVE and the evidence is listed
    assert summary["outcomes"][0]["outcome"] == p5a.INCONCLUSIVE
    assert summary["outcomes"][0]["reported_only_gate_results"]["controls"] == {
        "single_snapshot": p5a.FIT_FAIL
    }


def test_training_stop_status_survives_a_later_byte_refusal(tmp_path):
    """v2.1 A4.2 S1: an interrupted training keeps INCOMPLETE (INTERRUPTED) when one of its own files is refused."""

    def crash(ae, xs, norm, *, iterations, device, gpu):
        gpu.start()
        gpu.stop()
        raise run.TrainingStop(p5a.INCOMPLETE_INTERRUPTED, "RuntimeError: synthetic", [], 1)

    ctx, pops = _ready(tmp_path, trainer=crash)
    real_check = ctx.store.check

    def refuse_provenance(
        rel, data, reserved_other
    ):  # the provenance set of the first control is over its cap
        if rel.endswith("config.resolved.yaml"):
            data = data + b"\0" * 150_001
        return real_check(rel, data, reserved_other)

    ctx.store.check = refuse_provenance
    summary = run.run_invocation(ctx, loader(pops))
    assert summary["stop"]["resource_stop_cause"] == "per_file_cap" and summary["closed"]
    rec = _read(ctx.store, p5a.record_path("R2", "single_snapshot"))
    assert rec["status"] == p5a.INCOMPLETE_INTERRUPTED and rec["reported_only_gate_result"] is None
    assert _read(ctx.store, p5a.failure_path("R2", "single_snapshot"))["status"] == p5a.INCOMPLETE_INTERRUPTED
    assert _read(ctx.store, p5a.record_path("R2", "fixed_subset4"))["status"] == p5a.INCOMPLETE_RESOURCE


def test_status_precedence_helpers():
    for s in p5a.TRAINING_STOP_STATUSES:
        assert run.started_status(s) == s and run.ended_status(s) == s  # S1 is kept
    for gate in (None, p5a.FIT_PASS, p5a.FIT_FAIL):
        assert run.started_status(gate) == p5a.INCOMPLETE_PUBLICATION  # S3, whatever the cause
    assert run.ended_status("completed") == p5a.INCOMPLETE_PUBLICATION
    assert run.ended_status("unknown_process_death") == p5a.INCOMPLETE_INTERRUPTED
    assert run.ended_status(None) == p5a.INCOMPLETE_INTERRUPTED


def test_recovered_ended_training_stop_keeps_its_status(tmp_path):
    """v2.1 A4.4: an ENDED control without a record takes its status from the end record, not a blanket PUBLICATION."""
    ctx, pops = _ready(tmp_path, trainer=fake_trainer(exact))
    for k in (1,):
        ctx.store.publish(
            p5a.invocation_path(k), json_bytes({**run._header(ctx), "invocation": k}), reserved_other=0
        )
    ctx.store.publish(p5a.start_path("R2", "single_snapshot"), b"{}", reserved_other=0)
    ctx.store.publish(p5a.end_path("R2", "single_snapshot"), json_bytes(
        {"training_end": p5a.INCOMPLETE_NON_FINITE, "timing": st.timing(12.5)}), reserved_other=0)  # fmt: skip
    ctx.invocation = 2
    out = run.recover(ctx)
    assert out == [{"representation": "R2", "control": "single_snapshot", "from": st.ENDED,
                    "status": p5a.INCOMPLETE_NON_FINITE}]  # fmt: skip
    rec = _read(ctx.store, p5a.record_path("R2", "single_snapshot"))
    assert rec["status"] == p5a.INCOMPLETE_NON_FINITE and rec["timing"]["gpu_wall_s_charged"] == 12.5


def test_entry_rules_v1_root_missing_preflight_and_identity(tmp_path):
    pops = all_pops()
    v1 = context(tmp_path / p5a.P5A_V1_RUN_NAME, pops)
    with pytest.raises(p5a.P5AProtocolError, match="v1"):
        run.run_invocation(v1, loader(pops))
    bare = context(tmp_path / p5a.P5A_RUN_NAME, pops)
    with pytest.raises(p5a.P5AProtocolError, match="preflight"):
        run.run_invocation(bare, loader(pops))
    assert not bare.store.root.exists()  # refused before anything is published
    preflight(bare, pops)
    other = dataclasses.replace(bare, identity={**bare.identity, "config_sha256": "0" * 64})
    summary = run.run_invocation(other, loader(pops))  # stale preflight: every representation is PROVENANCE
    assert {r["pre_start"]["status"] for r in summary["controls"]} == {p5a.INCOMPLETE_PROVENANCE}
    assert all("stale preflight" in r["pre_start"]["detail"] for r in summary["controls"])
    with pytest.raises(p5a.P5AProtocolError, match="different configuration"):
        run.run_invocation(dataclasses.replace(_again(bare)), loader(pops))


def test_changed_inputs_after_the_preflight_stop_that_representation(tmp_path):
    ctx, pops = _ready(tmp_path, trainer=fake_trainer(exact))
    changed = {**pops, "R3": [x.clone() for x in pops["R3"]]}
    changed["R3"][0][0, 0, 0] += 1e-6  # different Phi bytes, same frozen maximum and RMS within tolerance
    summary = run.run_invocation(ctx, loader(changed))
    r3 = [r for r in summary["controls"] if r["representation"] == "R3"]
    assert {r["pre_start"]["status"] for r in r3} == {p5a.INCOMPLETE_PROVENANCE}
    assert all("differ from the preflight" in r["pre_start"]["detail"] for r in r3)
    assert _read(ctx.store, p5a.record_path("R4", "fixed_subset4"))["status"] == p5a.FIT_PASS  # isolated


def test_fourth_invocation_and_closure_only_step(tmp_path):
    ctx, pops = _ready(tmp_path, trainer=fake_trainer(exact))
    for _ in range(2):  # invocations 1 and 2 stop at once on a publication failure of the first start record
        c = _again(ctx)
        c.store = ArtifactStore(ctx.store.root, linker=lambda s, d: (_ for _ in ()).throw(OSError("x"))
                                if d.name.startswith("start_") else __import__("os").link(s, d))  # fmt: skip
        summary = run.run_invocation(c, loader(pops))
        assert not summary["closed"]
    # invocation 3 dies hard after its first start record: simulate exactly the files such a death leaves
    ctx.store.publish(p5a.invocation_path(3), json_bytes({"identity": ctx.identity}), reserved_other=0)
    ctx.store.publish(p5a.start_path("R2", "single_snapshot"), b"{}", reserved_other=0)
    assert ctx.store.exists(p5a.invocation_path(3)) and not ctx.store.exists(p5a.summary_path(3))
    with pytest.raises(p5a.P5AProtocolError, match="at most 3"):
        run.run_invocation(_again(ctx), loader(pops))
    other = dataclasses.replace(_again(ctx), identity={**ctx.identity, "config_sha256": "0" * 64})
    with pytest.raises(p5a.P5AProtocolError, match="different configuration"):
        run.closure_only(other)  # recovery requires matching provenance
    calls = []
    summary = run.closure_only(dataclasses.replace(_again(ctx), trainer=lambda *a, **k: calls.append(1)))
    assert summary["closed"] and summary["invocation"] == 3 and calls == []  # no training is granted
    assert not ctx.store.exists(p5a.invocation_path(4)) and ctx.store.exists(p5a.summary_path(3))
    assert _read(ctx.store, p5a.record_path("R2", "single_snapshot"))["status"] == p5a.INCOMPLETE_INTERRUPTED
    assert _read(ctx.store, p5a.record_path("R4", "fixed_subset4"))["status"] == p5a.INCOMPLETE_NOT_STARTED
    with pytest.raises(p5a.P5AProtocolError):
        run.closure_only(_again(ctx))  # at most once


def test_production_entry_refuses_without_cuda_or_with_overrides(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(p5a.P5AProtocolError, match="CUDA"):
        run.production_main(str(CONFIG), [], str(tmp_path), "x")
    (tmp_path / p5a.P5A_V1_RUN_NAME).mkdir()
    with pytest.raises(p5a.P5AProtocolError, match="v1"):
        run.production_main(str(CONFIG), [], str(tmp_path), "x")
    assert sorted(p.name for p in tmp_path.iterdir()) == [p5a.P5A_V1_RUN_NAME]  # nothing else was written
    with pytest.raises(PublicationError):
        ArtifactStore(tmp_path).publish("ledger/summary_9.json", b"{}", reserved_other=0)


def test_hard_kill_during_the_first_record_permits_the_invocation_one_slot(tmp_path):
    """v2.1 A2 (finding F1): invocation 1 dies while writing its own first record. ledger/ and an fsynced
    temporary exist, nothing is finalized. The next invocation takes the invocation-1 slot, keeps and counts the
    temporary, reuses the existing preflight record only after the section-5.4.1 checks, and grants nothing extra."""
    root = tmp_path / p5a.P5A_RUN_NAME
    _hard_kill(root, "during_first_record_write")
    store = ArtifactStore(root)
    temps = [f for f in store.files() if f.endswith(TMP_SUFFIX)]
    assert len(temps) == 1 and temps[0].startswith("ledger/.invocation_1.json.")
    assert not store.exists(p5a.invocation_path(1)) and store.files() == sorted(
        [p5a.preflight_path(1), temps[0]]
    )
    tmp_file = root / temps[0]
    tmp_bytes = tmp_file.read_bytes()
    pops = all_pops()
    trained = []
    ctx = context(root, pops, trainer=fake_trainer(lambda xs: trained.append(len(xs)) or exact(xs)))
    summary = run.run_invocation(ctx, loader(pops))
    assert summary["invocation"] == 1 and summary["closed"] and trained == [1, 4, 1, 4, 1, 4]
    inv = _read(store, p5a.invocation_path(1))
    assert inv["entry"]["state"] == "abandoned_first_publication" and inv["entry"]["ledger_existed"] is True
    assert inv["entry"]["retained_temporaries"] == [{"path": temps[0], "bytes": len(tmp_bytes)}]
    assert inv["retained_bytes_before_this_file"] == store.size(p5a.preflight_path(1)) + len(tmp_bytes)
    assert tmp_file.read_bytes() == tmp_bytes  # never deleted, renamed or opened as evidence
    on_disk = _read(store, p5a.summary_path(1))  # still counted at the end of the run
    assert on_disk["retained_bytes_before_this_file"] + store.size(p5a.summary_path(1)) == store.footprint()
    assert store.footprint() == sum((root / f).stat().st_size for f in store.files())
    assert st.resource_state(store)["trainings_started"] == 6  # nothing was granted beyond the six trainings


def test_invocation_one_slot_refuses_started_foreign_or_unknown_states(tmp_path):
    """v2.1 A2: only empty preflight/ledger directories and recognized first-publication or preflight temporaries
    are tolerated. Each refusal publishes, renames and deletes nothing."""
    pops = all_pops()
    tok = "deadbeef0000"
    cases = {
        "start temporary (uncertain start)": f"ledger/.start_R2_single_snapshot.json.{tok}{TMP_SUFFIX}",
        "finalized start record": p5a.start_path("R2", "single_snapshot"),
        "finalized summary": p5a.summary_path(1),
        "control directory file": f"{p5a.control_dir('R2', 'single_snapshot')}/config.resolved.yaml",
        "unknown file": "ledger/notes.txt",
        "temporary of a later invocation record": f"ledger/.invocation_2.json.{tok}{TMP_SUFFIX}",
        "foreign temporary token": f"ledger/.invocation_1.json.someoneelse{TMP_SUFFIX}",
        "unknown top-level file": "README.txt",
    }
    for i, (name, rel) in enumerate(cases.items()):
        ctx, _ = _ready(tmp_path / str(i), pops, trainer=fake_trainer(exact))
        path = ctx.store.path(rel) if not rel.endswith(TMP_SUFFIX) else ctx.store.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"{}")  # valid JSON, so only the A2 rule (not a parse error) can refuse
        before = sorted((p.relative_to(ctx.store.root).as_posix(), p.stat().st_size if p.is_file() else None)
                        for p in ctx.store.root.rglob("*"))  # fmt: skip
        with pytest.raises(p5a.P5AProtocolError, match="invocation-1 slot is refused"):
            run.run_invocation(_again(ctx), loader(pops))
        after = sorted((p.relative_to(ctx.store.root).as_posix(), p.stat().st_size if p.is_file() else None)
                       for p in ctx.store.root.rglob("*"))  # fmt: skip
        assert after == before, name
    # an empty control directory is evidence that training may have started; a nested directory is unknown
    for j, rel in enumerate((p5a.control_dir("R3", "fixed_subset4"), "ledger/sub")):
        ctx, _ = _ready(tmp_path / f"d{j}", pops)
        (ctx.store.root / rel).mkdir(parents=True)
        with pytest.raises(p5a.P5AProtocolError, match="invocation-1 slot is refused"):
            run.run_invocation(_again(ctx), loader(pops))
        assert not ctx.store.exists(p5a.invocation_path(1))
    # a malformed or foreign preflight record is ambiguous provenance
    ctx, _ = _ready(tmp_path / "pf", pops)
    ctx.store.path(p5a.preflight_path(2)).write_text('{"schema": "other"}', encoding="utf-8")
    with pytest.raises(p5a.P5AProtocolError, match="not a P5-A v2 preflight record"):
        run.run_invocation(_again(ctx), loader(pops))


def test_invocation_one_slot_accepts_empty_ledger_and_preflight_temporaries(tmp_path):
    pops = all_pops()
    ctx, _ = _ready(tmp_path, pops, trainer=fake_trainer(exact))
    (ctx.store.root / "ledger").mkdir()
    ptmp = ctx.store.root / f"preflight/.preflight_2.json.abcdef012345{TMP_SUFFIX}"
    ptmp.write_bytes(b"{")  # an interrupted second preflight: recognized, counted, untouched
    summary = run.run_invocation(_again(ctx), loader(pops))
    inv = _read(ctx.store, p5a.invocation_path(1))
    assert summary["invocation"] == 1 and inv["entry"]["state"] == "abandoned_first_publication"
    assert inv["entry"]["retained_temporaries"] == [{"path": ptmp.relative_to(ctx.store.root).as_posix(),
                                                     "bytes": 1}]  # fmt: skip
    assert ptmp.read_bytes() == b"{"
    fresh, _ = _ready(tmp_path / "fresh", pops, trainer=fake_trainer(exact))
    run.run_invocation(fresh, loader(pops))
    assert _read(fresh.store, p5a.invocation_path(1))["entry"] == {
        "state": "fresh", "ledger_existed": False, "retained_temporaries": [],
        "rule": "v2.1 A2: no finalized training-stage file and no evidence that training started"}  # fmt: skip


def test_retained_temporary_can_legitimately_refuse_the_first_record(tmp_path):
    """v2.1 A3: the 14,510,000-byte boundary of F before the first record (cap 50,000 + reservation 5,440,000)."""
    pops = all_pops()
    reserve = 3 * 200_000 + 2 * 50_000 + 6 * (20_000 + 20_000 + 100_000 + 150_000 + 500_000)
    assert reserve == 5_440_000 and 20_000_000 - 50_000 - reserve == 14_510_000
    for over, expect_ok in ((0, True), (1, False)):
        ctx, _ = _ready(tmp_path / str(over), pops, trainer=fake_trainer(exact))
        (ctx.store.root / "ledger").mkdir()
        pad = ctx.store.root / f"ledger/.invocation_1.json.0123456789ab{TMP_SUFFIX}"
        pad.write_bytes(b"\0" * (14_510_000 + over - ctx.store.footprint()))
        assert ctx.store.footprint() == 14_510_000 + over
        if expect_ok:
            assert run.run_invocation(_again(ctx), loader(pops))["invocation"] == 1
        else:
            with pytest.raises(p5a.P5AProtocolError, match="evidence_reservation"):
                run.run_invocation(_again(ctx), loader(pops))
            assert not ctx.store.exists(p5a.invocation_path(1))
            assert pad.stat().st_size == 14_510_001 - ctx.store.size(p5a.preflight_path(1))  # kept, counted
            assert st.resource_state(ctx.store)["trainings_started"] == 0


def test_concurrent_first_record_claims_finalize_exactly_one(tmp_path):
    """v2.1 A2: two claimants that both pass the eligibility check race only at the no-clobber link."""
    pops = all_pops()
    ctx, _ = _ready(tmp_path, pops, trainer=fake_trainer(exact))
    rival = _again(ctx)
    rival.store = ArtifactStore(ctx.store.root)
    real = ctx.store.linker
    raced = []

    def racing(src, dst):  # the rival finalizes its own invocation_1.json between our check and our link
        if dst.name == "invocation_1.json" and not raced:
            raced.append(1)
            rival.store.publish(p5a.invocation_path(1), b'{"rival": true}\n', reserved_other=0)
        real(src, dst)

    ctx.store.linker = racing
    with pytest.raises(p5a.P5AProtocolError, match="capability check failed"):
        run.run_invocation(ctx, loader(pops))
    assert ctx.store.path(p5a.invocation_path(1)).read_bytes() == b'{"rival": true}\n'  # the winner's record
    assert not [
        f for f in ctx.store.files() if f.endswith(f".{ctx.store.token}{TMP_SUFFIX}")
    ]  # own temp removed
    assert not ctx.store.exists(p5a.summary_path(1)) and not ctx.store.exists(
        p5a.start_path("R2", "single_snapshot")
    )


def test_resource_closure_never_relabels_a_trained_control(tmp_path):
    """Section 13.5 closure statuses apply to controls without a terminal record that never trained (sections 7 and
    13.4); a started control keeps its section-12 recovery status even when a resource ceiling closes the run."""
    ctx, pops = _ready(tmp_path, trainer=fake_trainer(exact))
    ctx.store.publish(p5a.invocation_path(1), json_bytes({"identity": ctx.identity,
                                                          "protocol": {"commit": p5a.P5A_PROTOCOL_COMMIT}}),
                      reserved_other=0)  # fmt: skip
    ctx.store.publish(p5a.start_path("R2", "single_snapshot"), b"{}", reserved_other=0)  # a hard-killed start
    ctx.invocation = 2
    issues = run.close_run(ctx, resource_closure=True, pre={})
    assert issues == []
    assert _read(ctx.store, p5a.record_path("R2", "single_snapshot"))["status"] == p5a.INCOMPLETE_INTERRUPTED
    assert _read(ctx.store, p5a.record_path("R2", "fixed_subset4"))["status"] == p5a.INCOMPLETE_RESOURCE
    assert _read(ctx.store, p5a.record_path("R2", "fixed_subset4"))["trained"] is False


@pytest.mark.parametrize("closure_only", [False, True])
@pytest.mark.parametrize("issue", ["summary_publication_error", "closure_issues"])
def test_cli_surfaces_failed_evidence_publication(tmp_path, monkeypatch, closure_only, issue):
    """A missing summary or failed closure record must not reach main's successful exit path."""
    from cg_fedllm import cli

    if issue == "summary_publication_error":
        ctx, pops = _ready(tmp_path, trainer=fake_trainer(exact))
        real_link = ctx.store.linker

        def refuse_summary(src, dst):
            if dst.name == "summary_1.json":
                raise OSError("synthetic summary publication failure")
            real_link(src, dst)

        ctx.store.linker = refuse_summary
        out = run.run_invocation(ctx, loader(pops))
        assert out["closed"] and out[issue]
        assert not ctx.store.exists(p5a.summary_path(1))
    else:
        out = {
            "invocation": 3,
            "closed": True,
            "closure_issues": [{"path": p5a.record_path("R4", "fixed_subset4"), "error": "synthetic"}],
        }
    entry = "production_closure" if closure_only else "production_main"
    monkeypatch.setattr(run, entry, lambda *_a, **_k: out)
    argv = ["p5a", "--config", str(CONFIG)]
    if closure_only:
        argv.append("--closure-only")
    with pytest.raises(p5a.P5AProtocolError, match="evidence publication incomplete") as exc:
        cli.main(argv)
    assert issue in str(exc.value)


@pytest.mark.parametrize(("available", "initialised"), [(True, False), (False, True)])
def test_production_closure_refuses_cuda_before_context_capture(
    tmp_path, monkeypatch, available, initialised
):
    """Refuse a visible GPU or an existing CUDA context before shared imports, environment capture or writes."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: initialised)

    def forbidden_context(*_a, **_k):
        raise AssertionError("CPU closure must refuse before context capture")

    monkeypatch.setattr(run, "_production_context", forbidden_context)
    with pytest.raises(p5a.P5AProtocolError, match="fresh CPU process"):
        run.production_closure(str(CONFIG), [], str(tmp_path), "synthetic closure")
    assert list(tmp_path.iterdir()) == []
