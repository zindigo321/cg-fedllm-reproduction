"""P5-A v2 CPU-only input preflight (section 5.4.1) on synthetic inputs and injected readers only."""

from __future__ import annotations

import dataclasses
import json

import pytest
import torch

from cg_fedllm.phase5 import p5a
from cg_fedllm.phase5 import p5a_preflight as pf
from cg_fedllm.phase5.p5a_artifacts import ArtifactStore
from tests.unit.p5a_fixtures import IDENTITY, all_pops, frozen_for, loader


def _frozen(pops):
    return {rid: frozen_for(rid, xs) for rid, xs in pops.items()}


def _run(tmp_path, pops, frozen=None, identity=IDENTITY):
    store = ArtifactStore(tmp_path / p5a.P5A_RUN_NAME)
    return store, pf.run_preflight(
        store, identity=identity, load=loader(pops), frozen=frozen or _frozen(pops)
    )


def test_preflight_builds_no_ae_starts_nothing_and_touches_no_cuda(tmp_path, monkeypatch):
    import cg_fedllm.compression.autoencoder as aem

    def no_ae(*a, **k):
        raise AssertionError("the preflight must not build an AE")

    def no_cuda(*a, **k):
        raise AssertionError("the preflight must not initialise CUDA")

    monkeypatch.setattr(aem.ResNetAutoEncoder, "__init__", no_ae)
    monkeypatch.setattr(torch.cuda, "init", no_cuda)
    monkeypatch.setattr(torch.cuda, "_lazy_init", no_cuda)
    before = torch.cuda.is_initialized()  # other tests in the same process may already have initialised CUDA
    pops = all_pops()
    assert pf.cpu_environment()["torch"] == torch.__version__  # reading the fingerprint initialises no CUDA
    store, rec = _run(tmp_path, pops)
    assert rec["pass"] and rec["trainings_started"] == 0 and rec["ae_built"] is False
    assert rec["cuda_initialized"] is before and torch.cuda.is_initialized() is before  # reported truthfully
    assert store.files() == [p5a.preflight_path(1)]  # no ledger, no start record, no control directory
    on_disk = json.loads(store.path(p5a.preflight_path(1)).read_text(encoding="utf-8"))
    assert on_disk["protocol"]["commit"] == p5a.P5A_PROTOCOL_COMMIT and on_disk["config_sha256"] == "f" * 64
    r4 = on_disk["representations"][2]
    assert r4["representation"] == "R4" and r4["designation"] == "secondary"
    assert (
        r4["scale_check"]["bitwise_equal"]
        and r4["evidence"]["frozen_scale"] == r4["evidence"]["frozen_max_abs"] / 0.95
    )


def test_a_mismatch_is_recorded_and_substitutes_nothing(tmp_path):
    pops = all_pops()
    frozen = _frozen(pops)
    m = frozen["R3"].max_abs * (1 + 1.1e-6)
    frozen["R3"] = dataclasses.replace(frozen["R3"], max_abs=m, scale=m / 0.95)
    store, rec = _run(tmp_path, pops, frozen)
    rows = {r["representation"]: r for r in rec["representations"]}
    assert not rec["pass"] and rows["R2"]["pass"] and rows["R4"]["pass"] and not rows["R3"]["pass"]
    assert "not replaced" in rows["R3"]["failure"] and "evidence" not in rows["R3"]
    assert frozen["R3"].max_abs == m  # the frozen value was not replaced


def test_preflight_root_rules_and_bounded_repetition(tmp_path):
    pops = all_pops()
    store, _ = _run(tmp_path, pops)
    for k in (2, 3):
        _, rec = _run(tmp_path, pops)
        assert rec["preflight"] == k
    with pytest.raises(p5a.P5AProtocolError, match="at most 3"):
        _run(tmp_path, pops)
    other = tmp_path / "other"
    (other / p5a.P5A_RUN_NAME / "ledger").mkdir(parents=True)
    (other / p5a.P5A_RUN_NAME / "ledger" / "invocation_1.json").write_text("{}", encoding="utf-8")
    with pytest.raises(p5a.P5AProtocolError, match="other than preflight"):
        _run(other, pops)


def test_training_validation_refuses_missing_failed_stale_or_changed_evidence(tmp_path):
    pops = all_pops()
    store, rec = _run(tmp_path, pops)
    ev = {r["representation"]: r["evidence"] for r in rec["representations"]}
    assert pf.validate_against_preflight(rec, IDENTITY, "R2", ev["R2"]) is None
    assert "no preflight" in pf.validate_against_preflight(None, IDENTITY, "R2", ev["R2"])
    for key in ("execution_commit", "config_sha256"):
        stale = {**IDENTITY, key: "0" * 40}
        assert "stale" in pf.validate_against_preflight(rec, stale, "R2", ev["R2"])
    assert "protocol" in pf.validate_against_preflight({**rec, "protocol": {"commit": p5a.P5A_V1_COMMIT}},
                                                       IDENTITY, "R2", ev["R2"])  # fmt: skip
    changed = json.loads(json.dumps(ev["R2"]))
    changed["records"][0]["phi_sha256"] = "0" * 64
    assert "differ" in pf.validate_against_preflight(rec, IDENTITY, "R2", changed)
    failed = {**rec, "representations": [{"representation": "R2", "pass": False, "failure": "x"}]}
    assert "did not pass" in pf.validate_against_preflight(failed, IDENTITY, "R2", ev["R2"])
    # v2.1 A1: the protocol identity is the pair (v2, v2.1); a v2-only record is stale
    v2_only = {**rec, "protocol": {k: v for k, v in rec["protocol"].items() if k != "addendum"}}
    assert "protocol identity" in pf.validate_against_preflight(v2_only, IDENTITY, "R2", ev["R2"])
    # N3: exact agreement, not allclose: one ulp in the recomputed RMS refuses
    import math

    nudged = json.loads(json.dumps(ev["R2"]))
    nudged["train_rms_recomputed"] = math.nextafter(nudged["train_rms_recomputed"], 1.0)
    assert "differ" in pf.validate_against_preflight(rec, IDENTITY, "R2", nudged)


def test_cpu_environment_fingerprint_must_match_exactly(tmp_path, monkeypatch):
    """v2.1 A5 P1: the preflight's CPU preprocessing environment is part of its identity; any difference is stale."""
    env = pf.cpu_environment()
    assert (
        set(env["env"]) == set(pf.ENV_VARS) and env["deterministic_algorithms"] is True
    )  # conftest settings
    assert env["deterministic_warn_only"] is False and env["torch_num_threads"] == 2
    assert env["default_dtype"] == "torch.float32" and env["float32_matmul_precision"] == "highest"
    pops = all_pops()
    ident = {**IDENTITY, "cpu_environment": env}
    store, rec = _run(tmp_path, pops, identity=ident)
    ev = {r["representation"]: r["evidence"] for r in rec["representations"]}
    assert pf.validate_against_preflight(rec, ident, "R2", ev["R2"]) is None
    for change in ({"torch_num_threads": 3}, {"env": {**env["env"], "OMP_NUM_THREADS": "4"}},
                   {"deterministic_warn_only": True}, {"numpy": "0.0"}):  # fmt: skip
        other = {**ident, "cpu_environment": {**env, **change}}
        assert "stale preflight: cpu_environment" in pf.validate_against_preflight(rec, other, "R2", ev["R2"])


def test_training_uses_only_the_latest_preflight_and_never_falls_back(tmp_path):
    """N2: an earlier passing record does not rescue a later failing one."""
    pops = all_pops()
    frozen = _frozen(pops)
    store, first = _run(tmp_path, pops, frozen)
    assert first["pass"]
    bad = dict(frozen)
    m = frozen["R2"].max_abs * 2
    bad["R2"] = dataclasses.replace(frozen["R2"], max_abs=m, scale=m / 0.95)
    _, second = _run(tmp_path, pops, bad)
    assert not second["pass"] and pf.latest_preflight(store)["preflight"] == 2
    ev = {r["representation"]: r["evidence"] for r in first["representations"]}
    why = pf.validate_against_preflight(pf.latest_preflight(store), IDENTITY, "R2", ev["R2"])
    assert why and "did not pass for R2" in why


def test_production_preflight_refuses_overrides_and_the_v1_root(tmp_path):
    with pytest.raises(p5a.P5AProtocolError, match="overrides"):
        pf.production_preflight("configs/phase5/p5a_v2_seed1.yaml", ["run.seed=2"], str(tmp_path), "x")
    (tmp_path / p5a.P5A_V1_RUN_NAME).mkdir()
    with pytest.raises(p5a.P5AProtocolError, match="v1"):
        pf.production_preflight("configs/phase5/p5a_v2_seed1.yaml", [], str(tmp_path), "x")
    assert sorted(p.name for p in tmp_path.iterdir()) == [p5a.P5A_V1_RUN_NAME]
    with pytest.raises(p5a.P5AProtocolError, match="launch record not found"):
        (tmp_path / p5a.P5A_V1_RUN_NAME).rmdir()
        pf.production_preflight(
            "configs/phase5/p5a_v2_seed1.yaml", [], str(tmp_path), "x", str(tmp_path / "no.md")
        )


def test_preflight_module_does_not_import_the_trainer():
    import ast
    from pathlib import Path

    src = Path(pf.__file__).read_text(encoding="utf-8")
    names = {a.name for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Import | ast.ImportFrom)
             for a in n.names} | {n.module for n in ast.walk(ast.parse(src)) if isinstance(n, ast.ImportFrom)}  # fmt: skip
    assert not any(
        "autoencoder" in (m or "") or "p5a_run" in (m or "") or "codecs" in (m or "") for m in names
    )


def test_production_preflight_route_on_an_empty_synthetic_root(tmp_path, monkeypatch):
    """The real CLI route end to end on an empty temporary CGFED_RUNS: no frozen input exists there, so every
    representation fails closed; a failing record is still published; no AE, CUDA or training is involved."""
    import cg_fedllm.compression.autoencoder as aem
    from cg_fedllm.cli import main

    def forbidden(*a, **k):
        raise AssertionError("not allowed in the preflight")

    monkeypatch.setattr(aem.ResNetAutoEncoder, "__init__", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    assert (
        main(["p5a-preflight", "--config", "configs/phase5/p5a_v2_seed1.yaml", "--runs-root", str(tmp_path)])
        == 0
    )
    store = ArtifactStore(tmp_path / p5a.P5A_RUN_NAME)
    assert store.files() == [p5a.preflight_path(1)]
    rec = store.read_json(p5a.preflight_path(1))
    assert rec["pass"] is False and rec["trainings_started"] == 0 and rec["ae_built"] is False
    assert [r["representation"] for r in rec["representations"]] == ["R2", "R3", "R4"]
    assert all("index missing" in r["failure"] for r in rec["representations"])
    assert rec["config_sha256"] == "e5ac614c5f176f39b40ddd96cfe311f50c8209151578a1535794cd55d87bb122"
    assert rec["launch_record"] is None and rec["protocol"] == p5a.protocol_identity()
    assert torch.are_deterministic_algorithms_enabled()  # the same settings as training
    env = rec["cpu_environment"]  # v2.1 A5 P1: recorded in full, from the locked configuration's settings
    assert env == pf.cpu_environment() and env["deterministic_algorithms"] is True
    assert env["deterministic_warn_only"] is False and set(env["env"]) == set(pf.ENV_VARS)
