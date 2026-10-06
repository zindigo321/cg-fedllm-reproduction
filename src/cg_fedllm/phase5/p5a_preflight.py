"""P5-A v2 CPU-only input preflight (protocol section 5.4.1, approved choice 6).

Verifies the frozen protocol/config identity, the index and payload hashes, the training populations, geometry and
layout, the selected identities, the exact maxima and frozen scales, finiteness and the normalised range, and
publishes one bounded, provenance-bearing record ``preflight/preflight_<k>.json`` under the run root. It builds no
AE, allocates no CUDA context, starts no training and writes no start record. It opens only the payloads that
:func:`cg_fedllm.phase5.p5a_inputs.build_population` opens (training records; R4 allowlist only). A mismatch is
recorded and substitutes nothing.

A passing preflight is never a bypass: every training invocation repeats every input check itself and compares its
evidence with the most recent preflight record (:func:`validate_against_preflight`).

This module does not and cannot verify a human authorisation; the launch record is the only authorisation.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from cg_fedllm.phase5 import p5a
from cg_fedllm.phase5.p5a_artifacts import ArtifactStore, json_bytes, training_worst_case_bytes
from cg_fedllm.phase5.p5a_inputs import FROZEN, FrozenInputs, Population, evidence, prepare_representation

PREFLIGHT_SCHEMA = "cg_fedllm.p5a_v2_preflight/v1"
# v2.1 A5 P1: the environment variables that can change CPU reductions or thread use, recorded present or absent
ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "MKL_DYNAMIC",
    "OMP_DYNAMIC",
    "MKL_CBWR",
    "KMP_AFFINITY",
    "KMP_BLOCKTIME",
    "CUBLAS_WORKSPACE_CONFIG",
)


def _utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def cpu_environment() -> dict[str, Any]:
    """v2.1 A5 P1: the CPU preprocessing environment, read after ``configure_determinism``. The preflight records it
    and every training invocation must reproduce it exactly before it rebuilds the inputs. Reads flags and versions
    only; initialises no CUDA context."""
    import os
    import sys
    from importlib import metadata

    import numpy
    import torch

    return {
        "interpreter": sys.executable,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "torch_cuda_build": torch.version.cuda,
        "numpy": numpy.__version__,
        "safetensors": metadata.version("safetensors"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "default_dtype": str(torch.get_default_dtype()),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "env": {k: os.environ.get(k) for k in ENV_VARS},
    }


def existing_preflights(store: ArtifactStore) -> list[int]:
    return [k for k in range(1, p5a.MAX_PREFLIGHTS + 1) if store.exists(p5a.preflight_path(k))]


def check_root_for_preflight(store: ArtifactStore) -> int:
    """Section 13.1: preflight k requires an absent root (k = 1) or a root holding only earlier preflight records."""
    if not store.root.exists():
        return 1
    files = store.files()
    done = existing_preflights(store)
    others = [f for f in files if not any(f == p5a.preflight_path(k) for k in done)]
    if others:
        raise p5a.P5AProtocolError(f"run root holds files other than preflight records: {others[:10]}")
    if not done:
        raise p5a.P5AProtocolError("run root exists but holds no preflight record (unexpected state)")
    if done != list(range(1, len(done) + 1)):
        raise p5a.P5AProtocolError(f"preflight records are not consecutive: {done}")
    if len(done) >= p5a.MAX_PREFLIGHTS:
        raise p5a.P5AProtocolError(f"at most {p5a.MAX_PREFLIGHTS} preflights are permitted")
    return len(done) + 1


def assess_representation(frozen: FrozenInputs, load: Callable[[FrozenInputs], Population]) -> dict[str, Any]:
    """Every input check of sections 3, 5.4-5.6 and 6 for one representation; never raises on a protocol mismatch."""
    try:
        rs = prepare_representation(frozen, load(frozen))
    except p5a.P5AProtocolError as exc:
        return {"representation": frozen.rid, "pass": False, "failure": str(exc)}
    ev = evidence(rs)
    sig_zero = {c: e for c, e in ev["selected_sig_zero"].items() if e}
    return {
        "representation": frozen.rid,
        "designation": p5a.DESIGNATION[frozen.rid],
        "pass": True,
        "evidence": ev,
        "scale_check": rs.scale_check,
        "rms_check": rs.rms_check,
        "controls_with_sig_zero": sig_zero,  # those controls become INCOMPLETE (PROVENANCE) at training time
    }


def run_preflight(
    store: ArtifactStore,
    *,
    identity: Mapping[str, Any],
    load: Callable[[FrozenInputs], Population],
    frozen: Mapping[str, FrozenInputs] = FROZEN,
) -> dict[str, Any]:
    """Assess R2, R3, R4 in plan order and publish ``preflight/preflight_<k>.json``. ``identity`` holds the protocol,
    execution, configuration and launch-record identities (recorded, not interpreted as an authorisation)."""
    k = check_root_for_preflight(store)
    reps = [assess_representation(frozen[rid], load) for rid in p5a.REPRESENTATIONS]
    record = {
        "schema": PREFLIGHT_SCHEMA,
        "label": p5a.P5A_LABEL,
        "preflight": k,
        "protocol": p5a.protocol_identity(),
        **dict(identity),
        "created_utc": _utc(),
        "representations": reps,
        "pass": all(r["pass"] for r in reps),
        "trainings_started": 0,
        "ae_built": False,
        "cuda_initialized": _cuda_initialized(),
        "retained_bytes_before_this_file": store.footprint(),
        "note": "input integrity evidence only; no AE was built and nothing was trained (section 5.4.1)",
    }
    store.publish(p5a.preflight_path(k), json_bytes(record), reserved_other=training_worst_case_bytes())
    return record


def _cuda_initialized() -> bool:
    """True only if some code initialised CUDA in this process; the preflight never does."""
    import torch

    return bool(torch.cuda.is_initialized())


def latest_preflight(store: ArtifactStore) -> dict[str, Any] | None:
    done = existing_preflights(store)
    return store.read_json(p5a.preflight_path(done[-1])) if done else None


def validate_against_preflight(
    preflight: Mapping[str, Any] | None, identity: Mapping[str, Any], rid: str, ev: Mapping[str, Any]
) -> str | None:
    """Section 5.4.1: ``None`` if representation ``rid`` may train on evidence ``ev`` of this invocation, otherwise
    the reason it is ``INCOMPLETE (PROVENANCE)`` (missing, failed, stale or disagreeing preflight evidence)."""
    if preflight is None:
        return "no preflight record exists (section 5.4.1)"
    if preflight.get("schema") != PREFLIGHT_SCHEMA:
        return "the preflight record has an unknown schema"
    if preflight.get("protocol") != p5a.protocol_identity():  # v2 commit and v2.1 addendum (v2.1 A1)
        return "stale preflight: different protocol identity (v2 commit and v2.1 addendum)"
    for key in ("execution_commit", "config_sha256", "cpu_environment"):  # cpu_environment: v2.1 A5 P1
        if preflight.get(key) != identity.get(key):
            return f"stale preflight: {key} {preflight.get(key)!r} != {identity.get(key)!r}"
    rows = {r.get("representation"): r for r in preflight.get("representations") or []}
    row = rows.get(rid)
    if row is None or not row.get("pass"):
        return f"the preflight did not pass for {rid}: {(row or {}).get('failure')}"
    if row.get("evidence") != _jsonable(ev):
        return f"the inputs of {rid} differ from the preflight evidence"
    return None


def _jsonable(obj: Any) -> Any:
    """``ev`` as it reads back from the JSON record (tuples become lists), for an exact comparison."""
    import json

    return json.loads(json_bytes(obj))


def production_preflight(
    config_path: str,
    overrides: list[str],
    runs_root: str | None,
    command: str,
    launch_record: str | None = None,
) -> dict:
    """``cgfed p5a-preflight``: the CPU-only preflight on the frozen inputs. Real execution requires the reviewed
    launch record and a separate written authorisation; this function cannot check either."""
    from cg_fedllm.config import load_config, resolve_path
    from cg_fedllm.phase5.p5a_inputs import build_population
    from cg_fedllm.utils.provenance import git_info
    from cg_fedllm.utils.seeding import configure_determinism

    if overrides:
        raise p5a.P5AProtocolError(f"P5-A accepts no --set overrides, got {list(overrides)}")
    cfg = load_config(config_path)
    p5a.check_config(cfg)
    # the same settings as the training invocation (from the locked configuration), which rebuilds the inputs and
    # must reproduce this evidence exactly (section 5.4.1, v2.1 A5 P1); sets flags only, no CUDA context
    configure_determinism(cfg.run.deterministic, cfg.run.num_threads)
    runs = Path(runs_root) if runs_root else resolve_path("${CGFED_RUNS}")
    if not runs.is_absolute() or "${" in str(runs) or not runs.is_dir():
        raise p5a.P5AProtocolError(f"CGFED_RUNS must name an existing absolute directory, got {runs}")
    if (runs / p5a.P5A_V1_RUN_NAME).exists():
        raise p5a.P5AProtocolError(f"the v1 run root {p5a.P5A_V1_RUN_NAME}/ exists; v1 is never executed")
    launch = None
    if launch_record:  # cited by path and SHA-256; not, and cannot be, an authorisation check
        from cg_fedllm.utils.hashing import sha256_file

        lp = Path(launch_record)
        if not lp.is_file():
            raise p5a.P5AProtocolError(f"launch record not found: {lp}")
        launch = {"path": lp.as_posix(), "sha256": sha256_file(lp)}
    git = git_info()
    identity = {
        "execution_commit": git.get("commit"),
        "execution_git": git,
        "config_path": config_path,
        "config_sha256": cfg.sha256(),
        "launch_record": launch,
        "command": command,
        "cpu_environment": cpu_environment(),
    }
    store = ArtifactStore(runs / p5a.P5A_RUN_NAME)
    return run_preflight(store, identity=identity, load=lambda fz: build_population(fz, runs))
