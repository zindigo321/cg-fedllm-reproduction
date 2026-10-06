"""P5-A v2 persistent attempt states, time accounting and byte reservation (protocol sections 12-13).

Every value here is derived only from finalized files under the run root, so a later invocation reconstructs the
same state after a process death. Nothing here reads a payload, builds an AE or touches CUDA.
"""

from __future__ import annotations

import math
from collections.abc import Collection
from typing import Any

from cg_fedllm.phase5 import p5a
from cg_fedllm.phase5.p5a_artifacts import CAPS, PROVENANCE_FILES, ArtifactStore, provenance_set_bytes

NOT_STARTED, STARTED, ENDED, RECORDED = "NOT_STARTED", "STARTED", "ENDED", "RECORDED"
CHARGE_MEASURED = "measured"
CHARGE_UNOBSERVED = "conservative_full_allocation_after_unobserved_process_death"


def control_state(store: ArtifactStore, rid: str, control: str) -> str:
    """Section 12: RECORDED (terminal record) > ENDED (end record) > STARTED (start record) > NOT_STARTED."""
    if store.exists(p5a.record_path(rid, control)):
        return RECORDED
    if store.exists(p5a.end_path(rid, control)):
        return ENDED
    if store.exists(p5a.start_path(rid, control)):
        return STARTED
    return NOT_STARTED


def states(store: ArtifactStore) -> dict[tuple[str, str], str]:
    return {(rid, c): control_state(store, rid, c) for rid, c in p5a.PLAN}


def anomalous_files(store: ArtifactStore, rid: str, control: str) -> list[str]:
    """Finalized files of a ``NOT_STARTED`` control (section 7: an anomalous state, ``INCOMPLETE (PROVENANCE)``)."""
    d = p5a.control_dir(rid, control)
    rels = [p5a.end_path(rid, control), p5a.failure_path(rid, control), p5a.record_path(rid, control),
            p5a.checkpoint_path(rid, control)] + [f"{d}/{n}" for n in PROVENANCE_FILES]  # fmt: skip
    return [r for r in rels if store.exists(r)]


def timing(measured_s: float | None) -> dict[str, Any]:
    """Section 12 timing fields: a measured time is charged as measured; an unknown one is charged 1,800 s."""
    if measured_s is not None and math.isfinite(measured_s) and measured_s >= 0:
        return {"gpu_wall_s_measured": float(measured_s), "timing_status": "measured",
                "gpu_wall_s_charged": float(measured_s), "charge_reason": CHARGE_MEASURED}  # fmt: skip
    return {"gpu_wall_s_measured": None, "timing_status": "unknown_process_death",
            "gpu_wall_s_charged": p5a.TRAINING_LIMIT_S, "charge_reason": CHARGE_UNOBSERVED}  # fmt: skip


def charged_s(store: ArtifactStore, rid: str, control: str) -> float:
    """Charged GPU time of a started control: its end record, else its control record, else the full 1,800 s."""
    for rel in (p5a.end_path(rid, control), p5a.record_path(rid, control)):
        doc = store.read_json(rel)
        t = (doc or {}).get("timing") or {}
        v = t.get("gpu_wall_s_charged")
        if isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v) and v >= 0:
            return float(v)
    return p5a.TRAINING_LIMIT_S


def started(store: ArtifactStore) -> list[tuple[str, str]]:
    return [(rid, c) for rid, c in p5a.PLAN if store.exists(p5a.start_path(rid, c))]


def resource_state(store: ArtifactStore) -> dict[str, Any]:
    run = started(store)
    charged = math.fsum(charged_s(store, rid, c) for rid, c in run)
    return {
        "trainings_started": len(run),
        "gpu_wall_s_charged_total": charged,
        "may_start": len(run) < p5a.MAX_TRAININGS and charged + p5a.TRAINING_LIMIT_S <= p5a.AGGREGATE_LIMIT_S,
        "limits": {
            "max_trainings": p5a.MAX_TRAININGS,
            "training_limit_s": p5a.TRAINING_LIMIT_S,
            "aggregate_limit_s": p5a.AGGREGATE_LIMIT_S,
            "retained_bytes": store.limit,
        },  # fmt: skip
    }


def reserved_caps(
    store: ArtifactStore,
    *,
    invocation: int,
    exclude: Collection[str] = (),
    checkpoint_for: tuple[str, str] | None = None,
) -> int:
    """``sum(caps(Q))`` of section 13.3 during training invocation ``invocation`` (``exclude``: the file about to be
    published, whose cap the store adds itself). Already finalized files are never reserved again."""
    total = 0

    def add(rel: str, kind: str) -> None:
        nonlocal total
        if rel not in exclude and not store.exists(rel):
            total += CAPS[kind][0]

    for j in range(invocation, p5a.MAX_INVOCATIONS + 1):
        add(p5a.summary_path(j), "summary")
        if j > invocation:
            add(p5a.invocation_path(j), "invocation")
    for rid, c in p5a.PLAN:
        if control_state(store, rid, c) == RECORDED:
            continue
        for rel, kind in ((p5a.start_path(rid, c), "start"), (p5a.end_path(rid, c), "end"),
                          (p5a.failure_path(rid, c), "failure"), (p5a.record_path(rid, c), "record")):  # fmt: skip
            add(rel, kind)
        d = p5a.control_dir(rid, c)
        if not any(f"{d}/{n}" in exclude for n in PROVENANCE_FILES):
            total += max(0, CAPS["provenance_set"][0] - provenance_set_bytes(store, d))
        if checkpoint_for == (rid, c):
            add(p5a.checkpoint_path(rid, c), "checkpoint")
    return total
