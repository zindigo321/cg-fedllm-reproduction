"""Phase-5 P5-A v2: AutoEncoder fit control for R2-R4 (result label PHASE5-DIAGNOSTIC).

Implements ``docs/phase5_preregistration_v2.md`` (approved for implementation only; committed in
``P5A_PROTOCOL_COMMIT``) with the v2.1 addendum ``docs/phase5_preregistration_v2_1.md`` (``P5A_ADDENDUM_COMMIT``). An outcome is a fit check of the fixed instrument on its own training snapshots under the
stated protocol only (sections 1 and 11). This module holds the frozen protocol values, record ordering and control
selection (section 6), the control statuses and the section-11 outcome records. Metrics and criteria:
:mod:`.p5a_metrics`; input identity and construction: :mod:`.p5a_inputs`; artifact publication and the byte policy:
:mod:`.p5a_artifacts`; the CPU input preflight: :mod:`.p5a_preflight`; training and runner: :mod:`.p5a_run`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

P5A_PROTOCOL_COMMIT = "dd4bf3f4348fd3b2b1b2c4ad23e7204fbef8369c"
P5A_PROTOCOL_PATH = "docs/phase5_preregistration_v2.md"
P5A_PROTOCOL_BLOB = "e80d21aeda653ab6d1339d34f7b88118588855df"  # git blob id of the approved v2 text
# v2.1 addendum (sections A2 and A4 supersede the named v2 text); the protocol identity is the pair (v2, v2.1)
P5A_ADDENDUM_COMMIT = "33e51fd63c6cd0679e76e12380ae5d418158d698"
P5A_ADDENDUM_PATH = "docs/phase5_preregistration_v2_1.md"
P5A_ADDENDUM_BLOB = "28e05f9ee33ed13386d346fce6e47e9108c937d7"
# v1 (52a0dd4) stays the historical approved record; it is never executed and its run root is never created
P5A_V1_COMMIT = "52a0dd4e7bb8521752ac80ff8f77072dc0e0761c"
P5A_V1_RUN_NAME = "phase5_p5a_seed1"
P5A_LABEL = "PHASE5-DIAGNOSTIC"
P5A_RUN_NAME = "phase5_p5a_v2_seed1"
NORMALIZATION_MODE = "global_exact_maxabs_train"

REPRESENTATIONS = ("R2", "R3", "R4")
CONTROLS = ("single_snapshot", "fixed_subset4")
PLAN = tuple((rid, c) for rid in REPRESENTATIONS for c in CONTROLS)  # section 7: fixed order, positions 1-6
CONTROL_SIZE = {"single_snapshot": 1, "fixed_subset4": 4}
DESIGNATION = {"R2": "primary", "R3": "primary", "R4": "secondary"}  # P5v2-D03; never result-dependent

AE_SEED = 1  # section 8.3
ITERATIONS = 3_000
MAX_BATCH = 4  # bs = min(4, k)
CURVE_EVERY = 50

# section 10: literal float64 thresholds; every bound is inclusive and no epsilon is ever added
THRESHOLDS = {
    "single_snapshot": {"rse_max": 0.01, "cosine_min": 0.99},  # memorisation standard (P5v2-D29)
    "fixed_subset4": {"rse_max": 0.50, "cosine_min": 0.90, "subset_mean_factor": 0.8},  # subset fit standard
}
NEAR_THRESHOLD_ULPS = 4  # section 11.4

MAX_TRAININGS = 6  # section 12
TRAINING_LIMIT_S = 1_800.0
AGGREGATE_LIMIT_S = 10_800.0
MAX_INVOCATIONS = 3  # section 13.1
MAX_PREFLIGHTS = 3  # section 5.4.1
SCALE_REL_TOL = 1e-6  # section 5.4: |m_rec - m_frozen| <= 1e-6 * m_frozen
RMS_REL_TOL = 1e-6  # section 3 item 5
RANGE_REL_ALLOWANCE = 2e-6  # section 5.5: on the normalised values around 0.95

FIT_PASS = "FIT PASS"
FIT_FAIL = "FIT FAIL"
NOT_INFORMATIVE = "NOT INFORMATIVE (TANH RANGE)"
INCOMPLETE_PROVENANCE = "INCOMPLETE (PROVENANCE)"
INCOMPLETE_RESOURCE = "INCOMPLETE (RESOURCE STOP)"
INCOMPLETE_NON_FINITE = "INCOMPLETE (NON-FINITE LOSS)"
INCOMPLETE_INTERRUPTED = "INCOMPLETE (INTERRUPTED)"
INCOMPLETE_PUBLICATION = "INCOMPLETE (PUBLICATION)"
INCOMPLETE_NOT_STARTED = "INCOMPLETE (NOT STARTED)"
CONTROL_STATUSES = (
    FIT_PASS,
    FIT_FAIL,
    NOT_INFORMATIVE,
    INCOMPLETE_PROVENANCE,
    INCOMPLETE_RESOURCE,
    INCOMPLETE_NON_FINITE,
    INCOMPLETE_INTERRUPTED,
    INCOMPLETE_PUBLICATION,
    INCOMPLETE_NOT_STARTED,
)
PRE_START_STATUSES = (INCOMPLETE_PROVENANCE, NOT_INFORMATIVE, INCOMPLETE_RESOURCE)
# v2.1 A4.2 S1: a started training that ended without a completed evaluation keeps this status for good
TRAINING_STOP_STATUSES = (INCOMPLETE_NON_FINITE, INCOMPLETE_RESOURCE, INCOMPLETE_INTERRUPTED)
# v2.1 A4.1: the cause of a byte-policy refusal, recorded separately from every control status
STOP_PER_FILE_CAP = "per_file_cap"
STOP_AGGREGATE_CAP = "aggregate_byte_cap"
STOP_RESERVATION = "evidence_reservation"
RESOURCE_STOP_CAUSES = (STOP_PER_FILE_CAP, STOP_AGGREGATE_CAP, STOP_RESERVATION)

DEMONSTRATED = "CAPACITY FIT DEMONSTRATED"
NOT_DEMONSTRATED = "CAPACITY FIT NOT DEMONSTRATED"
INCONCLUSIVE = "INCONCLUSIVE"
QUALIFIER = (
    "under the P5-A v2 protocol (fixed ResNet-3, exact training-population max-abs scaling to 0.95, Adam 2e-4, "
    "3,000 iterations, final checkpoint, AE seed 1, training snapshots only)"
)
STANDARDS = {
    "single_snapshot": "memorisation (RSE <= 0.01, cos >= 0.99)",
    "fixed_subset4": "subset fit (RSE <= 0.50, cos >= 0.90 per snapshot; C4)",
}
MEANING = {  # section 11.3
    DEMONSTRATED: "under this protocol, the fixed AE met the memorisation standard on the one single_snapshot record "
    "and the subset fit standard on the four fixed_subset4 records of this representation's training population",
    NOT_DEMONSTRATED: "under this protocol, at least one completed control did not meet its standard; this does not "
    "show that the representation is incompressible or that the AE has insufficient capacity",
    INCONCLUSIVE: "no completed control failed, but at least one control did not complete or was not informative; "
    "this is not evidence either way",
}
DOES_NOT_ESTABLISH = (  # section 1, fixed list
    "compressibility or incompressibility of the representation at 1/64",
    "sufficient or insufficient AE capacity in general",
    "generalisation to unseen snapshots",
    "stability across seeds",
    "the cause of the F6 failures",
    "anything about FAF, operational server semantics or the paper's compression claims",
)
AUTHORISES = {
    DEMONSTRATED: "only drafting a separate scale-normalised re-screen preregistration for this representation",
    NOT_DEMONSTRATED: "nothing",
    INCONCLUSIVE: "nothing",
}
KIND_DIR = {"R2": "balanced_effective_state", "R3": "balanced_effective_delta_r8", "R4": "mean_step_gradient"}


LAYOUT_ID = "layer_major_qkvo_AtB"
# section 8.3: every AE / run field the P5-A protocol fixes (the production routes refuse any other value)
AE_LOCK = {
    "arch": "resnet3",
    "stem_kernel": 3,
    "stem_channels": 1,
    "down_channels": [2, 4, 8, 16, 32, 64],
    "num_res_blocks": 3,
    "final_kernel": 7,
    "norm": "batch",
    "output_activation": "tanh",
    "layout": LAYOUT_ID,
    "batch_size": MAX_BATCH,
    "iterations": ITERATIONS,
    "learning_rate": 2e-4,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_epsilon": 1e-8,
    "weight_decay": 0.0,
    "init_seed": AE_SEED,
    "eval_every": CURVE_EVERY,
    "checkpoint_policy": "final",
    "normalization": NORMALIZATION_MODE,
}
RUN_LOCK = {
    "name": P5A_RUN_NAME,
    "device": "cuda",
    "deterministic": True,
    "result_label": P5A_LABEL,
    "allocator_cap_margin_mb": 256,
    "num_threads": None,  # the configured value; preflight and training apply the same (v2.1 A5 P1)
}
LORA_LOCK = {"r": 8, "alpha": 16}
P5A_CONFIG = "configs/phase5/p5a_v2_seed1.yaml"


class P5AProtocolError(RuntimeError):
    """A provenance or input check failed; the affected controls are ``INCOMPLETE (PROVENANCE)``, not a result."""


def check_config(cfg: Any) -> None:
    """Refuse a configuration that differs from the frozen P5-A v2 protocol in any locked field."""
    bad = {}
    for section, lock in (("autoencoder", AE_LOCK), ("run", RUN_LOCK), ("lora", LORA_LOCK)):
        sec = getattr(cfg, section, None)
        for key, value in lock.items():
            got = getattr(sec, key, None) if sec is not None else None
            if got != value or type(got) is bool and type(value) is not bool:
                bad[f"{section}.{key}"] = {"expected": value, "actual": got}
    if bad:
        raise P5AProtocolError(f"configuration differs from the P5-A v2 protocol: {bad}")


# ---- run-root layout (section 13) -------------------------------------------------------------------------
def control_dir(rid: str, control: str) -> str:
    return f"{rid}_{KIND_DIR[rid]}_{control}"


def start_path(rid: str, control: str) -> str:
    return f"ledger/start_{rid}_{control}.json"


def end_path(rid: str, control: str) -> str:
    return f"ledger/end_{rid}_{control}.json"


def failure_path(rid: str, control: str) -> str:
    return f"ledger/failure_{rid}_{control}.json"


def record_path(rid: str, control: str) -> str:
    return f"{control_dir(rid, control)}/p5a_record.json"


def checkpoint_path(rid: str, control: str) -> str:
    return f"{control_dir(rid, control)}/autoencoder_p5a_final.safetensors"


def invocation_path(k: int) -> str:
    return f"ledger/invocation_{k}.json"


def summary_path(k: int) -> str:
    return f"ledger/summary_{k}.json"


def preflight_path(k: int) -> str:
    return f"preflight/preflight_{k}.json"


def protocol_identity() -> dict[str, Any]:
    """The protocol identity every P5-A record carries: the base v2 commit plus the v2.1 addendum (v2.1 A1)."""
    return {
        "commit": P5A_PROTOCOL_COMMIT,
        "path": P5A_PROTOCOL_PATH,
        "blob": P5A_PROTOCOL_BLOB,
        "addendum": {"commit": P5A_ADDENDUM_COMMIT, "path": P5A_ADDENDUM_PATH, "blob": P5A_ADDENDUM_BLOB},
    }


def record_key(record: Mapping[str, Any]) -> tuple[int, int]:
    return int(record["time_index"]), int(record["client_id"])


def sort_records(records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """``E``: ascending numeric ``(time_index, client_id)``; a duplicate key aborts (no tie-break, section 6)."""
    keys = [record_key(r) for r in records]
    if len(set(keys)) != len(keys):
        dup = sorted({k for k in keys if keys.count(k) > 1})
        raise P5AProtocolError(f"(time_index, client_id) is not unique: {dup}")
    return sorted(records, key=record_key)


def round_half_up_ratio(num: int, den: int) -> int:
    """``floor(num / den + 0.5)`` in exact integer arithmetic for non-negative integers."""
    return (2 * num + den) // (2 * den)


def control_positions(control: str, n: int) -> list[int]:
    """0-based positions in ``E`` (section 6): the median position, or four evenly spaced positions."""
    if control == "single_snapshot":
        positions = [(n - 1) // 2]
    elif control == "fixed_subset4":
        positions = [round_half_up_ratio(j * (n - 1), 3) for j in range(4)]
    else:
        raise ValueError(f"unknown P5-A control {control!r}")
    if n < 1 or len(set(positions)) != CONTROL_SIZE[control] or not 0 <= min(positions) <= max(positions) < n:
        raise P5AProtocolError(f"{control}: N = {n} does not give {CONTROL_SIZE[control]} distinct positions")
    return positions


def select_control(sorted_records: Sequence[Mapping[str, Any]], control: str) -> list[dict[str, Any]]:
    """Metadata-only selection: ``[{"position", "time_index", "client_id"}]`` in position order."""
    return [
        {
            "position": p,
            "time_index": record_key(sorted_records[p])[0],
            "client_id": record_key(sorted_records[p])[1],
        }
        for p in control_positions(control, len(sorted_records))
    ]


def representation_outcome(single: str, subset: str) -> str:
    """Section 11.2 table over the nine statuses: one representation, never aggregated across representations."""
    for status in (single, subset):
        if status not in CONTROL_STATUSES:
            raise ValueError(f"not a P5-A control status: {status!r}")
    if single == FIT_PASS and subset == FIT_PASS:
        return DEMONSTRATED
    if FIT_FAIL in (single, subset):
        return NOT_DEMONSTRATED
    return INCONCLUSIVE


def outcome_record(
    representation: str, statuses: Mapping[str, str], criteria: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Section 11.4 structured outcome of one representation, computed only at run closure."""
    outcome = representation_outcome(statuses["single_snapshot"], statuses["fixed_subset4"])
    return {
        "representation": representation,
        "designation": DESIGNATION[representation],
        "controls": {c: statuses[c] for c in CONTROLS},
        "outcome": outcome,
        "meaning": MEANING[outcome],
        "qualifier": QUALIFIER,
        "criteria": {c: (criteria or {}).get(c) for c in CONTROLS},
        "standards": dict(STANDARDS),
        "does_not_establish": list(DOES_NOT_ESTABLISH),
        "authorises": AUTHORISES[outcome],
    }
