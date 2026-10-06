"""P5-A v2 frozen inputs, identity checks, AE-input construction and the frozen scale (protocol sections 3-5).

Fail closed: every check runs before any training, and any mismatch raises :class:`P5AProtocolError`
(``INCOMPLETE (PROVENANCE)``). The training population is chosen from index metadata before any payload is opened;
validation-split payloads are never opened or hashed.

R4 never calls :func:`cg_fedllm.forensics.gradients.load_client_round`, which also opens the ``*_grad_clipped`` and
``*_delta`` files, and never opens its start/end state files (its input depends only on the gradients).
:class:`AllowlistedGradientReader` opens only the 14 frozen files, verifies each SHA-256 from the bytes it read, and
deserialises those same verified bytes.

Normalisation is ``global_exact_maxabs_train`` with the frozen ``m_r`` / ``s_r`` of section 5.3: the exact maximum
is recomputed and must agree to ``|rec - frozen| <= 1e-6 * frozen``; the **frozen** scale is applied, never a
recomputed one.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import torch

from cg_fedllm.compression.layout import Layout, LoRAGeometry, geometry_from_dict, get_layout
from cg_fedllm.compression.normalization import MAXABS_TARGET, Normalizer, train_abs_max
from cg_fedllm.forensics.gradients import mean_state
from cg_fedllm.forensics.representations import balanced_effective_delta, balanced_effective_state
from cg_fedllm.models.adapter import AdapterState
from cg_fedllm.phase5.p5a import (
    LAYOUT_ID,
    NORMALIZATION_MODE,
    RANGE_REL_ALLOWANCE,
    RMS_REL_TOL,
    SCALE_REL_TOL,
    P5AProtocolError,
    record_key,
    sort_records,
)
from cg_fedllm.tgap.snapshots import SnapshotError, load_states, read_index
from cg_fedllm.utils.hashing import sha256_file, tensor_sha256

PHASE3_ROOT = "phase3_tierb_qwen15_1p8b_seed1/tgap_federated_state"
F5_ROOT = "phase4_gradient_forensics_seed1/gradient_forensics"
INDEX_SHA256 = {
    PHASE3_ROOT: "b6834e476f21f3d6b07f6489b5113576e566ab3987a2f4593405664476d07240",
    F5_ROOT: "560cb0f8153fc0a44e2910ae119ebdccdf228271f963969ff4352401833af9b2",
}

# section 3: the only files P5-A may open under <F5_ROOT>/gradients/ (path relative to gradients/, bytes, SHA-256)
R4_GRADIENT_FILES: tuple[tuple[str, int, str], ...] = (
    ("t0000_c0002/meta.json", 280, "796d4c1c3b81040fdd2150d3556093b75d1f1ca4eca42e77fe0454960ae3b0ee"),
    (
        "t0000_c0002/step000_grad.safetensors",
        12_612_736,
        "79a309a868ed8428fe7ab2040a67825abf37aff353fcc2bca1904deef0f89f99",
    ),
    (
        "t0000_c0002/step001_grad.safetensors",
        12_612_736,
        "0cfdfc3bc9a05d9105ac04d08b9fd9a8ead3852d3c13c3575edb2e53bd1a4370",
    ),
    ("t0000_c0026/meta.json", 393, "2f475d8a86f8c337e111414c0d68e6a0d9718271c2d631dfe76860b03ba94e4e"),
    (
        "t0000_c0026/step000_grad.safetensors",
        12_612_736,
        "4f3da4a516c76f629f5e7e6b417eb95de97bb638ece738689d45431fab2b5892",
    ),
    (
        "t0000_c0026/step001_grad.safetensors",
        12_612_736,
        "2b2fccb15830b444dae70f47c0c580f9f3eb9140ec925adaa45d4ee9c9fa28b1",
    ),
    (
        "t0000_c0026/step002_grad.safetensors",
        12_612_736,
        "d8cee48f6a0278728a63325a1687bb0541f4683265fd53c6a081970302e2e9b6",
    ),
    ("t0000_c0055/meta.json", 170, "6da7293551ad7eade1fd8d7846d6f4298ca08164ea9bda34c20369bc125b1884"),
    (
        "t0000_c0055/step000_grad.safetensors",
        12_612_736,
        "a271c7d09bfe38a2b6347088ba95dfb189443b05912f8e3cc095e1ab7c5cb808",
    ),
    ("t0000_c0075/meta.json", 168, "43eacf9cd7cbb5b1dfb7fc5af6c031ca818de6a475895c1213edbe1bbdcc9ac0"),
    (
        "t0000_c0075/step000_grad.safetensors",
        12_612_736,
        "d9f5a9f12f8bb181d5fab1118ce2c2ee619b80678165b8751eee10c6eaeab5d9",
    ),
    ("t0000_c0086/meta.json", 282, "5bd53055f71e2ed906f2b1f3cb3ceccdffb840fb00864f7663df520fce062fa7"),
    (
        "t0000_c0086/step000_grad.safetensors",
        12_612_736,
        "6167d489cf105cf77e71dab957a13ac313eea68c79d8724598a77b8b7d8998a7",
    ),
    (
        "t0000_c0086/step001_grad.safetensors",
        12_612_736,
        "236d29f16d13369cd2e9792ba469ce716c3aadce1b48748134fac1108fd1001e",
    ),
)

F6_EVIDENCE_COMMIT = "d3c4eb3d631e0280533f35eb652fb24f68870079"  # results/phase4/screen/f6_r{2,3,4}_*.json
# section 5.3: frozen exact maxima m_r (committed F6 input_stats.train_max_abs) and s_r = m_r / 0.95 (float64)
FROZEN_MAX = {"R2": 0.032825905829668045, "R3": 0.014964050613343716, "R4": 0.0643031895160675}
FROZEN_SCALE = {"R2": 0.034553585083861103, "R3": 0.015751632224572334, "R4": 0.06768756791165001}
# section 3 item 5: committed F6 input_stats.train_rms (secondary semantic check only)
F6_TRAIN_RMS = {"R2": 0.004050884395837784, "R3": 0.0020410344004631042, "R4": 0.0010228796163573861}


LORA_SCALE, LORA_RANK = 16 / 8, 8  # s = alpha / r (section 4)
QWEN_GEOMETRY = {
    "num_layers": 24,
    "modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "rank": 8,
    "hidden": 2048,
}
PHI_SHAPE = (1, 2048, 1536)
LATENT_SHAPE = (64, 32, 24)
KINDS = {"R2": "balanced_effective_state", "R3": "balanced_effective_delta_r8", "R4": "mean_step_gradient"}


@dataclass(frozen=True)
class FrozenInputs:
    """Every frozen input fact of one representation. Only :data:`FROZEN` is used by the production runner."""

    rid: str
    root: str  # relative to CGFED_RUNS
    index_sha256: str
    train_times: frozenset[int]
    n_train: int
    geometry: Mapping[str, Any]
    phi_shape: tuple[int, int, int]
    max_abs: float  # frozen m_r (section 5.3)
    scale: float  # frozen s_r = m_r / 0.95, applied
    train_rms: float  # committed F6 train_rms (section 3 item 5)
    gradient_files: tuple[tuple[str, int, str], ...] | None = None
    lora_scale: float = LORA_SCALE
    rank: int = LORA_RANK

    @property
    def kind(self) -> str:
        return KINDS[self.rid]


def _frozen(rid: str, root: str, times: frozenset[int], n: int, files=None) -> FrozenInputs:
    return FrozenInputs(
        rid, root, INDEX_SHA256[root], times, n, QWEN_GEOMETRY, PHI_SHAPE, FROZEN_MAX[rid], FROZEN_SCALE[rid],
        F6_TRAIN_RMS[rid], files,
    )  # fmt: skip


FROZEN = {
    "R2": _frozen("R2", PHASE3_ROOT, frozenset(range(16)), 80),
    "R3": _frozen("R3", PHASE3_ROOT, frozenset(range(16)), 80),
    "R4": _frozen("R4", F5_ROOT, frozenset({0}), 5, R4_GRADIENT_FILES),
}


def rel_close(a: float, b: float, tol: float) -> bool:
    """``|a - b| <= tol * |b|`` in float64, inclusive (relative to the frozen value ``b``); non-finite fails."""
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tol * abs(b)


def _safe_rel(rel: str) -> PurePosixPath:
    p = PurePosixPath(rel)
    if p.is_absolute() or ".." in p.parts or "\\" in rel or not p.parts:
        raise P5AProtocolError(f"unsafe relative path {rel!r}")
    return p


class AllowlistedGradientReader:
    """Reads only allowlisted files below ``base``; verifies size and SHA-256 of the exact bytes it returns.

    ``opener`` (default: read the whole file in binary mode) is injectable so tests can prove that no other file
    is opened. A file is read at most once; its digest is recorded in ``read_log`` for provenance.
    """

    def __init__(
        self,
        base: Path,
        allowlist: Sequence[tuple[str, int, str]] = R4_GRADIENT_FILES,
        opener: Callable[[Path], bytes] | None = None,
    ) -> None:
        self.base = Path(base)
        self.allow = {str(_safe_rel(rel)): (size, sha) for rel, size, sha in allowlist}
        if len(self.allow) != len(allowlist):
            raise P5AProtocolError("duplicate entry in the R4 file allowlist")
        self.opener = opener or (lambda p: p.read_bytes())
        self.read_log: dict[str, str] = {}

    def read(self, rel: str) -> bytes:
        key = str(_safe_rel(rel))
        if key not in self.allow:
            raise P5AProtocolError(f"R4 protocol violation: {key} is not one of the preregistered files")
        if key in self.read_log:
            raise P5AProtocolError(f"R4 file {key} requested twice")
        path = self.base.joinpath(*PurePosixPath(key).parts)
        if not path.is_file():
            raise P5AProtocolError(f"R4 file missing: {key}")
        data = self.opener(path)
        size, sha = self.allow[key]
        digest = hashlib.sha256(data).hexdigest()
        if len(data) != size or digest != sha:
            raise P5AProtocolError(f"R4 identity mismatch: {key} ({len(data)} B, {digest})")
        self.read_log[key] = digest
        return data


def _adapter_from_bytes(data: bytes, name: str) -> AdapterState:
    """``AdapterState.load(verify=True)`` semantics on already-verified bytes (embedded tensor hash checked)."""
    from safetensors.torch import load as st_load

    state = AdapterState(st_load(data))
    header_len = int.from_bytes(data[:8], "little")
    meta = json.loads(data[8 : 8 + header_len]).get("__metadata__") or {}
    if "sha256" in meta and meta["sha256"] != state.sha256():
        raise P5AProtocolError(f"{name}: stored adapter hash does not match its tensors")
    return state


def load_pre_clip_round(reader: AllowlistedGradientReader, t: int, cid: int) -> dict[str, Any]:
    """Pre-clip gradients of one client round: ``meta.json`` plus ``step{k:03d}_grad.safetensors`` only."""
    d = f"t{t:04d}_c{cid:04d}"
    meta = json.loads(reader.read(f"{d}/meta.json").decode("utf-8"))
    if (int(meta.get("time_index", -1)), int(meta.get("client_id", -1))) != (t, cid):
        raise P5AProtocolError(f"{d}/meta.json names a different client round")
    steps = meta.get("steps")
    if not isinstance(steps, list) or not steps:
        raise P5AProtocolError(f"{d}/meta.json has no recorded optimizer steps")
    if [int(s.get("step", -1)) for s in steps] != list(range(len(steps))):
        raise P5AProtocolError(f"{d}/meta.json step numbering is not 0..n-1")
    grads = [
        _adapter_from_bytes(reader.read(f"{d}/step{k:03d}_grad.safetensors"), f"{d}/step{k:03d}_grad")
        for k in range(len(steps))
    ]
    return {"meta": meta, "grads": grads}


def check_r4_allowlist_consumed(reader: AllowlistedGradientReader) -> None:
    missing = sorted(set(reader.allow) - set(reader.read_log))
    if missing:
        raise P5AProtocolError(
            f"R4 allowlist files not consumed (step counts disagree with section 3): {missing}"
        )


@dataclass
class Population:
    """The hash-verified training population of one representation, sorted as ``E`` (section 6)."""

    rid: str
    records: list[Mapping[str, Any]]  # E: sorted by (time_index, client_id)
    xs: list[torch.Tensor]  # Phi inputs, fp32, same order
    provenance: dict[str, Any] = field(default_factory=dict)


def training_records(frozen: FrozenInputs, runs_root: Path) -> tuple[list[Mapping[str, Any]], str]:
    """Verify the index hash, then select the training population from metadata alone (no payload opened)."""
    root = Path(runs_root) / frozen.root
    index = root / "index.jsonl"
    if not index.is_file():
        raise P5AProtocolError(f"{frozen.rid}: index missing at {index}")
    digest = sha256_file(index)
    if digest != frozen.index_sha256:
        raise P5AProtocolError(f"{frozen.rid}: index.jsonl SHA-256 {digest} != frozen {frozen.index_sha256}")
    try:
        records = read_index(root)
    except SnapshotError as exc:
        raise P5AProtocolError(f"{frozen.rid}: {exc}") from exc
    sort_records(records)  # key uniqueness over the whole index (section 3, item 2)
    train = sort_records([r for r in records if int(r["time_index"]) in frozen.train_times])
    if len(train) != frozen.n_train:
        raise P5AProtocolError(f"{frozen.rid}: {len(train)} training records, expected {frozen.n_train}")
    for r in train:
        if dict(r["geometry"]) != dict(frozen.geometry) or r["layout_id"] != LAYOUT_ID:
            raise P5AProtocolError(
                f"{frozen.rid}: record {record_key(r)} has an unexpected geometry or layout"
            )
    return train, digest


def _representation(
    frozen: FrozenInputs, start: AdapterState | None, end: AdapterState | None, cr: dict | None
) -> AdapterState:
    if frozen.rid == "R2":
        return balanced_effective_state(end, frozen.lora_scale)[0]
    if frozen.rid == "R3":
        return balanced_effective_delta(start, end, frozen.lora_scale, rank=frozen.rank)[0]
    return mean_state(cr["grads"])


def build_population(
    frozen: FrozenInputs, runs_root: Path, opener: Callable[[Path], bytes] | None = None
) -> Population:
    """Construct the AE inputs exactly as F6 (``cmd_forensic_screen``) for the training population only.

    R2/R3 read their states through ``load_states(verify=True)``; R4 reads only its allowlisted pre-clip gradients
    (section 3, P5v2-D06) and never opens a state, clipped or delta file. Every constructed input is checked for
    shape, dtype and finiteness (section 3 item 3) and its tensor SHA-256 is recorded.
    """
    records, index_sha = training_records(frozen, runs_root)
    root = Path(runs_root) / frozen.root
    geom: LoRAGeometry = geometry_from_dict(dict(frozen.geometry))
    layout: Layout = get_layout(LAYOUT_ID)
    reader = (
        AllowlistedGradientReader(root / "gradients", frozen.gradient_files, opener)
        if frozen.gradient_files is not None
        else None
    )
    if (reader is None) != (frozen.rid != "R4"):
        raise P5AProtocolError(f"{frozen.rid}: the R4 allowlist must be given for R4 and only for R4")
    xs, rows = [], []
    for pos, r in enumerate(records):
        key = record_key(r)
        start = end = cr = None
        if reader is None:
            try:
                start, end = load_states(root, r, verify=True)  # file and tensor-content hashes
            except (SnapshotError, OSError) as exc:
                raise P5AProtocolError(f"{frozen.rid} {key}: {exc}") from exc
        else:
            cr = load_pre_clip_round(reader, *key)
        x = layout.forward(_representation(frozen, start, end, cr), geom)
        if tuple(x.shape) != tuple(frozen.phi_shape) or x.dtype != torch.float32:
            raise P5AProtocolError(
                f"{frozen.rid}: Phi input {tuple(x.shape)} {x.dtype} != {frozen.phi_shape} fp32"
            )
        if not bool(torch.isfinite(x).all()):
            raise P5AProtocolError(f"{frozen.rid} {key}: non-finite AE input")
        xs.append(x)
        row = {"position": pos, "time_index": key[0], "client_id": key[1], "phi_sha256": tensor_sha256(x)}
        if reader is None:
            row.update({k: r[k] for k in IDENTITY_FIELDS})
        else:
            d = f"t{key[0]:04d}_c{key[1]:04d}/"
            row["gradient_files"] = {k: v for k, v in reader.read_log.items() if k.startswith(d)}
        rows.append(row)
    if reader is not None:
        check_r4_allowlist_consumed(reader)
    return Population(
        frozen.rid,
        records,
        xs,
        {
            "root": frozen.root,
            "index_sha256": index_sha,
            "n_train": len(records),
            "records": rows,
            "gradient_files_read": dict(reader.read_log) if reader is not None else None,
        },
    )


IDENTITY_FIELDS = (
    "file",
    "file_sha256",
    "start_file",
    "start_file_sha256",
    "end_adapter_hash",
    "start_adapter_hash",
)


def frozen_normalizer(frozen: FrozenInputs, xs: Sequence[torch.Tensor]) -> tuple[Normalizer, dict[str, Any]]:
    """Section 5.4: recompute the exact maximum over the whole population, require
    ``|m_rec - m_frozen| <= 1e-6 * m_frozen``, and apply the FROZEN ``s_r`` (never a recomputed scale)."""
    if len(xs) != frozen.n_train:
        raise P5AProtocolError(
            f"{frozen.rid}: scale population has {len(xs)} tensors, expected {frozen.n_train}"
        )
    if not all(bool(torch.isfinite(x).all()) for x in xs):  # before the maximum: a NaN is never skipped
        raise P5AProtocolError(f"{frozen.rid}: non-finite input element in the scale population")
    recomputed = train_abs_max(xs, range(len(xs)))
    if not (math.isfinite(recomputed) and recomputed > 0):
        raise P5AProtocolError(f"{frozen.rid}: degenerate population maximum {recomputed!r}")
    check = {
        "statistic": "exact_max_abs",
        "frozen_max_abs": frozen.max_abs,
        "recomputed_max_abs": recomputed,
        "abs_diff": abs(recomputed - frozen.max_abs),
        "rel_diff": abs(recomputed - frozen.max_abs) / frozen.max_abs,
        "tolerance": SCALE_REL_TOL,
        "rule": "|recomputed - frozen| <= 1e-6 * frozen",
        "bitwise_equal": recomputed == frozen.max_abs,
        "frozen_scale": frozen.scale,
        "target": MAXABS_TARGET,
    }
    check["pass"] = rel_close(recomputed, frozen.max_abs, SCALE_REL_TOL)
    if not check["pass"]:
        raise P5AProtocolError(
            f"{frozen.rid}: recomputed max |x| {recomputed!r} != frozen {frozen.max_abs!r} (rel. 1e-6); "
            "the frozen value is not replaced"
        )
    try:
        norm = Normalizer(
            NORMALIZATION_MODE,
            scale_global=frozen.scale,
            fit={
                "fitted_on": "p5a_training_population",
                "num_snapshots": len(xs),
                "statistic": "exact_max_abs",
                "target": MAXABS_TARGET,
                "frozen_max_abs": frozen.max_abs,
                "recomputed_max_abs": recomputed,
            },
        )
    except ValueError as exc:  # degenerate frozen scale (section 5.6)
        raise P5AProtocolError(f"{frozen.rid}: {exc}") from exc
    check["range"] = range_check(norm, xs)
    if not check["range"]["pass"]:
        raise P5AProtocolError(f"{frozen.rid}: post-normalisation range check failed {check['range']}")
    return norm, check


def range_check(norm: Normalizer, xs: Sequence[torch.Tensor]) -> dict[str, Any]:
    """Section 5.5 on the fp32 values the training uses: every snapshot max <= 0.95 (1 + 2e-6), population max
    >= 0.95 (1 - 2e-6), compared in float64, inclusive. Integrity only; nothing is clipped or adjusted."""
    upper = MAXABS_TARGET * (1 + RANGE_REL_ALLOWANCE)
    lower = MAXABS_TARGET * (1 - RANGE_REL_ALLOWANCE)
    maxima = [float(norm.normalize(x).abs().max()) for x in xs]
    pop = max(maxima, key=lambda v: math.inf if math.isnan(v) else v)
    ok = all(math.isfinite(m) and m <= upper for m in maxima) and math.isfinite(pop) and pop >= lower
    return {
        "upper_bound": upper,
        "lower_bound_population": lower,
        "population_max_normalized": pop,
        "snapshots_over_upper": sum(1 for m in maxima if not (math.isfinite(m) and m <= upper)),
        "pass": ok,
    }


@dataclass
class ControlSetup:
    """Everything a control needs, fixed before any training (sections 3, 5 and 6 already passed)."""

    rid: str
    control: str
    selected: list[dict[str, Any]]  # position, (t, client), payload identities, Phi SHA-256
    xs: list[torch.Tensor]  # unique selected snapshots, original space
    provenance_error: str | None = None  # a selected snapshot with sig = 0 (section 9.2)


@dataclass
class RepresentationSetup:
    rid: str
    frozen: FrozenInputs
    normalizer: Normalizer
    scale_check: dict[str, Any]
    rms_check: dict[str, Any]
    population: dict[str, Any]
    controls: dict[str, ControlSetup]


def prepare_representation(frozen: FrozenInputs, pop: Population) -> RepresentationSetup:
    """Frozen-scale, range and RMS checks on the whole population, then metadata-only selection of both controls.
    Shared by the CPU preflight and the training invocations, so both check exactly the same things."""
    from cg_fedllm.phase5.p5a import CONTROLS, select_control

    norm, scale_check = frozen_normalizer(frozen, pop.xs)
    rms = secondary_statistics(frozen, pop.xs)
    controls = {}
    for control in CONTROLS:
        sel = select_control(pop.records, control)
        xs = [pop.xs[s["position"]] for s in sel]
        rows = [{**s, **pop.provenance["records"][s["position"]]} for s in sel]
        zero_sig = [s["position"] for s, x in zip(sel, xs) if float(x.double().pow(2).sum()) == 0.0]
        err = f"selected snapshot(s) at positions {zero_sig} have sig = 0" if zero_sig else None
        controls[control] = ControlSetup(frozen.rid, control, rows, xs, err)
    return RepresentationSetup(frozen.rid, frozen, norm, scale_check, rms, pop.provenance, controls)


def evidence(rs: RepresentationSetup) -> dict[str, Any]:
    """The input evidence that the preflight records and every training invocation must reproduce exactly
    (section 5.4.1): identities, Phi hashes, frozen and recomputed statistics and the selection."""
    return {
        "representation": rs.rid,
        "root": rs.population["root"],
        "index_sha256": rs.population["index_sha256"],
        "n_train": rs.population["n_train"],
        "records": rs.population["records"],
        "gradient_files_read": rs.population["gradient_files_read"],
        "frozen_max_abs": rs.frozen.max_abs,
        "frozen_scale": rs.frozen.scale,
        "recomputed_max_abs": rs.scale_check["recomputed_max_abs"],
        "max_bitwise_equal": rs.scale_check["bitwise_equal"],
        "population_max_normalized": rs.scale_check["range"]["population_max_normalized"],
        "train_rms_recomputed": rs.rms_check["recomputed"],
        "selected": {
            c: [{k: s[k] for k in ("position", "time_index", "client_id", "phi_sha256")} for s in cs.selected]
            for c, cs in rs.controls.items()
        },
        "selected_sig_zero": {c: cs.provenance_error for c, cs in rs.controls.items()},
    }


def secondary_statistics(frozen: FrozenInputs, xs: Sequence[torch.Tensor]) -> dict[str, Any]:
    """Section 3 item 5: the F6 RMS expression on the fp32 population vs the committed F6 ``train_rms``."""
    got = float(torch.stack(list(xs)).pow(2).mean().sqrt())
    row = {
        "statistic": "train_rms",
        "recomputed": got,
        "committed": frozen.train_rms,
        "tolerance": RMS_REL_TOL,
        "pass": rel_close(got, frozen.train_rms, RMS_REL_TOL),
    }
    if not row["pass"]:
        raise P5AProtocolError(f"{frozen.rid}: F6 train_rms mismatch {row}")
    return row
