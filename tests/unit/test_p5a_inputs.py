"""P5-A v2 input identity and construction on synthetic snapshot/gradient trees (no real payload is read)."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest
import torch

from cg_fedllm.compression.layout import get_layout
from cg_fedllm.forensics.gradients import GradientDumper, load_client_round, mean_state
from cg_fedllm.forensics.representations import balanced_effective_delta, balanced_effective_state
from cg_fedllm.models.adapter import AdapterState
from cg_fedllm.phase5 import p5a_inputs as inp
from cg_fedllm.phase5.p5a import P5AProtocolError
from cg_fedllm.tgap.snapshots import SnapshotWriter
from cg_fedllm.utils.hashing import sha256_file

HIDDEN, LAYERS, RANK = 64, 1, 8  # Phi [1, 64, 64]
GEOM = {
    "num_layers": LAYERS,
    "modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "rank": RANK,
    "hidden": HIDDEN,
}
LAYOUT = get_layout("layer_major_qkvo_AtB")
TRAIN_TIMES, VAL_TIMES = (0, 1), (2,)
CLIENTS = (3, 7)


def _state(seed: int, b_zero: bool = False) -> AdapterState:
    g = torch.Generator().manual_seed(seed)
    out = {}
    for m in GEOM["modules"]:
        out[f"layers.0.self_attn.{m}.lora_A.weight"] = torch.randn(RANK, HIDDEN, generator=g) * 0.02
        b = torch.randn(HIDDEN, RANK, generator=g) * 0.01
        out[f"layers.0.self_attn.{m}.lora_B.weight"] = torch.zeros_like(b) if b_zero else b
    return AdapterState(out)


@pytest.fixture
def snap_root(tmp_path) -> Path:
    """A federated snapshot set with training (t 0-1) and validation (t 2) records; start of t0 has B = 0."""
    root = tmp_path / "runs" / "synthetic_set"
    w = SnapshotWriter(
        root, run_id="syn", source_mode="federated_pretrain", representation="adapter_state", layout=LAYOUT
    )
    for t in (*TRAIN_TIMES, *VAL_TIMES):
        start = _state(100 + t, b_zero=(t == 0))
        for c in reversed(CLIENTS):  # write order differs from E order
            w.write(t, c, start, _state(10 * t + c), num_samples=4)
    return root


def _frozen(root: Path, rid: str = "R2", **kw) -> inp.FrozenInputs:
    base = dataclasses.replace(
        inp.FROZEN[rid],
        root=root.name,
        index_sha256=sha256_file(root / "index.jsonl"),
        train_times=frozenset(TRAIN_TIMES),
        n_train=len(TRAIN_TIMES) * len(CLIENTS),
        geometry=GEOM,
        phi_shape=(1, HIDDEN, 2 * 4 * LAYERS * RANK),
    )
    return dataclasses.replace(base, **kw)


class Spy:
    """Records every path opened through the P5-A reader and through ``Path.open`` (hashing and safetensors)."""

    def __init__(self, monkeypatch):
        self.opened: list[str] = []
        real_open = Path.open

        def spy_open(p, *a, **k):
            self.opened.append(Path(p).as_posix())
            return real_open(p, *a, **k)

        monkeypatch.setattr(Path, "open", spy_open)
        import safetensors.torch as st

        real_load_file = st.load_file

        def spy_load_file(path, *a, **k):
            self.opened.append(Path(path).as_posix())
            return real_load_file(path, *a, **k)

        monkeypatch.setattr(st, "load_file", spy_load_file)
        monkeypatch.setattr("cg_fedllm.utils.io.load_file", spy_load_file)


def test_r2_r3_match_the_f6_construction_and_skip_validation_payloads(snap_root, monkeypatch):
    spy = Spy(monkeypatch)
    fz2, fz3 = _frozen(snap_root, "R2"), _frozen(snap_root, "R3")
    pop2 = inp.build_population(fz2, snap_root.parent)
    pop3 = inp.build_population(fz3, snap_root.parent)
    assert [(r["time_index"], r["client_id"]) for r in pop2.records] == [(0, 3), (0, 7), (1, 3), (1, 7)]
    assert not any("t0002_" in p for p in spy.opened), "a validation-split payload was opened"
    from cg_fedllm.compression.layout import geometry_from_dict
    from cg_fedllm.tgap.snapshots import load_states

    geom = geometry_from_dict(GEOM)
    for r, x2, x3 in zip(pop2.records, pop2.xs, pop3.xs):
        start, end = load_states(snap_root, r)
        exp2 = LAYOUT.forward(balanced_effective_state(end, 2.0)[0], geom)
        exp3 = LAYOUT.forward(balanced_effective_delta(start, end, 2.0, rank=8)[0], geom)
        assert x2.dtype == torch.float32 and torch.equal(x2, exp2) and torch.equal(x3, exp3)
    assert (
        pop2.provenance["index_sha256"] == fz2.index_sha256 and pop2.provenance["gradient_files_read"] is None
    )


def test_index_hash_counts_geometry_and_payload_tampering_fail_closed(snap_root):
    fz = _frozen(snap_root)
    with pytest.raises(P5AProtocolError, match="SHA-256"):
        inp.build_population(dataclasses.replace(fz, index_sha256="0" * 64), snap_root.parent)
    with pytest.raises(P5AProtocolError, match="training records"):
        inp.build_population(dataclasses.replace(fz, n_train=5), snap_root.parent)
    with pytest.raises(P5AProtocolError, match="geometry"):
        inp.build_population(dataclasses.replace(fz, geometry={**GEOM, "hidden": 128}), snap_root.parent)
    with pytest.raises(P5AProtocolError, match="missing"):
        inp.build_population(dataclasses.replace(fz, root="nowhere"), snap_root.parent)
    target = snap_root / "snapshots" / "t0001_c0007.safetensors"
    data = bytearray(target.read_bytes())
    data[-1] ^= 1
    target.write_bytes(bytes(data))
    with pytest.raises(P5AProtocolError, match="hash mismatch"):
        inp.build_population(fz, snap_root.parent)


STEPS = {(0, 3): 2, (0, 7): 1, (1, 3): 1, (1, 7): 3}  # (t, client) -> optimizer steps


class _P:
    """Minimal parameter stand-in for GradientDumper (``.grad`` and ``.detach()``)."""

    def __init__(self, value: torch.Tensor, grad: torch.Tensor) -> None:
        self.value, self.grad, self.shape = value, grad, value.shape

    def detach(self):
        return self.value


def _dump_round(grad_root: Path, t: int, cid: int, steps: int) -> list[AdapterState]:
    """Write one client round with the real GradientDumper layout (pre-clip, clipped and delta files + meta.json)."""
    dumper = GradientDumper(grad_root, t, cid)
    grads = []
    for k in range(steps):
        g = _state(1000 + 31 * t + 7 * cid + k)
        grads.append(g)
        params = {key: _P(torch.zeros_like(v), v) for key, v in g.tensors.items()}
        dumper.on_gradients(k, params, loss=1.0)
        dumper.on_clipped(k, {key: _P(torch.zeros_like(v), v * 0.5) for key, v in g.tensors.items()}, 2.0)
        dumper.on_step(k, params, {key: torch.ones_like(v) for key, v in g.tensors.items()})
    return grads


def _allowlist(grad_root: Path, keys) -> tuple[tuple[str, int, str], ...]:
    rows = []
    for t, c in keys:
        d = f"t{t:04d}_c{c:04d}"
        for name in ["meta.json"] + [f"step{k:03d}_grad.safetensors" for k in range(STEPS[(t, c)])]:
            p = grad_root / d / name
            rows.append((f"{d}/{name}", p.stat().st_size, sha256_file(p)))
    return tuple(rows)


@pytest.fixture
def r4_tree(snap_root):
    grad_root = snap_root / "gradients"
    grads = {key: _dump_round(grad_root, *key, steps) for key, steps in STEPS.items()}
    _dump_round(grad_root, 2, 3, 2)  # a validation round: must never be opened
    allow = _allowlist(grad_root, sorted(STEPS))
    fz = _frozen(snap_root, "R4", gradient_files=allow)
    return snap_root, grad_root, grads, fz


def test_r4_reads_only_the_allowlist_and_matches_mean_state(r4_tree, monkeypatch):
    root, grad_root, grads, fz = r4_tree
    spy = Spy(monkeypatch)
    pop = inp.build_population(
        fz, root.parent
    )  # default opener: Path.read_bytes, seen once by the Path.open spy
    grad_opened = [p for p in spy.opened if "/gradients/" in p]
    allowed = {(grad_root / rel).as_posix() for rel, _, _ in fz.gradient_files}
    assert set(grad_opened) == allowed and len(grad_opened) == len(allowed)
    assert not any("_clipped" in p or "_delta" in p or "t0002_" in p for p in spy.opened)
    # R4 depends only on its gradients: no start/end state file and no index-named payload is opened (P5v2-D06)
    assert not any("/snapshots/" in p or "/starts/" in p for p in spy.opened)
    from cg_fedllm.compression.layout import geometry_from_dict

    for r, x in zip(pop.records, pop.xs):
        key = (r["time_index"], r["client_id"])
        expected = LAYOUT.forward(mean_state(grads[key]), geometry_from_dict(GEOM))
        assert torch.equal(x, expected) and x.dtype == torch.float32
        # the opt-out path the F6 code uses opens clipped/delta files; P5-A's reader must agree with its grads
        legacy = mean_state(load_client_round(grad_root / f"t{key[0]:04d}_c{key[1]:04d}")["grads"])
        assert torch.equal(x, LAYOUT.forward(legacy, geometry_from_dict(GEOM)))
    assert pop.provenance["gradient_files_read"] == {rel: sha for rel, _, sha in fz.gradient_files}
    from cg_fedllm.utils.hashing import tensor_sha256

    for row, x in zip(pop.provenance["records"], pop.xs):
        assert row["phi_sha256"] == tensor_sha256(x) and "file_sha256" not in row
        d = f"t{row['time_index']:04d}_c{row['client_id']:04d}/"
        assert set(row["gradient_files"]) == {r for r, _, _ in fz.gradient_files if r.startswith(d)}


def test_r4_identity_failures_are_detected_before_use(r4_tree):
    root, grad_root, _grads, fz = r4_tree
    rel, size, sha = fz.gradient_files[1]
    bad_sha = tuple((r, s, "0" * 64 if r == rel else h) for r, s, h in fz.gradient_files)
    with pytest.raises(P5AProtocolError, match="identity mismatch"):
        inp.build_population(dataclasses.replace(fz, gradient_files=bad_sha), root.parent)
    bad_size = tuple((r, s + 1 if r == rel else s, h) for r, s, h in fz.gradient_files)
    with pytest.raises(P5AProtocolError, match="identity mismatch"):
        inp.build_population(dataclasses.replace(fz, gradient_files=bad_size), root.parent)
    missing = tuple(row for row in fz.gradient_files if row[0] != rel)  # a needed file is not allowlisted
    with pytest.raises(P5AProtocolError, match="not one of the preregistered"):
        inp.build_population(dataclasses.replace(fz, gradient_files=missing), root.parent)
    extra = (*fz.gradient_files, ("t0000_c0003/step009_grad.safetensors", 1, "0" * 64))
    with pytest.raises(P5AProtocolError, match="not consumed"):
        inp.build_population(dataclasses.replace(fz, gradient_files=extra), root.parent)


def test_r4_meta_mismatch_missing_file_and_path_escape(r4_tree):
    root, grad_root, _grads, fz = r4_tree
    meta = grad_root / "t0001_c0007" / "meta.json"
    m = json.loads(meta.read_text(encoding="utf-8"))
    m["client_id"] = 8
    meta.write_text(json.dumps(m), encoding="utf-8")
    rows = tuple(
        (r, meta.stat().st_size, sha256_file(meta)) if r == "t0001_c0007/meta.json" else (r, s, h)
        for r, s, h in fz.gradient_files
    )
    with pytest.raises(P5AProtocolError, match="different client round"):
        inp.build_population(dataclasses.replace(fz, gradient_files=rows), root.parent)
    (grad_root / "t0000_c0003" / "step001_grad.safetensors").unlink()
    with pytest.raises(P5AProtocolError, match="missing"):
        inp.build_population(fz, root.parent)
    reader = inp.AllowlistedGradientReader(grad_root, fz.gradient_files)
    for bad in (
        "../index.jsonl",
        "/etc/passwd",
        "t0000_c0003\\meta.json",
        "t0000_c0003/step000_grad_clipped.safetensors",
    ):
        with pytest.raises(P5AProtocolError):
            reader.read(bad)
    with pytest.raises(P5AProtocolError, match="duplicate"):
        inp.AllowlistedGradientReader(grad_root, (*fz.gradient_files, fz.gradient_files[0]))


def test_production_r4_allowlist_is_the_preregistered_table():
    rows = inp.R4_GRADIENT_FILES
    assert len(rows) == 14 and sum(1 for r, _, _ in rows if r.endswith("meta.json")) == 5
    steps = {}
    for rel, size, sha in rows:
        d, name = rel.split("/")
        assert len(sha) == 64 and int(sha, 16) >= 0
        if name.startswith("step"):
            assert name.endswith("_grad.safetensors") and size == 12_612_736
            steps[d] = steps.get(d, 0) + 1
    # step counts 2, 3, 1, 1, 2 from each meta.json (section 3)
    assert steps == {"t0000_c0002": 2, "t0000_c0026": 3, "t0000_c0055": 1, "t0000_c0075": 1, "t0000_c0086": 2}
    assert hashlib.sha256(json.dumps(rows).encode()).hexdigest()  # stable, hashable table


def test_rms_check_reproduces_f6_expression_and_fails_on_mismatch(snap_root):
    fz = _frozen(snap_root)
    pop = inp.build_population(fz, snap_root.parent)
    flat = [v for x in pop.xs for v in x.reshape(-1).double().tolist()]
    oracle = (sum(v * v for v in flat) / len(flat)) ** 0.5  # independent float64 RMS
    got = inp.secondary_statistics(dataclasses.replace(fz, train_rms=oracle), pop.xs)
    assert got["pass"] and abs(got["recomputed"] - oracle) <= 1e-6 * oracle
    near = dataclasses.replace(fz, train_rms=got["recomputed"] * (1 + 0.5e-6))
    assert inp.secondary_statistics(near, pop.xs)["pass"]
    far = dataclasses.replace(fz, train_rms=got["recomputed"] * (1 + 2e-6))
    with pytest.raises(P5AProtocolError, match="train_rms"):
        inp.secondary_statistics(far, pop.xs)


def test_r2_records_carry_payload_identities_and_phi_hashes(snap_root):
    from cg_fedllm.utils.hashing import tensor_sha256

    pop = inp.build_population(_frozen(snap_root), snap_root.parent)
    for row, rec, x in zip(pop.provenance["records"], pop.records, pop.xs):
        assert row["phi_sha256"] == tensor_sha256(x)
        assert {k: row[k] for k in inp.IDENTITY_FIELDS} == {k: rec[k] for k in inp.IDENTITY_FIELDS}


def test_r4_allowlist_is_required_for_r4_only(snap_root, r4_tree):
    root, _grad_root, _grads, fz = r4_tree
    with pytest.raises(P5AProtocolError, match="allowlist"):
        inp.build_population(dataclasses.replace(fz, gradient_files=None), root.parent)
    with pytest.raises(P5AProtocolError, match="allowlist"):
        inp.build_population(
            dataclasses.replace(_frozen(snap_root), gradient_files=fz.gradient_files), root.parent
        )


def test_population_count_and_r4_step_numbering_fail_closed(r4_tree):
    root, grad_root, _grads, fz = r4_tree
    with pytest.raises(P5AProtocolError, match="training records"):
        inp.build_population(dataclasses.replace(fz, n_train=3), root.parent)
    meta = grad_root / "t0001_c0007" / "meta.json"
    m = json.loads(meta.read_text(encoding="utf-8"))
    m["steps"][1]["step"] = 5
    meta.write_text(json.dumps(m), encoding="utf-8")
    rows = tuple(
        (r, meta.stat().st_size, sha256_file(meta)) if r == "t0001_c0007/meta.json" else (r, s, h)
        for r, s, h in fz.gradient_files
    )
    with pytest.raises(P5AProtocolError, match="numbering"):
        inp.build_population(dataclasses.replace(fz, gradient_files=rows), root.parent)


def test_prepare_representation_selects_from_metadata_and_shares_one_scale(snap_root):
    fz = _frozen(snap_root)
    pop = inp.build_population(fz, snap_root.parent)
    m = max(abs(v) for x in pop.xs for v in x.reshape(-1).tolist())
    rms = float(torch.stack(pop.xs).pow(2).mean().sqrt())
    fz = dataclasses.replace(fz, max_abs=m, scale=m / 0.95, train_rms=rms)
    rs = inp.prepare_representation(fz, pop)
    assert rs.normalizer.scale_global == m / 0.95 and rs.scale_check["bitwise_equal"]
    assert [s["position"] for s in rs.controls["single_snapshot"].selected] == [1]  # N = 4: floor(3 / 2)
    assert [s["position"] for s in rs.controls["fixed_subset4"].selected] == [0, 1, 2, 3]
    ev = inp.evidence(rs)
    assert ev["recomputed_max_abs"] == m and ev["selected"]["single_snapshot"][0]["phi_sha256"]


def test_duplicate_record_keys_abort(snap_root):
    lines = (snap_root / "index.jsonl").read_text(encoding="utf-8").splitlines()
    (snap_root / "index.jsonl").write_text("\n".join([*lines, lines[0]]) + "\n", encoding="utf-8")
    with pytest.raises(P5AProtocolError, match="not unique"):
        inp.build_population(_frozen(snap_root), snap_root.parent)


def test_preflight_on_a_synthetic_tree_opens_only_allowed_files(r4_tree, snap_root, monkeypatch, tmp_path):
    """The real preflight route (build_population + preflight record) on synthetic files, with clipped, delta and
    validation-round files present; only allowlisted gradient files and R2/R3 training payloads are opened."""
    from cg_fedllm.phase5 import p5a as protocol
    from cg_fedllm.phase5.p5a_artifacts import ArtifactStore
    from cg_fedllm.phase5.p5a_preflight import run_preflight

    root, grad_root, _grads, fz4 = r4_tree
    frozen = {}
    for rid, base in (("R2", _frozen(snap_root, "R2")), ("R3", _frozen(snap_root, "R3")), ("R4", fz4)):
        pop = inp.build_population(base, root.parent)
        m = max(abs(v) for x in pop.xs for v in x.reshape(-1).tolist())
        frozen[rid] = dataclasses.replace(base, max_abs=m, scale=m / 0.95,
                                          train_rms=float(torch.stack(pop.xs).pow(2).mean().sqrt()))  # fmt: skip
    spy = Spy(monkeypatch)
    store = ArtifactStore(tmp_path / protocol.P5A_RUN_NAME)
    rec = run_preflight(store, identity={"execution_commit": "x", "config_sha256": "y"},
                        load=lambda fz: inp.build_population(fz, root.parent), frozen=frozen)  # fmt: skip
    assert rec["pass"] and rec["trainings_started"] == 0
    payloads = [p for p in spy.opened if p.startswith(root.as_posix())]
    assert not any("_clipped" in p or "_delta" in p or "t0002_" in p for p in payloads)
    allowed_grads = {(grad_root / rel).as_posix() for rel, _, _ in fz4.gradient_files}
    assert {p for p in payloads if "/gradients/" in p} == allowed_grads
    assert store.files() == [protocol.preflight_path(1)]
