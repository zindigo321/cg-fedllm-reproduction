"""P5-A v2 artifact transactions, caps and the run-wide byte reservation (sections 13.2-13.4), in a temporary dir."""

from __future__ import annotations

import os

import pytest

from cg_fedllm.phase5 import p5a
from cg_fedllm.phase5 import p5a_state as st
from cg_fedllm.phase5.p5a_artifacts import (
    CAPS,
    TMP_SUFFIX,
    ArtifactStore,
    ByteCeilingError,
    ExistingPathError,
    PublicationError,
    _write_tmp,
    artifact_kind,
    json_bytes,
    training_worst_case_bytes,
    worst_case_bytes,
)

REC = p5a.record_path("R2", "single_snapshot")
CKPT = p5a.checkpoint_path("R2", "single_snapshot")


def _files(root):
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


def test_cap_table_and_worst_case_arithmetic():
    # section 13.2, written out: 3 x 300,000 + 3 x 50,000 + 3 x 200,000 + 6 x (20k + 20k + 100k + 150k + 500k + 2.1M)
    assert worst_case_bytes() == 900_000 + 150_000 + 600_000 + 17_340_000 == 18_990_000 <= 20_000_000
    assert training_worst_case_bytes() == 18_090_000  # the draft total, which did not list preflight records
    assert CAPS["checkpoint"] == (2_100_000, 6) and CAPS["preflight"] == (300_000, 3)
    assert ArtifactStore("x").limit == 20_000_000


def test_only_section_13_2_artifacts_are_accepted():
    for rel, kind in ((p5a.preflight_path(2), "preflight"), (p5a.invocation_path(3), "invocation"),
                      (p5a.summary_path(1), "summary"), (p5a.start_path("R4", "fixed_subset4"), "start"),
                      (p5a.end_path("R3", "single_snapshot"), "end"), (p5a.failure_path("R2", "fixed_subset4"), "failure"),
                      (REC, "record"), (CKPT, "checkpoint"),
                      (f"{p5a.control_dir('R4', 'single_snapshot')}/run_metadata.json", "provenance_set")):  # fmt: skip
        assert artifact_kind(rel) == kind
    for bad in ("ledger/summary_4.json", "ledger/summary_20260101.json", "ledger/attempt_R2_single_snapshot.json",
                "R2_x/dump.safetensors", "preflight/preflight_4.json", "notes.txt"):  # fmt: skip
        with pytest.raises(PublicationError):
            artifact_kind(bad)


def test_exact_footprint_includes_temporaries_and_every_subdirectory(tmp_path):
    store = ArtifactStore(tmp_path / "run")
    store.publish(p5a.summary_path(1), b"x" * 10, reserved_other=0)
    store.publish(REC, b"y" * 37, reserved_other=0)
    (store.root / "R2_x").mkdir()
    (store.root / "R2_x" / ".stray.p5a-tmp").write_bytes(b"zzz")  # a surviving temporary counts
    assert store.footprint() == 10 + 37 + 3
    entry = store.publish(p5a.summary_path(2), b"w", reserved_other=0)
    assert entry["footprint_before"] == 50  # a file reports the footprint measured before it


def test_cap_and_reservation_are_enforced_before_writing(tmp_path):
    store = ArtifactStore(tmp_path / "run", limit=1_000_000)
    with pytest.raises(ByteCeilingError) as err:  # over its own cap: never written
        store.publish(p5a.start_path("R2", "single_snapshot"), b"s" * 20_001, reserved_other=0)
    assert err.value.accounting["within_cap"] is False and _files(tmp_path) == []
    store.publish(
        p5a.start_path("R2", "single_snapshot"), b"s" * 20_000, reserved_other=0
    )  # at the cap: allowed
    # the reservation counts the file's cap, not its size: footprint 20,000 + cap 500,000 + others
    store.publish(REC, b"r", reserved_other=1_000_000 - 20_000 - 500_000)  # exactly at the limit
    with pytest.raises(ByteCeilingError) as err:
        store.publish(
            p5a.record_path("R2", "fixed_subset4"), b"r", reserved_other=1_000_000 - 20_001 - 500_000 + 1
        )
    assert err.value.accounting["reservation_total"] == 1_000_001 and err.value.accounting["within_cap"]
    assert store.footprint() == 20_001


def test_refusal_cause_is_classified_separately_from_any_status(tmp_path):
    """v2.1 A4.1 at the exact boundaries: per-file cap, aggregate ceiling (by the file's own bytes), reservation."""
    store = ArtifactStore(tmp_path / "run", limit=100_000)
    start = p5a.start_path("R2", "single_snapshot")
    assert store.check(start, b"s" * 20_000, 0)["resource_stop_cause"] is None
    assert store.check(start, b"s" * 20_001, 0)["resource_stop_cause"] == "per_file_cap"
    store.path("ledger").mkdir(parents=True)
    (store.root / "ledger" / f".x.0123456789ab{TMP_SUFFIX}").write_bytes(
        b"\0" * 90_000
    )  # a retained temporary
    acc = store.check(
        start, b"s" * 10_000, 0
    )  # 90,000 + 10,000 = limit: the file itself fits the ceiling ...
    assert (
        acc["fits"] is False and acc["resource_stop_cause"] == "evidence_reservation"
    )  # ... its cap does not
    acc = store.check(start, b"s" * 10_001, 0)
    assert acc["resource_stop_cause"] == "aggregate_byte_cap"
    with pytest.raises(ByteCeilingError) as err:
        store.publish(start, b"s" * 10_001, reserved_other=0)
    assert err.value.accounting["resource_stop_cause"] == "aggregate_byte_cap" and not store.exists(start)


def test_provenance_set_shares_one_cap(tmp_path):
    store = ArtifactStore(tmp_path / "run")
    d = p5a.control_dir("R2", "single_snapshot")
    store.publish(f"{d}/config.resolved.yaml", b"a" * 100_000, reserved_other=0)
    store.publish(f"{d}/config.sha256.json", b"b" * 50_000, reserved_other=0)  # 150,000 together: allowed
    with pytest.raises(ByteCeilingError):
        store.publish(f"{d}/run_metadata.json", b"c", reserved_other=0)  # the set's cap is used up


def test_reservation_keeps_evidence_of_every_unrecorded_control(tmp_path):
    store = ArtifactStore(tmp_path / "run")
    full = st.reserved_caps(store, invocation=1)
    per_control = 20_000 + 20_000 + 100_000 + 150_000 + 500_000
    assert full == 3 * 200_000 + 2 * 50_000 + 6 * per_control  # summaries, later invocations, all controls
    assert st.reserved_caps(store, invocation=3) == 200_000 + 6 * per_control
    with_ckpt = st.reserved_caps(store, invocation=1, checkpoint_for=("R2", "single_snapshot"))
    assert with_ckpt == full + 2_100_000
    excl = st.reserved_caps(store, invocation=1, exclude={REC})
    assert excl == full - 500_000
    store.publish(REC, b"{}", reserved_other=0)  # RECORDED: nothing more is reserved for that control
    assert st.reserved_caps(store, invocation=1) == full - per_control


def test_existing_files_are_never_overwritten_or_deleted(tmp_path):
    store = ArtifactStore(tmp_path / "run")
    store.path(REC).parent.mkdir(parents=True)
    store.path(REC).write_bytes(b"user")
    with pytest.raises(ExistingPathError):
        store.publish(REC, b"mine", reserved_other=0)
    assert store.path(REC).read_bytes() == b"user"


def test_collision_after_the_check_keeps_the_intruder_and_earlier_files(tmp_path):
    store = ArtifactStore(tmp_path / "run")
    store.publish(CKPT, b"ckpt", reserved_other=0)
    real_link = store.linker

    def racing_link(src, dst):
        if dst == store.path(REC):  # another writer creates the final path between the check and the link
            store.path(REC).write_bytes(b"intruder")
        real_link(src, dst)

    store.linker = racing_link
    with pytest.raises(ExistingPathError):
        store.publish(REC, b"rec", reserved_other=0)
    assert store.path(REC).read_bytes() == b"intruder" and store.path(CKPT).read_bytes() == b"ckpt"
    assert not [f for f in _files(store.root) if f.endswith(".p5a-tmp")]


def test_failed_write_removes_only_own_temporary(tmp_path):
    store = ArtifactStore(tmp_path / "run")
    seen = []

    def failing_writer(path, data):
        seen.append(path)
        path.write_bytes(data[:1])  # a partial temporary file
        raise OSError("disk full")

    store.path(REC).parent.mkdir(parents=True)
    foreign = store.path(REC).parent / ".p5a_record.json.someoneelse.p5a-tmp"
    foreign.write_bytes(b"not ours")
    store.writer = failing_writer
    with pytest.raises(PublicationError, match="disk full"):
        store.publish(REC, b"rrrr", reserved_other=0)
    assert not seen[0].exists() and not store.path(REC).exists()  # own failed temporary removed
    assert foreign.read_bytes() == b"not ours" and store.footprint() == len(b"not ours")  # counted, untouched


def test_interruption_cleans_own_temporary_and_propagates(tmp_path):
    store = ArtifactStore(tmp_path / "run")

    def interrupted(path, data):
        path.write_bytes(data[:2])
        raise KeyboardInterrupt

    store.writer = interrupted
    with pytest.raises(KeyboardInterrupt):
        store.publish(p5a.start_path("R2", "single_snapshot"), b"xxxx", reserved_other=0)
    assert _files(store.root) == []


def test_no_overwrite_fallback_when_links_are_unsupported(tmp_path):
    store = ArtifactStore(tmp_path / "run")

    def no_links(src, dst):
        raise OSError("hard links not supported")

    store.linker = no_links
    with pytest.raises(PublicationError):
        store.publish(p5a.invocation_path(1), b"a", reserved_other=0)
    assert _files(store.root) == []


def test_real_os_link_is_no_clobber_here(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_bytes(b"1")
    b.write_bytes(b"2")
    with pytest.raises(FileExistsError):
        os.link(a, b)
    assert b.read_bytes() == b"2"
    _write_tmp(tmp_path / "c", b"3")
    with pytest.raises(FileExistsError):
        _write_tmp(tmp_path / "c", b"4")  # exclusive creation


def test_paths_are_confined_and_json_is_strict(tmp_path):
    store = ArtifactStore(tmp_path / "run")
    for bad in ("../x", "/abs", "a\\b", "x.p5a-tmp"):
        with pytest.raises(PublicationError):
            store.publish(bad, b"1", reserved_other=0)
    with pytest.raises(ValueError):
        json_bytes({"x": float("nan")})
    assert json_bytes({"b": 1, "a": 2}) == b'{\n  "a": 2,\n  "b": 1\n}\n'
