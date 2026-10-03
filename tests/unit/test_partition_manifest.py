"""Shepherd-compatible partition (semantic equivalence with the oracle), D1/D2 disjointness (16) and
canonical manifest stability (17). Runs offline from category-label fixtures."""

from __future__ import annotations

import json
from collections import Counter

import pytest

from cg_fedllm.config import load_config
from cg_fedllm.data.dolly import DollyRecord
from cg_fedllm.data.manifest import ManifestError, PartitionManifest, build_manifest, validate_manifest
from cg_fedllm.data.partition import d1_count, select_subset, shepherd_partition, split_d1_d2
from cg_fedllm.utils.hashing import canonical_json, sha256_bytes, sha256_file
from tests.conftest import FIXTURES, REPO


@pytest.fixture(scope="module")
def dolly_labels():
    fx = json.loads((FIXTURES / "dolly_categories.json").read_text(encoding="utf-8"))
    cats = [fx["categories"][int(c)] for c in fx["codes"]]
    assert len(cats) == fx["num_records"] == 15015
    return fx["source_sha256"], cats


def _oracle(name: str) -> dict:
    return json.loads((FIXTURES / f"shepherd_oracle_{name}.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", ["dirichlet100_seed42", "shards10_seed42"])
def test_partition_is_semantically_equivalent_to_shepherd_oracle(dolly_labels, name):
    src_sha, cats = dolly_labels
    ref = _oracle(name)
    assert ref["source_sha256"] == src_sha
    p = shepherd_partition(
        list(range(len(cats))),
        cats,
        num_clients=ref["num_clients"],
        mode=ref["mode"],
        holdout_per_category=ref["holdout_per_category"],
        dirichlet_alpha=ref["dirichlet_alpha"],
        min_require_size=ref["min_require_size"],
        shards_per_client=ref["shards_per_client"],
        seed=ref["seed"],
    )
    # same held-out examples, same remaining order, same examples per client (in the same order)
    assert p.holdout_ids == ref["holdout_ids"]
    assert p.remaining_ids == ref["remaining_ids"]
    assert [sorted(c) for c in p.client_ids] == [sorted(c) for c in ref["client_ids"]]
    assert p.client_ids == ref["client_ids"]
    assert [len(c) for c in p.client_ids] == [len(c) for c in ref["client_ids"]]
    for mine, theirs in zip(p.client_ids, ref["client_ids"]):
        assert Counter(cats[i] for i in mine) == Counter(cats[i] for i in theirs)


def test_d1_d2_split_is_disjoint_complete_and_deterministic():
    clients = [list(range(i * 100, i * 100 + n)) for i, n in enumerate([47, 10, 3, 1, 303])]
    d1, d2 = split_d1_d2(clients, 0.3, 42)
    d1b, d2b = split_d1_d2(clients, 0.3, 42)
    assert (d1, d2) == (d1b, d2b)
    for ids, a, b in zip(clients, d1, d2):
        assert set(a).isdisjoint(b)
        assert sorted(a + b) == sorted(ids)
        assert len(a) == d1_count(len(ids), 0.3)
        assert a == [x for x in ids if x in set(a)]  # original order preserved
    assert d1_count(47, 0.3) == 14 and d1_count(5, 0.3) == 2 and d1_count(1, 0.3) == 0
    assert split_d1_d2(clients, 0.3, 43)[0] != d1


def test_subset_selection_is_deterministic(dolly_labels):
    _, cats = dolly_labels
    a = select_subset(list(range(len(cats))), cats, 12, 0)
    assert a == select_subset(list(range(len(cats))), cats, 12, 0)
    assert len(a) == 96 and Counter(cats[i] for i in a) == Counter({c: 12 for c in set(cats)})


@pytest.mark.parametrize(
    "cfg_path",
    ["configs/base/shepherd_dolly.yaml", "configs/smoke/llama160m_smoke.yaml"],
)
def test_committed_manifest_is_reproduced_byte_for_byte(dolly_labels, cfg_path):
    src_sha, cats = dolly_labels
    cfg = load_config(REPO / cfg_path)
    records = [
        DollyRecord(i, "", "", "", c) for i, c in enumerate(cats)
    ]  # text is irrelevant to the manifest
    built = build_manifest(cfg.data, records, src_sha)
    committed = REPO / cfg.data.manifest_path
    assert sha256_bytes(canonical_json(built.data)) == sha256_file(committed)
    again = build_manifest(cfg.data, records, src_sha)
    assert canonical_json(again.data) == canonical_json(built.data)


def test_manifest_validation_catches_corruption():
    cfg = load_config(REPO / "configs/smoke/llama160m_smoke.yaml")
    m = json.loads((REPO / cfg.data.manifest_path).read_text(encoding="utf-8"))
    validate_manifest(PartitionManifest(m))
    bad = json.loads(json.dumps(m))
    bad["clients"][0]["d2_ids"].append(bad["clients"][0]["d1_ids"][0])
    with pytest.raises(ManifestError):
        validate_manifest(PartitionManifest(bad))
    leak = json.loads(json.dumps(m))
    leak["clients"][1]["ids"].append(leak["holdout_ids"][0])
    with pytest.raises(ManifestError):
        validate_manifest(PartitionManifest(leak))


def test_canonical_json_is_stable():
    assert canonical_json({"b": 1, "a": [1, 2], "c": "é"}) == '{"a":[1,2],"b":1,"c":"é"}\n'.encode()
