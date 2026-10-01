"""Config validation (testing standard item 1)."""

from __future__ import annotations

import pytest

from cg_fedllm.config import (
    PHASE2_RESULT_LABELS,
    RESULT_LABELS,
    ConfigError,
    FederatedSection,
    ModelSection,
    apply_overrides,
    build_dataclass,
    config_from_dict,
    expand_env,
    load_config,
)
from tests.conftest import REPO

BASE = {"run": {"name": "x"}}


def test_all_repo_configs_load_and_validate():
    for path in (REPO / "configs").rglob("*.yaml"):
        if path.parent.name == "feasible" and path.name == "gpu_microbench.yaml":
            continue  # benchmark spec, not an ExperimentConfig
        cfg = load_config(path)
        assert cfg.run.name
        assert len(cfg.sha256()) == 64


def test_unknown_key_is_an_error():
    with pytest.raises(ConfigError, match="unknown key"):
        config_from_dict({"run": {"name": "x", "sede": 3}})


def test_literal_values_are_checked():
    with pytest.raises(ConfigError, match="not in allowed"):
        config_from_dict({**BASE, "federated": {"aggregation": "fedprox"}})
    with pytest.raises(ConfigError, match="not in allowed"):
        config_from_dict({**BASE, "federated": {"representation": "gradients"}})


def test_result_label_is_restricted_to_the_reviewer_labels():
    assert RESULT_LABELS == (
        "PAPER-REPORTED", "PHASE2-SMOKE", "LOCAL-MICROBENCH", "DERIVED", "UNKNOWN",
        "PHASE3-DIAGNOSTIC", "PHASE3-TIERB-CORE", "PHASE3-SENSITIVITY",
    )
    # migration rule: the schema only grows -- every label a Phase-2 record carries stays valid
    assert PHASE2_RESULT_LABELS == ("PAPER-REPORTED", "PHASE2-SMOKE", "LOCAL-MICROBENCH", "DERIVED", "UNKNOWN")
    assert set(PHASE2_RESULT_LABELS) <= set(RESULT_LABELS)
    for label in RESULT_LABELS:
        assert config_from_dict({"run": {"name": "x", "result_label": label}}).run.result_label == label
    assert config_from_dict(BASE).run.result_label == "UNKNOWN"
    with pytest.raises(ConfigError, match="not in allowed"):
        config_from_dict({"run": {"name": "x", "result_label": "LOCAL-EVALUATOR-VALIDATION"}})
    # every Phase-2 config that produces results says so explicitly
    for rel in ("smoke/llama160m_smoke.yaml", "smoke/tiny_cpu.yaml", "eval/qwen15_0p5b_validation.yaml"):
        assert load_config(REPO / "configs" / rel).run.result_label == "PHASE2-SMOKE"


def test_bool_is_not_an_int_and_types_are_strict():
    with pytest.raises(ConfigError, match="expected int"):
        config_from_dict({"run": {"name": "x", "seed": True}})
    with pytest.raises(ConfigError, match="expected str"):
        config_from_dict({"run": {"name": 5}})
    cfg = config_from_dict({**BASE, "local_train": {"learning_rate": 1}})
    assert isinstance(cfg.local_train.learning_rate, float)


def test_missing_required_and_section_validators():
    with pytest.raises(ConfigError, match="missing required key 'name'"):
        config_from_dict({"run": {}})
    with pytest.raises(ConfigError, match="full 40-char"):
        build_dataclass(ModelSection, {"kind": "hf", "id": "a/b", "revision": "main"})
    with pytest.raises(ConfigError, match="pooled"):
        build_dataclass(FederatedSection, {"pooled": True, "client_fraction": 0.5})
    with pytest.raises(ConfigError, match="multiple of micro_batch_size"):
        config_from_dict({**BASE, "local_train": {"batch_size": 10, "micro_batch_size": 4}})
    with pytest.raises(ConfigError, match="ae_checkpoint"):
        config_from_dict({**BASE, "codec": {"type": "autoencoder"}})


def test_inheritance_and_overrides(tmp_path):
    (tmp_path / "base.yaml").write_text("run: {name: base, seed: 1}\nfederated: {num_rounds: 3}\n", encoding="utf-8")
    (tmp_path / "child.yaml").write_text("inherit: base.yaml\nrun: {name: child}\n", encoding="utf-8")
    cfg = load_config(tmp_path / "child.yaml", ["federated.client_fraction=0.25", "run.seed=9"])
    assert (cfg.run.name, cfg.run.seed, cfg.federated.num_rounds, cfg.federated.client_fraction) == ("child", 9, 3, 0.25)
    assert apply_overrides({"a": {"b": 1}}, ["a.c=[1, 2]"]) == {"a": {"b": 1, "c": [1, 2]}}


def test_circular_inheritance_is_detected(tmp_path):
    (tmp_path / "a.yaml").write_text("inherit: b.yaml\nrun: {name: a}\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("inherit: a.yaml\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="circular"):
        load_config(tmp_path / "a.yaml")


def test_env_default_expansion(monkeypatch):
    monkeypatch.delenv("CGFED_TEST_VAR", raising=False)
    assert expand_env("${CGFED_TEST_VAR:-fallback}/x") == "fallback/x"
    monkeypatch.setenv("CGFED_TEST_VAR", "set")
    assert expand_env("${CGFED_TEST_VAR:-fallback}/x") == "set/x"


def test_config_hash_is_stable_and_sensitive():
    a = config_from_dict({**BASE, "federated": {"num_rounds": 2}})
    b = config_from_dict({**BASE, "federated": {"num_rounds": 2}})
    c = config_from_dict({**BASE, "federated": {"num_rounds": 3}})
    assert a.sha256() == b.sha256() != c.sha256()
