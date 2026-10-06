"""Validated, declarative experiment configuration.

Design rules (see docs/reproduction_protocol.md):

* Every scientifically relevant value is an explicit config field; library defaults (PEFT, Trainer,
  AdamW) are never relied upon silently.
* Unknown keys are errors (typos must not silently fall back to defaults).
* ``Literal`` fields are checked, ``bool`` is never accepted where an ``int``/``float`` is expected.
* YAML files may ``inherit:`` other YAML files (paths relative to the inheriting file); later files
  and ``--set a.b=value`` overrides win.
* The fully resolved configuration is saved with every run, together with its canonical SHA-256.
"""

from __future__ import annotations

import copy
import dataclasses
import os
import re
import types
import typing
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

import yaml

from cg_fedllm.utils.hashing import canonical_json_sha256

Representation = Literal["adapter_state", "adapter_delta"]
Aggregation = Literal["sample_weighted_mean", "uniform_mean", "literal_sum"]
TGAPSource = Literal["local_pretrain", "federated_pretrain"]
CodecType = Literal["none", "identity", "autoencoder", "constant_mean", "gaussian_noise"]
# AE input normalisation (Phase 3, A2): a DIAGNOSTIC/STABILISED variant -- the paper specifies none.
NormalizationMode = Literal[
    "none", "global_rms", "factor_rms", "global_maxabs_train", "global_exact_maxabs_train"
]
# Every reported result carries exactly one of these labels (reviewer scientific-integrity rule). The schema
# only ever grows: labels written by earlier phases stay valid (tests/unit/test_config.py).
ResultLabel = Literal[
    "PAPER-REPORTED",
    "PHASE2-SMOKE",
    "LOCAL-MICROBENCH",
    "DERIVED",
    "UNKNOWN",
    "PHASE3-DIAGNOSTIC",
    "PHASE3-TIERB-CORE",
    "PHASE3-SENSITIVITY",
    "PHASE4-FORENSIC",
    "PHASE4-BASELINE",
    "PHASE4-DIAGNOSTIC",
    "PHASE5-DIAGNOSTIC",
]
RESULT_LABELS: tuple[str, ...] = get_args(ResultLabel)
PHASE2_RESULT_LABELS: tuple[str, ...] = (
    "PAPER-REPORTED",
    "PHASE2-SMOKE",
    "LOCAL-MICROBENCH",
    "DERIVED",
    "UNKNOWN",
)
PHASE3_RESULT_LABELS: tuple[str, ...] = (
    *PHASE2_RESULT_LABELS,
    "PHASE3-DIAGNOSTIC",
    "PHASE3-TIERB-CORE",
    "PHASE3-SENSITIVITY",
)


def canonical_result_label(value: str) -> str:
    """Read-side migration for committed records: an exact label, or the historical annotated form
    ``"<LABEL> (<annotation>)"`` (one Phase-2 record uses it), maps to ``<LABEL>``. Writers always emit exact labels."""
    head = value.split(" (", 1)[0] if value.endswith(")") else value
    if head not in RESULT_LABELS:
        raise ValueError(f"{value!r} is not a result label of the current schema {RESULT_LABELS}")
    return head


class ConfigError(ValueError):
    """Raised for any invalid, missing or unknown configuration value."""


# --------------------------------------------------------------------------------------------------
# Section schemas
# --------------------------------------------------------------------------------------------------


@dataclass
class RunSection:
    name: str
    seed: int = 1234
    output_root: str = "runs"
    device: Literal["cpu", "cuda"] = "cpu"
    deterministic: bool = True
    num_threads: int | None = None
    result_label: ResultLabel = "UNKNOWN"
    # cap the CUDA caching allocator at (free dedicated VRAM - margin) before loading anything, so that an
    # over-sized job raises OOM instead of spilling into WDDM shared memory (utils/gpu.py); None = no cap
    allocator_cap_margin_mb: int | None = None


@dataclass
class ModelSection:
    """A pinned pretrained model (``kind='hf'``) or a synthetic tiny LLaMA (``kind='tiny_llama'``)."""

    kind: Literal["hf", "tiny_llama"] = "hf"
    id: str = ""
    revision: str = ""
    license: str = ""
    dtype: Literal["float32", "bfloat16", "float16"] = "float32"
    quantization: Literal["none", "int8", "nf4"] = "none"
    attn_implementation: Literal["eager", "sdpa"] = "eager"
    gradient_checkpointing: bool = False
    allow_patterns: list[str] = field(default_factory=list)
    local_files_only: bool = False
    padding_side: Literal["left", "right"] = "left"
    pad_token_id: int | None = None
    # tiny_llama only (synthetic CPU fixture; weights are random but seeded)
    tiny_hidden_size: int = 128
    tiny_intermediate_size: int = 256
    tiny_num_layers: int = 2
    tiny_num_heads: int = 4
    tiny_vocab_size: int = 128
    tiny_max_position_embeddings: int = 256
    tiny_init_seed: int = 0

    def validate(self, path: str) -> None:
        if self.kind == "hf":
            if not self.id or not self.revision:
                raise ConfigError(f"{path}: kind='hf' requires both 'id' and a pinned 'revision'")
            if len(self.revision) < 40:
                raise ConfigError(f"{path}.revision must be a full 40-char commit SHA, got {self.revision!r}")


@dataclass
class LoRASection:
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    bias: Literal["none"] = "none"
    init: Literal["kaiming_uniform_A_zero_B"] = "kaiming_uniform_A_zero_B"
    init_seed: int = 1234

    def validate(self, path: str) -> None:
        if self.r <= 0 or self.alpha <= 0:
            raise ConfigError(f"{path}: r and alpha must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ConfigError(f"{path}.dropout must be in [0, 1)")


@dataclass
class PromptTemplate:
    prompt_input: str
    prompt_no_input: str


@dataclass
class DataSection:
    source_name: str
    source_url: str
    source_sha256: str
    source_revision: str
    source_license: str
    source_num_records: int
    cache_dir: str
    manifest_path: str
    prompt_template: PromptTemplate
    subset_per_category: int | None = None
    subset_seed: int = 0
    holdout_per_category: int = 10
    num_clients: int = 100
    partition_mode: Literal["dirichlet", "shards"] = "dirichlet"
    dirichlet_alpha: float = 0.5
    min_require_size: int = 40
    shards_per_client: int = 2
    partition_seed: int = 42
    d1_fraction: float = 0.3
    d1d2_split_seed: int = 42
    cutoff_len: int = 512
    train_on_inputs: bool = True
    add_eos_token: bool = True
    heldout_max_examples: int | None = None

    def validate(self, path: str) -> None:
        if not 0.0 < self.d1_fraction < 1.0:
            raise ConfigError(f"{path}.d1_fraction must be in (0, 1)")
        if self.num_clients <= 0:
            raise ConfigError(f"{path}.num_clients must be positive")


@dataclass
class LocalTrainSection:
    epochs: int = 1
    batch_size: int = 32
    micro_batch_size: int = 16
    learning_rate: float = 1.5e-4
    optimizer: Literal["adamw_torch"] = "adamw_torch"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.0
    lr_scheduler: Literal["linear"] = "linear"
    warmup_steps: int = 0
    max_grad_norm: float | None = 1.0
    precision: Literal["fp32", "bf16"] = "fp32"
    loss_reduction: Literal["token_mean_per_microbatch"] = "token_mean_per_microbatch"
    accumulation_normalization: Literal["mean_over_group"] = "mean_over_group"
    partial_accumulation: Literal["flush_at_epoch_end"] = "flush_at_epoch_end"
    pad_to_multiple_of: int | None = 8
    reset_optimizer_each_round: bool = True
    # ``physical_microbatch``: each micro-batch is one forward/backward (Phase 2/3). ``virtual_paper_microbatch``
    # (Phase 4): each LOGICAL micro-batch of ``micro_batch_size`` examples is collated exactly like a real batch
    # (same padding length, labels, masks) and streamed through the model in chunks of ``physical_chunk_size``
    # rows; the summed chunk gradients equal the gradient of the logical micro-batch's token-mean loss.
    microbatch_mode: Literal["physical_microbatch", "virtual_paper_microbatch"] = "physical_microbatch"
    physical_chunk_size: int | None = None
    # held-out loss micro-batch (None = micro_batch_size, the Phase-2 behaviour); the held-out loss depends on
    # padding through the left-padding first-token label, so it is fixed explicitly for comparisons
    eval_micro_batch_size: int | None = None

    def validate(self, path: str) -> None:
        if self.batch_size % self.micro_batch_size != 0:
            raise ConfigError(f"{path}: batch_size must be a multiple of micro_batch_size")
        if not self.reset_optimizer_each_round:
            raise ConfigError(f"{path}: only Shepherd-compatible optimizer reset is implemented in Phase 2")
        if self.microbatch_mode == "virtual_paper_microbatch":
            if self.physical_chunk_size is None or not 1 <= self.physical_chunk_size <= self.micro_batch_size:
                raise ConfigError(
                    f"{path}: virtual_paper_microbatch needs 1 <= physical_chunk_size <= micro_batch_size"
                )
        elif self.physical_chunk_size is not None:
            raise ConfigError(f"{path}: physical_chunk_size only applies to virtual_paper_microbatch")
        if self.eval_micro_batch_size is not None and self.eval_micro_batch_size <= 0:
            raise ConfigError(f"{path}: eval_micro_batch_size must be positive")


@dataclass
class FederatedSection:
    num_rounds: int = 20
    client_fraction: float = 0.05
    sampler: Literal["shepherd_round_seeded"] = "shepherd_round_seeded"
    aggregation: Aggregation = "sample_weighted_mean"
    representation: Representation = "adapter_state"
    client_split: Literal["d1", "d2", "all"] = "d2"
    pooled: bool = False  # centralized training: one virtual client holding the union of all clients' data
    heldout_eval_every: int = 1
    stop_after_round: int | None = None

    def validate(self, path: str) -> None:
        if not 0.0 < self.client_fraction <= 1.0:
            raise ConfigError(f"{path}.client_fraction must be in (0, 1]")
        if self.pooled and self.client_fraction != 1.0:
            raise ConfigError(f"{path}: pooled (centralized) training requires client_fraction = 1.0")


@dataclass
class CodecSection:
    type: CodecType = "none"
    layout: str = "layer_major_qkvo_AtB"
    ae_checkpoint: str | None = None
    latent_dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    device: str | None = None
    noise_sigma: float | None = None
    mean_path: str | None = None

    def validate(self, path: str) -> None:
        if self.type == "autoencoder" and not self.ae_checkpoint:
            raise ConfigError(f"{path}: codec 'autoencoder' requires ae_checkpoint")
        if self.type == "gaussian_noise" and (self.noise_sigma is None or self.noise_sigma < 0):
            raise ConfigError(f"{path}: codec 'gaussian_noise' requires noise_sigma >= 0")
        if self.type == "constant_mean" and not self.mean_path:
            raise ConfigError(f"{path}: codec 'constant_mean' requires mean_path")


@dataclass
class TGAPSection:
    source: TGAPSource = "local_pretrain"
    representation: Representation = "adapter_state"
    num_time_steps: int = 20
    client_fraction: float = 0.05
    client_split: Literal["d1", "d2", "all"] = "d1"
    layout: str = "layer_major_qkvo_AtB"
    # local_pretrain only: which K clients run the local trajectories. ``shepherd_round0`` uses the clients
    # the FL sampler selects in round 0, so time index 0 matches the federated schedule exactly.
    local_client_selection: Literal["seeded_random", "shepherd_round0"] = "seeded_random"


@dataclass
class AESection:
    arch: Literal["resnet3"] = "resnet3"
    stem_kernel: int = 3
    stem_channels: int = 1
    down_channels: list[int] = field(default_factory=lambda: [2, 4, 8, 16, 32, 64])
    num_res_blocks: int = 3
    final_kernel: int = 7
    norm: Literal["batch"] = "batch"
    output_activation: Literal["tanh"] = "tanh"
    representation: Representation = "adapter_state"
    layout: str = "layer_major_qkvo_AtB"
    batch_size: int = 4
    iterations: int = 600
    learning_rate: float = 2e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.0
    split: Literal["temporal", "random"] = "temporal"
    val_fraction: float = 0.2
    split_seed: int = 0
    init_seed: int = 0
    eval_every: int = 50
    # ``final_and_best_val`` also keeps the AE with the lowest D1-validation loss (evaluated at multiples of
    # eval_every); compressor selection on D1 validation never touches D2 or benchmark results
    checkpoint_policy: Literal["final", "final_and_best_val"] = "final"
    normalization: NormalizationMode = "none"

    def validate(self, path: str) -> None:
        if self.eval_every <= 0 or self.iterations <= 0 or self.batch_size <= 0:
            raise ConfigError(f"{path}: eval_every, iterations and batch_size must be positive")


@dataclass
class BenchmarkSpec:
    name: Literal["ceval", "mmlu"]
    split: str
    num_shots: int = 5
    subjects: list[str] | None = None
    limit_per_subject: int | None = None
    # seeded stratified subset: ceil(fraction * n_subject) questions of every subject, chosen by a permutation
    # keyed on (subset_seed, benchmark, split, subject) and kept in dataset order; the qids are in the predictions
    subset_fraction: float | None = None
    subset_seed: int = 0

    def validate(self, path: str) -> None:
        if self.subset_fraction is not None:
            if not 0.0 < self.subset_fraction <= 1.0:
                raise ConfigError(f"{path}: subset_fraction must be in (0, 1]")
            if self.limit_per_subject is not None:
                raise ConfigError(f"{path}: subset_fraction and limit_per_subject are exclusive")


@dataclass
class EvalSection:
    protocol: Literal["reference_eval_v1"] = "reference_eval_v1"
    benchmarks: list[BenchmarkSpec] = field(default_factory=list)
    ceval_repo: str = "ceval/ceval-exam"
    ceval_revision: str = ""
    mmlu_repo: str = "cais/mmlu"
    mmlu_revision: str = ""
    max_context: int | None = None
    max_batch_tokens: int = 16384
    max_batch_size: int = 32
    max_batch_attention: int | None = 4_000_000  # B * L_max^2 bound (scoring numerics are unaffected)
    save_predictions: bool = True


@dataclass
class ExperimentConfig:
    run: RunSection
    model: ModelSection | None = None
    lora: LoRASection | None = None
    data: DataSection | None = None
    local_train: LocalTrainSection | None = None
    federated: FederatedSection | None = None
    codec: CodecSection | None = None
    tgap: TGAPSection | None = None
    autoencoder: AESection | None = None
    eval: EvalSection | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def sha256(self) -> str:
        return canonical_json_sha256(self.to_dict())

    def require(self, *names: str) -> None:
        missing = [n for n in names if getattr(self, n) is None]
        if missing:
            raise ConfigError(f"config section(s) required for this command: {missing}")


# --------------------------------------------------------------------------------------------------
# Strict construction from plain dicts
# --------------------------------------------------------------------------------------------------


def _type_name(tp: Any) -> str:
    return getattr(tp, "__name__", repr(tp))


def _coerce(tp: Any, value: Any, path: str) -> Any:
    origin = get_origin(tp)
    if tp is Any:
        return value
    if origin in (Union, types.UnionType):
        args = get_args(tp)
        if value is None:
            if type(None) in args:
                return None
            raise ConfigError(f"{path}: None is not allowed")
        errors = []
        for arg in args:
            if arg is type(None):
                continue
            try:
                return _coerce(arg, value, path)
            except ConfigError as exc:
                errors.append(str(exc))
        raise ConfigError(f"{path}: value {value!r} matches none of {args}: {errors}")
    if origin is Literal:
        allowed = get_args(tp)
        if value not in allowed:
            raise ConfigError(f"{path}: {value!r} not in allowed values {allowed}")
        return value
    if origin in (list, typing.List):  # noqa: UP006
        if not isinstance(value, list):
            raise ConfigError(f"{path}: expected a list, got {type(value).__name__}")
        (inner,) = get_args(tp) or (Any,)
        return [_coerce(inner, v, f"{path}[{i}]") for i, v in enumerate(value)]
    if origin in (dict, typing.Dict):  # noqa: UP006
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected a mapping, got {type(value).__name__}")
        _, inner = get_args(tp) or (str, Any)
        return {str(k): _coerce(inner, v, f"{path}.{k}") for k, v in value.items()}
    if is_dataclass(tp):
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected a mapping for {_type_name(tp)}, got {type(value).__name__}")
        return build_dataclass(tp, value, path)
    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path}: expected bool, got {value!r}")
        return value
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{path}: expected int, got {value!r}")
        return value
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path}: expected float, got {value!r}")
        return float(value)
    if tp is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path}: expected str, got {value!r}")
        return value
    raise ConfigError(f"{path}: unsupported config type {tp!r}")


def build_dataclass(cls: type, data: dict[str, Any], path: str = "config") -> Any:
    hints = get_type_hints(cls)
    names = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - names)
    if unknown:
        raise ConfigError(f"{path}: unknown key(s) {unknown} for {cls.__name__}")
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name in data:
            kwargs[f.name] = _coerce(hints[f.name], data[f.name], f"{path}.{f.name}")
        elif f.default is MISSING and f.default_factory is MISSING:
            raise ConfigError(f"{path}: missing required key '{f.name}'")
    obj = cls(**kwargs)
    validate = getattr(obj, "validate", None)
    if callable(validate):
        validate(path)
    return obj


# --------------------------------------------------------------------------------------------------
# YAML loading with inheritance and overrides
# --------------------------------------------------------------------------------------------------


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` into a copy of ``base`` (mappings merge, everything else replaces)."""
    out = copy.deepcopy(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_yaml_tree(path: Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.resolve()
    if path in stack:
        raise ConfigError(f"circular config inheritance: {' -> '.join(map(str, (*stack, path)))}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    parents = raw.pop("inherit", []) or []
    if isinstance(parents, str):
        parents = [parents]
    merged: dict[str, Any] = {}
    for parent in parents:
        merged = deep_merge(merged, _load_yaml_tree((path.parent / parent), (*stack, path)))
    return deep_merge(merged, raw)


def apply_overrides(data: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply ``a.b.c=value`` overrides; values are parsed as YAML scalars/lists."""
    out = copy.deepcopy(data)
    for item in overrides or []:
        if "=" not in item:
            raise ConfigError(f"override must look like key.path=value, got {item!r}")
        key, raw_value = item.split("=", 1)
        value = yaml.safe_load(raw_value)
        node = out
        parts = key.strip().split(".")
        for p in parts[:-1]:
            if node.get(p) is None:
                node[p] = {}
            if not isinstance(node[p], dict):
                raise ConfigError(f"override {item!r}: '{p}' is not a mapping")
            node = node[p]
        node[parts[-1]] = value
    return out


def load_config(path: str | Path, overrides: list[str] | None = None) -> ExperimentConfig:
    data = apply_overrides(_load_yaml_tree(Path(path)), overrides)
    return build_dataclass(ExperimentConfig, data, "config")


def config_from_dict(data: dict[str, Any]) -> ExperimentConfig:
    return build_dataclass(ExperimentConfig, copy.deepcopy(data), "config")


def dump_config_yaml(cfg: ExperimentConfig) -> str:
    return yaml.safe_dump(cfg.to_dict(), sort_keys=True, allow_unicode=True, default_flow_style=False)


# --------------------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------------------


def repo_root() -> Path:
    """Repository root (directory containing pyproject.toml), falling back to the CWD."""
    here = Path(__file__).resolve()
    for cand in here.parents:
        if (cand / "pyproject.toml").exists() and (cand / "src").exists():
            return cand
    return Path.cwd()


_ENV_DEFAULT_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*):-([^}]*)\}")


def expand_env(value: str) -> str:
    """Expand ``${NAME:-default}`` (default used when NAME is unset/empty), then ``${NAME}``/``$NAME`` and ``~``."""
    out = _ENV_DEFAULT_RE.sub(lambda m: os.environ.get(m.group(1)) or m.group(2), value)
    return os.path.expandvars(os.path.expanduser(out))


def resolve_path(value: str | Path, base: Path | None = None) -> Path:
    """Expand env vars (incl. ``${NAME:-default}``); relative paths are resolved against ``base`` (default: repo root)."""
    p = Path(expand_env(str(value)))
    if not p.is_absolute():
        p = (base or repo_root()) / p
    return p
