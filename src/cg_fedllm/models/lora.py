"""Explicit LoRA attachment and adapter-state access.

Every scientifically relevant PEFT argument is passed explicitly from :class:`LoRASection`; nothing
depends on PEFT defaults. LoRA initialisation (PEFT ``init_lora_weights=True``: Kaiming-uniform A with
``a=sqrt(5)``, zero B) draws from a torch generator seeded with ``derive_seed(init_seed, 'lora_init')``
inside ``fork_rng`` -- a reproducibility choice that Shepherd does not make (it leaves the initial A
unseeded), recorded in docs/deviations.md.
"""

from __future__ import annotations

import re

import torch

from cg_fedllm.config import LoRASection
from cg_fedllm.models.adapter import AdapterState, AdapterStateError, canonical_order
from cg_fedllm.utils.seeding import derive_seed

PEFT_LORA_RE = re.compile(
    r"(?:^|\.)layers\.(\d+)\.(self_attn|mlp)\.([A-Za-z0-9_]+)\.lora_(A|B)\.default\.weight$"
)


def canonical_key(peft_param_name: str) -> str | None:
    m = PEFT_LORA_RE.search(peft_param_name)
    if not m:
        return None
    return f"layers.{m.group(1)}.{m.group(2)}.{m.group(3)}.lora_{m.group(4)}.weight"


def attach_lora(model, cfg: LoRASection):
    from peft import LoraConfig, get_peft_model

    lcfg = LoraConfig(
        r=cfg.r,
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.dropout,
        target_modules=list(cfg.target_modules),
        bias=cfg.bias,
        task_type="CAUSAL_LM",
        init_lora_weights=True,
        use_rslora=False,
        use_dora=False,
        fan_in_fan_out=False,
        modules_to_save=None,
    )
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        seed = derive_seed(cfg.init_seed, "lora_init")
        torch.manual_seed(seed)
        peft_model = get_peft_model(model, lcfg, autocast_adapter_dtype=True)
    assert_only_lora_trainable(peft_model)
    return peft_model


def lora_parameters(peft_model) -> dict[str, torch.nn.Parameter]:
    """canonical key -> LoRA parameter; fails if any trainable parameter is not a mapped LoRA factor."""
    params: dict[str, torch.nn.Parameter] = {}
    for name, p in peft_model.named_parameters():
        key = canonical_key(name)
        if key is None:
            if p.requires_grad:
                raise AdapterStateError(f"trainable non-LoRA parameter found: {name}")
            continue
        if key in params:
            raise AdapterStateError(f"duplicate canonical key {key}")
        params[key] = p
    if not params:
        raise AdapterStateError("model has no LoRA parameters")
    return {k: params[k] for k in canonical_order(params)}


def assert_only_lora_trainable(peft_model) -> None:
    for name, p in peft_model.named_parameters():
        if p.requires_grad and canonical_key(name) is None:
            raise AdapterStateError(f"parameter {name} is trainable but is not a LoRA factor")
        if canonical_key(name) is not None and not p.requires_grad:
            raise AdapterStateError(f"LoRA parameter {name} is frozen")


def get_adapter_state(params: dict[str, torch.nn.Parameter]) -> AdapterState:
    return AdapterState({k: p.detach().to("cpu", torch.float32).clone() for k, p in params.items()})


@torch.no_grad()
def set_adapter_state(params: dict[str, torch.nn.Parameter], state: AdapterState) -> None:
    if sorted(params) != sorted(state.tensors):
        raise AdapterStateError("adapter state keys do not match the model's LoRA parameters")
    for k, p in params.items():
        t = state.tensors[k]
        if tuple(t.shape) != tuple(p.shape):
            raise AdapterStateError(f"{k}: shape {tuple(t.shape)} != parameter shape {tuple(p.shape)}")
        p.copy_(t.to(device=p.device, dtype=p.dtype))


def base_parameter_fingerprint(peft_model) -> dict[str, float]:
    """Cheap fingerprint (sum and sum of squares, float64) of all frozen base parameters."""
    s = torch.zeros((), dtype=torch.float64)
    s2 = torch.zeros((), dtype=torch.float64)
    n = 0
    for name, p in peft_model.named_parameters():
        if canonical_key(name) is None:
            d = p.detach().double()
            s += d.sum().cpu()
            s2 += (d * d).sum().cpu()
            n += p.numel()
    return {"sum": float(s), "sum_sq": float(s2), "numel": float(n)}
