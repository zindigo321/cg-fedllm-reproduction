"""What a client transmits (reviewer decision R2).

``adapter_state``  -- the complete local LoRA A/B state after local training (what Shepherd uploads).
``adapter_delta``  -- local post-training state minus the global adapter state from the start of that
                      client round; the server adds the round-start state back.

Phase 2 does NOT claim to know which one the CG-FedLLM paper used (see docs/evidence_ledger.md, L16).
Note: with ``adapter_delta`` the identity round trip ``start + (end - start)`` is exact only up to
float32 rounding; with ``adapter_state`` it is bitwise exact.
"""

from __future__ import annotations

from cg_fedllm.models.adapter import AdapterState

REPRESENTATIONS = ("adapter_state", "adapter_delta")


def to_representation(end: AdapterState, start: AdapterState, mode: str) -> AdapterState:
    if mode == "adapter_state":
        return end
    if mode == "adapter_delta":
        return end.sub(start)
    raise ValueError(f"unknown representation {mode!r}")


def recover_state(rep: AdapterState, start: AdapterState, mode: str) -> AdapterState:
    if mode == "adapter_state":
        return rep
    if mode == "adapter_delta":
        return start.add(rep)
    raise ValueError(f"unknown representation {mode!r}")
