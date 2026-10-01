"""F7 Identity regression: two federated run directories must agree bitwise on every recorded quantity.

Compared per round: the selected clients, sample counts, global adapter hash, raw A/B norms, effective (s B A)
norms, update norms, held-out loss and logical bytes; per client: start/end hashes, step/token counts and losses;
for a codec run, the codec round trip. Memory, throughput and wall times measure the machine, not the
computation, and are not compared. A safetensors file hash is reported but not gated: the metadata map in the
file header is serialised in a per-process order, so identical tensors can give different file bytes.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

from cg_fedllm.models.adapter import AdapterState
from cg_fedllm.utils.io import read_json

ROUND_FIELDS = ("round", "selected_clients", "sample_counts", "global_hash", "global_l2_sq", "effective_global", "update_norms", "heldout",
                "aggregation", "representation", "uplink_logical_bytes", "uplink_raw_fp32_bytes", "downlink_logical_bytes")
CLIENT_FIELDS = ("client_id", "start_hash", "end_hash", "num_samples", "num_optimizer_steps", "num_micro_batches", "num_forward_chunks",
                 "num_label_tokens", "num_padded_tokens", "first_step_loss", "last_step_loss", "mean_loss")


def _flat(d: dict | None, pre: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in (d or {}).items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{pre}{k}."))
        else:
            out[f"{pre}{k}"] = v
    return out


def _rounds(d: Path) -> list[dict]:
    out, t = [], 0
    while (d / "rounds" / f"r{t:04d}" / "DONE").exists():
        out.append(read_json(d / "rounds" / f"r{t:04d}" / "round.json"))
        t += 1
    return out


def _sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def compare_runs(dir_a: Path, dir_b: Path, expected_config_differences: tuple[str, ...] = ("codec.type",)) -> dict[str, Any]:
    dir_a, dir_b = Path(dir_a), Path(dir_b)
    ra, rb = _rounds(dir_a), _rounds(dir_b)
    mismatches: list[dict[str, Any]] = []
    per_field = {f: True for f in ROUND_FIELDS}
    clients_ok = codec_round_trip_ok = True
    for x, y in zip(ra, rb):
        t = x["round"]
        for f in ROUND_FIELDS:
            if x.get(f) != y.get(f):
                per_field[f] = False
                mismatches.append({"round": t, "field": f, "a": x.get(f), "b": y.get(f)})
        if len(x["clients"]) != len(y["clients"]):
            clients_ok = False
            mismatches.append({"round": t, "field": "num_clients", "a": len(x["clients"]), "b": len(y["clients"])})
        for cx, cy in zip(x["clients"], y["clients"]):
            for f in CLIENT_FIELDS:
                if cx.get(f) != cy.get(f):
                    clients_ok = False
                    mismatches.append({"round": t, "client": cx.get("client_id"), "field": f, "a": cx.get(f), "b": cy.get(f)})
            for side, c in (("a", cx), ("b", cy)):
                if "recovered_hash" in c and not (c.get("recovered_equals_local_bitwise") is True and c["recovered_hash"] == c["end_hash"]):
                    codec_round_trip_ok = False
                    mismatches.append({"round": t, "client": c.get("client_id"), "field": f"codec_round_trip_{side}", "value": {k: c.get(k) for k in ("recovered_hash", "end_hash", "state_relative_l2_error")}})
    sa, sb = read_json(dir_a / "summary.json"), read_json(dir_b / "summary.json")
    fa, fb = AdapterState.load(dir_a / "final_adapter.safetensors"), AdapterState.load(dir_b / "final_adapter.safetensors")
    ia, ib = AdapterState.load(dir_a / "initial_adapter.safetensors"), AdapterState.load(dir_b / "initial_adapter.safetensors")
    config: dict[str, Any] | None = None
    if (dir_a / "config.resolved.yaml").exists() and (dir_b / "config.resolved.yaml").exists():
        ca = _flat(yaml.safe_load((dir_a / "config.resolved.yaml").read_text(encoding="utf-8")))
        cb = _flat(yaml.safe_load((dir_b / "config.resolved.yaml").read_text(encoding="utf-8")))
        diff = {k: [ca.get(k), cb.get(k)] for k in sorted(set(ca) | set(cb)) if ca.get(k) != cb.get(k)}
        hashes = [read_json(d / "config.sha256.json").get("config_sha256") if (d / "config.sha256.json").exists() else None for d in (dir_a, dir_b)]
        config = {"config_sha256": hashes, "resolved_config_differences": diff, "expected_differences": list(expected_config_differences),
                  "only_expected_differences": set(diff) <= set(expected_config_differences)}
    checks = {
        "same_number_of_rounds": len(ra) == len(rb) == sa["rounds_completed"] == sb["rounds_completed"] > 0,
        "selected_clients_every_round": per_field["selected_clients"] and per_field["sample_counts"],
        "heldout_loss_every_round": per_field["heldout"],
        "global_adapter_hash_every_round": per_field["global_hash"],
        "final_adapter_bitwise_equal": fa.equal(fb) and sa["final_global_hash"] == sb["final_global_hash"] == fa.sha256() == fb.sha256(),
        "initial_adapter_bitwise_equal": ia.equal(ib),
        "A_B_norms_every_round": per_field["global_l2_sq"],
        "effective_BA_norms_every_round": per_field["effective_global"],
        "update_norms_every_round": per_field["update_norms"],
        "client_records_every_round": clients_ok,
        "codec_round_trip_bitwise": codec_round_trip_ok,
        "other_round_fields": all(per_field[f] for f in ("round", "aggregation", "representation", "uplink_logical_bytes", "uplink_raw_fp32_bytes", "downlink_logical_bytes")),
        "only_expected_config_differences": None if config is None else config["only_expected_differences"],
    }
    return {
        "pair": [dir_a.name, dir_b.name],
        "rounds_compared": min(len(ra), len(rb)),
        "checks": checks,
        "validated": all(v for v in checks.values() if v is not None) and not mismatches,
        "mismatches": mismatches[:50],
        "num_mismatches": len(mismatches),
        "first_mismatch_round": min((m["round"] for m in mismatches), default=None),
        "final_global_hash": [sa["final_global_hash"], sb["final_global_hash"]],
        "final_adapter_file_sha256": [_sha256_file(dir_a / "final_adapter.safetensors"), _sha256_file(dir_b / "final_adapter.safetensors")],
        "final_relative_l2": fa.relative_l2_diff(fb),
        "config": config,
        # backward-compatible keys (Phase-4 baseline-summary v1)
        "round_hashes_equal": per_field["global_hash"] and len(ra) == len(rb),
        "final_adapter_bitwise_equal": fa.equal(fb),
        "heldout_trajectories_equal": per_field["heldout"] and len(ra) == len(rb),
    }
