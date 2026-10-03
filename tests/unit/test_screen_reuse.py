"""Phase-4 F6 R0/R1 reuse: frozen Phase-3 reports read against the S1-S7 criteria (no recomputation)."""

from __future__ import annotations

import copy
import json
import math

from cg_fedllm.cli import main
from cg_fedllm.forensics.screen import phase3_gate_mapping
from tests.conftest import REPO

P3 = REPO / "results" / "phase3" / "ae_viability"


def test_mapping_reads_the_committed_phase3_reports(tmp_path):
    out = tmp_path / "reuse.json"
    assert (
        main(
            [
                "screen-reuse",
                "--phase3",
                f"R0_none={(P3 / 'a5_federated_state_none.json').as_posix()}",
                "--phase3",
                f"R1_factor_rms={(P3 / 'a8_federated_delta_factor_rms.json').as_posix()}",
                "--out",
                out.as_posix(),
            ]
        )
        == 0
    )
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res["label"] == "DERIVED"
    r0, r1 = res["candidates"]["R0_none"], res["candidates"]["R1_factor_rms"]
    assert r0["representation"] == "adapter_state" and r1["representation"] == "adapter_delta"
    assert r0["pass"] is False and r1["pass"] is False and r0["phase3_gate_pass"] is False
    assert len(r0["source_sha256"]) == 64
    src = json.loads((P3 / "a8_federated_delta_factor_rms.json").read_text(encoding="utf-8"))["val"][
        "predictors"
    ]["autoencoder_best_val"]
    # a delta is judged by its own product (the innovation), not by the start-dominated reconstructed state
    assert math.isclose(r1["values"]["product_cosine"], src["product"]["innovation"]["cosine"])
    assert math.isclose(
        r1["values"]["product_rel_fro_error"], math.sqrt(src["product"]["innovation"]["rel_sq_error"])
    )


def test_mapping_passes_a_perfect_report():
    p3 = json.loads((P3 / "a5_federated_state_none.json").read_text(encoding="utf-8"))
    good = copy.deepcopy(p3)
    ae = good["val"]["predictors"]["autoencoder_best_val"]
    ae["transmitted"]["all"].update(cosine=0.99, rel_sq_error=0.01)
    for k in ("state", "innovation"):
        ae["product"][k].update(cosine=0.99, rel_sq_error=0.01)
    ae["shift_control"].update(innovation_rel_sq_error_matched=0.01, innovation_rel_sq_error_shifted=2.0)
    good["val"]["predictors"]["train_mean"]["transmitted"]["all"]["rel_sq_error"] = 1.0
    m = phase3_gate_mapping(good)
    assert m["pass"] is True and all(m["criteria"].values())
    ae["finite_outputs"] = False
    assert phase3_gate_mapping(good)["pass"] is False
