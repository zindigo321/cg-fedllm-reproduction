# Phase 3 results (Tier B, Qwen1.5-1.8B, scientific seed 1)

Every JSON file carries a result label: `PHASE3-DIAGNOSTIC`, `PHASE3-SENSITIVITY` or `DERIVED`. The protocol,
metrics and gates were pre-registered in `docs/phase3_preregistration.md`; the findings are in
`docs/phase3_findings.md`. Snapshots, adapters and AE checkpoints stay outside Git (`CGFED_RUNS`). They are
referenced by SHA-256 in the manifests below.

| File | Content | Label |
|---|---|---|
| `calibration/a3_calibration_mb2.json` | A3 realistic-sequence timing/memory at micro-batch 2 (rule triggered) | PHASE3-DIAGNOSTIC |
| `calibration/a3_calibration_mb1.json` | the same at the selected micro-batch 1 | PHASE3-DIAGNOSTIC |
| `tgap/federated_state_seed1_stats.json` | A4 PRIMARY TGAP set: counts, participation, distributions, cosines, split, manifest | DERIVED |
| `tgap/local_state_seed1_stats.json` | A4 SENSITIVITY TGAP set (local_pretrain) | DERIVED |
| `microbatch/microbatch_diag.json` | micro-batch loss-normalisation diagnostic on one fixed batch | PHASE3-DIAGNOSTIC |
| `ae_viability/a5_federated_state_{none,global_rms,factor_rms}.json` | A5 ladder: AE training curve, A1 report (train/val), A6 gate | PHASE3-DIAGNOSTIC |
| `ae_viability/a6_selection.json` | A6 priority rule outcome | PHASE3-DIAGNOSTIC |
| `ae_viability/a8_local_state_none.json` | A8: local_pretrain + adapter_state, normalisation `none` (rule-determined: A6 selected no mode) | PHASE3-SENSITIVITY |
| `ae_viability/a8_federated_delta_factor_rms.json` | A8: federated_pretrain + adapter_delta, `factor_rms` | PHASE3-SENSITIVITY |
| `reference_codes/{federated_state,federated_delta,local_state}.json` | simple codes at the AE's element ratio (interpretation aid, never a gate) | DERIVED |

Outcome: A6 selected no candidate (NO PRIMARY CODEC IS VIABLE); A7 and Phase 3B were not run; both A8 sensitivities fail.
