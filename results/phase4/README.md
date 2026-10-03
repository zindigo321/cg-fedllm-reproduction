# Phase 4 results (Tier B, Qwen1.5-1.8B, scientific seed 1)

Every JSON file carries a result label: `PHASE4-FORENSIC`, `PHASE4-BASELINE`, `PHASE4-DIAGNOSTIC` or `DERIVED`.
* Protocol: `docs/phase4_preregistration.md`.
* Evidence audit: `docs/representation_forensics.md`.
* Findings: `docs/phase4_findings.md`.

Gradient dumps, snapshots, adapters and AE checkpoints stay outside Git (`CGFED_RUNS`). They are referenced by SHA-256 or run path.

None of R2-R4 is the paper's representation. Passing the F6 screen authorises no FAF run.

| File | Content | Label |
|---|---|---|
| `microbatch/f1_llama160m_validation.json` | F1 GPU validation: virtual paper micro-batch (16 x 2, chunks 1/2/16) vs physical 16, noise floor, dropout-on reproducibility | PHASE4-DIAGNOSTIC |
| `microbatch/f1_qwen_validation.json` | F1 on Qwen: micro-batch 1 and virtual chunk 2 vs virtual chunk 1 on one real batch | PHASE4-DIAGNOSTIC |
| `forensics/f4_federated_state_stats.json` | F2-F4 on the Phase-3 `federated_pretrain` snapshots: gauge diagnostics, R2 statistics, R3 rank / retained energy, gauge-invariance demo | PHASE4-FORENSIC |
| `forensics/f4_local_state_stats.json` | the same on the `local_pretrain` snapshots | PHASE4-FORENSIC |
| `microbatch/f1_*_validation_rerun_clean.json` | the same two validations rerun from a clean commit (`3c73e66`); every deterministic field bitwise identical to the originals | PHASE4-DIAGNOSTIC |
| `forensics/f5_gradient_forensics.json` | F5: 2 D1 rounds x 5 clients (20 optimizer steps); summary families, clipping, first-order check, bitwise check against Phase 3 | PHASE4-FORENSIC |
| `forensics/f5_distribution_shape.json` | F5 addendum: value-distribution shape (std, excess kurtosis, quantiles) of the gradient / update families | PHASE4-FORENSIC |
| `screen/f6_r0_r1_phase3_reuse.json` | F6: the frozen Phase-3 R0/R1 reports read against S1-S7 (nothing retrained) | DERIVED |
| `screen/f6_r2_balanced_effective_state.json` | F6 screen of R2 (`none`): AE curve, validation/train metrics, gate | PHASE4-FORENSIC |
| `screen/f6_r3_balanced_effective_delta_r8.json` | F6 screen of R3 (`none`), incl. product vs the exact dM | PHASE4-FORENSIC |
| `screen/f6_r4_mean_step_gradient.json` | F6 screen of R4 (`none`; 5 + 5 snapshots), incl. gradient-space aggregation | PHASE4-FORENSIC |
| `baseline/seed1_baseline.json` | F7: per-round trajectories of LoRA-FT, FAF-Identity and Cent; communication; the Identity regression (13 checks) | PHASE4-BASELINE |
| `eval/eval_cost_full.json`, `eval/eval_cost_timing_sample.json` | F8 cost plans (requests, padded tokens, batches) | PHASE4-DIAGNOSTIC |
| `eval/f8_timing_projection.json` | F8 decision: projected full time per model, rule, outcome `full` | DERIVED |
| `eval/f8_eval_base_full.json`, `eval/f8_eval_lora_ft_full.json` | F8: full MMLU test + C-Eval val, 5-shot, `reference_eval_v1`, bf16 (source-file SHA-256 inside) | PHASE4-BASELINE |
| `eval/f8_paired_comparison.json` | F8: Base vs LoRA-FT per-question paired counts and exact McNemar p | DERIVED |

**Outcome.**
* No representation passes F6, so no operational compressor run was started.
* F7: the seed-1 baseline is complete, and the Identity regression is bitwise.
* F8: full evaluation.
