# Phase 5 P5-A v2 implementation review record

Status: **implementation ready for human code review; post-audit Phase-5 CPU validation passed; real P5-A not run.** This record maps the frozen v2 protocol
and its v2.1 addendum to the code and tests. It contains no Phase-5 result. Synthetic test values are software
validation only. No real input preflight, payload read, real GPU training or AE forward pass on a real input has
taken place. Sections 9-12 hold the acceptance review of 2026-10-05; section 13 holds the reconciliation with the
v2.1 addendum and the F7 synthetic GPU diagnostic of the same day. Section 14 records the external diff audit and
proposed forward fixes. These are AI-assisted engineering reviews, not a human acceptance. The launch draft is
`docs/phase5_p5a_v2_launch_draft.md`.

## 1. Identities (verified locally)

| Item | Value |
|---|---|
| Baseline (`main`, local and remote) | `afc9e333b5b865b32d47f5634aa7db3ddbfa97ef` |
| v1 preregistration (historical, never executed) | `52a0dd4e7bb8521752ac80ff8f77072dc0e0761c`, `docs/phase5_preregistration.md`, blob `00f2477d164b795c4f35f717a3017888baee9b4e`; unchanged |
| **v2 protocol (implemented)** | **`dd4bf3f4348fd3b2b1b2c4ad23e7204fbef8369c`** (parent `52a0dd4`; local branch `worktree-p5a-engineering-support`, not pushed); `docs/phase5_preregistration_v2.md` blob `e80d21aeda653ab6d1339d34f7b88118588855df`; `docs/phase5_v2_audit_response.md` blob `7164034fe7a652f204e7120d59f638f2bb831ffd` |
| **v2.1 addendum (implemented)** | **`33e51fd63c6cd0679e76e12380ae5d418158d698`** (parent `dd4bf3f`; protocol-only, one file; local, not pushed); `docs/phase5_preregistration_v2_1.md` blob `28e05f9ee33ed13386d346fce6e47e9108c937d7`. Protocol identity of every record = the pair (v2, v2.1) |
| Environment | `D:\conda_envs\cgfedllm\python.exe`, Python 3.11.16; `cg_fedllm` imported from this worktree's `src`; numpy 2.4.6, torch 2.14.0+cu130, safetensors 0.8.0, ruff 0.16.9, pytest 9.1.1 |

The protocol SHA, path and blob are cited from code as `P5A_PROTOCOL_COMMIT` / `_PATH` / `_BLOB`, and the addendum
as `P5A_ADDENDUM_COMMIT` / `_PATH` / `_BLOB` (`phase5/p5a.py`); `protocol_identity()` writes both into every P5-A
record. A test compares both working-copy blobs with the frozen blobs. The implementation
commit SHA is not written into any file of that commit; records take it from `git rev-parse HEAD` at execution.

## 2. Code layout

| File | Role |
|---|---|
| `phase5/p5a.py` | frozen values, plan, designations, thresholds, statuses, selection, outcome table and records, config locks, run-root paths |
| `phase5/p5a_inputs.py` | frozen inputs (`FROZEN_MAX`, `FROZEN_SCALE`, `F6_TRAIN_RMS`), R2/R3 construction via `load_states(verify=True)`, restricted R4 reader, exact-max scale check, range check, RMS check, selection setup, preflight evidence |
| `phase5/p5a_metrics.py` | section-9 metrics with `null` + reason, C0-C4 with per-control thresholds, margins and `near_threshold` |
| `phase5/p5a_artifacts.py` | section-13.2 caps, measured footprint, reservation check, no-clobber publication |
| `phase5/p5a_state.py` | persistent attempt states, timing fields, charged time, resource start rule, reserved caps |
| `phase5/p5a_preflight.py` | CPU-only preflight route and the training-time validation of its evidence |
| `phase5/p5a_run.py` | trainer, invocation entry rules, recovery, control procedure, publication order, closure, summaries, production entry points |
| `configs/phase5/p5a_v2_seed1.yaml` | the locked v2 configuration (replaces the uncommitted v1 `p5a_seed1.yaml`, which was never committed or run) |
| `cli.py` | `cgfed p5a-preflight` and `cgfed p5a` (`--closure-only`, `--launch-record`, `--runs-root`) |
| `compression/normalization.py`, `config.py` | new mode `global_exact_maxabs_train` and `train_abs_max`; `global_maxabs_train` unchanged |

Tests: `tests/unit/test_p5a*.py`, the helper `tests/unit/p5a_fixtures.py` and the hard-kill child
`tests/unit/p5a_hard_kill.py`; `test_config.py` gains one test.

## 3. Decision-to-code mapping (v2 §19)

| ID | Implementation | Enforcing test(s) |
|---|---|---|
| D00, D01 | Only v2 code paths exist; `P5A_V1_RUN_NAME` is refused by every entry point; no real run performed | `test_p5a_run::test_entry_rules_v1_root…`, `…production_entry_refuses…`, `test_p5a_preflight::test_production_preflight_refuses…` |
| D02 | no capacity/compressibility claim in code; fixed `DOES_NOT_ESTABLISH`, `MEANING`, `QUALIFIER` texts | `test_p5a::test_aggregation_examples_and_structured_outcome_records` |
| D03 | `DESIGNATION` (R2/R3 primary, R4 secondary); `PLAN` always contains R4; outcome records carry `designation`; summary lists primary first | `test_p5a::test_plan_order…`, `test_p5a_run::test_end_to_end…` |
| D04, D05 | F6 construction; `training_records` selects 80 / 80 / 5 from metadata before any payload | `test_p5a_inputs::test_r2_r3_match…`, `…population_count…` |
| D06 | `AllowlistedGradientReader` (14 files, size + SHA-256 of the bytes read, once each, no listing); **R4 opens no start/end state file** (changed from the v1 reconciliation, which also read them) | `test_p5a_inputs::test_r4_reads_only_the_allowlist…`, `…r4_allowlist_is_required…`, `…preflight_on_a_synthetic_tree…` |
| D07 | primary checks of §3 items 1-4; secondary `train_rms` only (p99.9 no longer computed) | `…index_hash_counts_geometry…`, `…rms_check_reproduces_f6…` |
| D08, D09, D14 | `Normalizer("global_exact_maxabs_train")`; `train_abs_max` exact over every element; float64 scale; fp32 application via `_scale_like` | `test_p5a_normalization::test_exact_max_covers…`, `…new_mode_is_distinct…`, `…application_is_fp32…` |
| D10 | `prepare_representation`: one scale per representation, before selection, shared by both controls | `test_p5a_run::test_end_to_end…` (scale set per representation) |
| D11 | `FROZEN_MAX` / `FROZEN_SCALE`; `|rec - frozen| <= 1e-6 * frozen` inclusive; frozen value applied; mismatch raises "not replaced" | `test_p5a_normalization::test_frozen_values_match…` (against `git show d3c4eb3…`), `…tolerance_is_inclusive…` |
| D11a | `p5a_preflight.run_preflight` and `validate_against_preflight` | `test_p5a_preflight` (all), `test_p5a_run::test_changed_inputs…`, `…entry_rules…` |
| D12 | `range_check` on the fp32 normalised values, float64 bounds `0.95 (1 ± 2e-6)` | `…range_check_boundaries_on_both_sides` |
| D13 | finiteness before the maximum; zero maximum, NaN, wrong count stop | `…degenerate_and_non_finite_populations_stop` |
| D15-D18 | `sort_records`, `control_positions` (integer half-up), `batch_schedule` (`min(4, k)`, no cloning) | `test_p5a::test_selection_*`, `test_p5a_run::test_batch_schedule…` |
| D19, D20 | `PLAN` order; a control outcome never prunes; only the §7 rules prevent a start | `test_p5a_run::test_end_to_end…` (order `[1,4,1,4,1,4]`), `…single_snapshot_failure_does_not_skip…`, `…c0_is_an_integrity_check…` |
| D21, D22 | predictors; C0 with the control's own thresholds; `NOT INFORMATIVE` keeps the control `NOT_STARTED` until closure | `test_p5a_metrics::test_tanh_range_ceiling_and_c0…`, `test_p5a_run::test_c0_is_an_integrity_check…` |
| D23-D27 | `AE_LOCK`, `build_ae` seeded init, Adam settings, exactly 3,000 iterations in production, eval-mode final checkpoint with BatchNorm buffers | `test_p5a_run::test_production_config…`, `…real_resnet_training_and_final_eval_on_cpu` |
| D28, D31, D33 | float64 sums, `math.fsum` pools, `null` + `undefined` reasons, `allow_nan=False` | `test_p5a_metrics::test_per_snapshot_and_pooled…`, `…undefined_and_non_finite…` |
| D29 | `THRESHOLDS["single_snapshot"] = 0.01 / 0.99` wired through `gate` / `ae_decision` / C0 | `test_p5a_metrics::test_c2_c3_inclusive…[single_snapshot]`, `…strict_single_snapshot…`, `test_p5a_run::test_strict_single_snapshot_gate_is_wired…` |
| D30 | subset thresholds 0.50 / 0.90 / 0.8 unchanged | `test_p5a_metrics::test_c2_c3_inclusive…[fixed_subset4]`, `…every_snapshot_must_pass…` |
| D32 | `criterion_subset_mean`: plain inequality, no zero shortcut, no epsilon | `test_p5a_metrics::test_c4_boundary_zero_subset_mean_error…` |
| D34 | `compare`: literal float64 thresholds, inclusive, undefined fails with `fail_reason` | `…c2_c3_inclusive…`, `…near_threshold_flag_uses_four_ulps` |
| D35, D36 | nine statuses; §11.2 table | `test_p5a::test_section11_aggregation_is_exhaustive` (81 cases), `…status_set_is_the_nine…` |
| D37 | `outcome_record` structured fields; outcomes only in the closing summary; no combined pass; `check_wording` removed | `test_p5a::test_aggregation_examples…`, `test_p5a_run::test_c0_is_an_integrity_check…` (no outcomes before closure) |
| D38 | `AUTHORISES` | `test_p5a::test_aggregation_examples…` |
| D39, D40 | `GpuClock` from before `ae.to(device)` to after evaluation; per-iteration, before-evaluation and end-of-evaluation checks; start rule `trainings < 6` and `charged + 1,800 <= 10,800` | `test_p5a_run::test_nonfinite_loss_timeout…`, `…evaluation_overrun…`, `…aggregate_time_reservation…`, `…per_training_limit_and_time_ceiling…` |
| D41 | `p5a_state.timing`: measured / status / charged / reason; 1,800 s only for an unobserved death | `…process_death_after_the_start_record…`, `…per_training_limit…` |
| D42 | `control_state` from finalized files; start record before `ae.to(device)`; `recover` publishes end + record without retraining | `…process_death_after_the_start_record…`, `…death_before_the_start_record…`, `…ended_control_without_record…` |
| D43 | `next_invocation`, `invocation_<k>` first (capability check), `summary_<k>`, ≤ 3 invocations, closure rules, `closure_only` | `…fourth_invocation_and_closure_only_step`, `…crash_counts_is_never_retried…`, `…operator_interrupt…` |
| D44, D45 | `CAPS`, `worst_case_bytes() = 18,990,000`; `ArtifactStore.check`/`publish` (cap + `F + cap + Σ caps(Q) <= 20,000,000`); `reserved_caps` | `test_p5a_artifacts` (all), `test_p5a_run::test_byte_reservation_refusal…`, `…checkpoint_is_not_published_unless…` |
| D46, D47 | exclusive temporary + `os.link`; own-temporary cleanup only; no overwrite fallback; partial publication keeps files, `INCOMPLETE (PUBLICATION)`, failure record, invocation stop | `test_p5a_artifacts::test_collision…`, `…failed_write…`, `…no_overwrite_fallback…`, `test_p5a_run::test_ended_control_without_record…` |
| D48 | record fields of §14; preflight record fields | `test_p5a_run::test_end_to_end…`, `test_p5a_preflight::test_preflight_builds_no_ae…` |
| D49 | the synthetic suite of §15 | this table |
| D50 | `--launch-record` is cited by path and SHA-256 only; no authorisation flag exists | `test_p5a_preflight::test_production_preflight_refuses…` |
| D51 | `PHASE5-DIAGNOSTIC` label | `test_p5a::test_phase5_label…`, `test_config::test_result_label…` |

## 4. v1-to-v2 implementation changes

* Normalisation: `global_maxabs_train` p99.9 quantile and frozen p99.9 scales removed from P5-A; exact-max mode and
  `FROZEN_MAX` / `FROZEN_SCALE` from committed F6 `train_max_abs` (`d3c4eb3…`). `global_maxabs_train` is unchanged
  for Phase 4.
* Gates: per-control thresholds; strict 0.01 / 0.99 single-snapshot standard; C4 zero-error shortcut removed;
  structured comparisons with margins.
* Records: strict JSON with `null` + reasons (no `json_safe` strings); no forbidden-word filter.
* Accounting: `attempt_*` renamed `start_*`; four persistent states; recovery instead of an exception on re-entry;
  timing fields; ≤ 3 invocations with `invocation_<k>` / `summary_<k>`; closure records; closure-only step.
* Bytes: the 2 MB reserve / 18 MB split removed; per-artifact caps and the reservation check; one file per
  publication.
* Inputs: R4 no longer opens start/end state files; per-record Phi SHA-256 and identities in the population.
* New CPU preflight route; training refuses missing, failed, stale or disagreeing preflight evidence.
* Run root `phase5_p5a_v2_seed1/`; config `configs/phase5/p5a_v2_seed1.yaml`; the v1 root is refused.

## 5. Engineering choices not spelled out in v2 (flagged for review; no protocol value changed)

1. **Pre-start `PROVENANCE` and `NOT INFORMATIVE` stay `NOT_STARTED`** until closure, as §12 says; their terminal
   records are issued at closure from the last recorded pre-start status (§13.5). A run in which some control is
   only ever pre-start-blocked therefore closes only after invocation 3. This follows the text literally.
2. **Over-cap vs reservation refusal during a started control's publication.** An artifact over its own cap makes the
   control `INCOMPLETE (RESOURCE STOP)` (§13.3); a reservation refusal of a later file of a completed training makes
   it `INCOMPLETE (PUBLICATION)` (§13.4, partial publication). Both close the run as a byte-ceiling stop.
3. **Preflight / training identity**: compared on `execution_commit` and `config_sha256`. The launch-record identity
   is cited and compared between training invocations only (§13.1), because the launch record is updated to cite the
   preflight record after the preflight.
4. **Reported `cuda_initialized`** in the preflight record is the process state, reported truthfully; the preflight
   code never initialises CUDA (tested by patching CUDA initialisation to fail).
5. **`run_metadata.json` of the provenance set** is serialised once per invocation and copied into each control
   directory (3 files per control, one shared cap).

These five choices are assessed in section 11. Choice 2 and N1 (= finding F1) were decided on 2026-10-05 and
frozen in the v2.1 addendum; choice 2 as described above is **superseded** (section 13). Choices 1, 3, 4 and 5 are
recommended for acceptance and remain pending human acceptance.

## 6. Validation (CPU, synthetic; software validation only)

Run from this worktree with `D:\conda_envs\cgfedllm\python.exe -m ...` and `HF_HUB_OFFLINE=1`:

| Command | Exit | Result |
|---|---|---|
| `ruff check src tests scripts` | 0 | all checks passed |
| `ruff format --check src tests scripts` | 0 | 117 files already formatted |
| `pytest -q -m "not gpu and not model"` | 0 | 308 passed, 2 deselected (gpu/model-marked) |
| `git diff --check` | 0 | no whitespace errors |

(Historical, 2026-10-04. Superseded by section 11. That run, like every earlier full CPU run in this environment,
also initialised CUDA through an inherited `peft` import; see section 9, finding F5.)

The CPU ResNet integration tests train the real ResNet-3 for 2-3 iterations on random `[1, 128, 128]` tensors. No
test reads a real payload, writes under `CGFED_RUNS` or downloads anything.

## 7. Remaining limitations

* All validation is CPU and synthetic. GPU behaviour, CUDA determinism under `configure_determinism`, timing and
  memory on the RTX 4060 are unverified. v2 §15 allows a ≤ 50-iteration GPU smoke test on random tensors of the real
  shape; it was not run.
* The real-input checks (index hashes, R2/R3 payload hashes, the 14 R4 hashes, exact maxima, RMS, range) are
  implemented and tested on synthetic fixtures only. The frozen `m_r` / `s_r` are DERIVED from committed F6 JSON and
  are not yet confirmed against the payloads.
* `production_main`, `production_preflight` and `production_closure` are exercised only up to their refusal paths;
  their components are tested end to end with synthetic profiles.

## 8. Future sequence (none authorised yet)

1. Human review of this implementation diff; an authorised implementation commit (no AI trailer).
2. Launch record (`reproduction_protocol.md` §6; v2 §16) citing `dd4bf3f4348fd3b2b1b2c4ad23e7204fbef8369c` and the
   reviewed implementation SHA, `configs/phase5/p5a_v2_seed1.yaml` and its SHA-256, no overrides, the environment,
   the frozen input identities and `m_r` / `s_r`, the selected records and plan order, the ceilings and caps, the
   run root `<CGFED_RUNS>/phase5_p5a_v2_seed1/` confirmed absent, and the planned evidence location.
3. Separate written authorisation for the CPU preflight; then
   `cgfed p5a-preflight --config configs/phase5/p5a_v2_seed1.yaml --runs-root D:\cgfed-runs --launch-record <path>`.
4. A launch-record update citing the preflight record; a separate written authorisation for training; then
   `cgfed p5a --config configs/phase5/p5a_v2_seed1.yaml --runs-root D:\cgfed-runs --launch-record <path>`.
5. Evidence commit in a separate reviewed change (`results/phase5/p5a_v2/`).

## 9. Acceptance review of 2026-10-05: findings and fixes

Scope: the whole uncommitted implementation diff against the frozen v2 text at `dd4bf3f`, including untracked files,
traced along the production path (section 10). Evidence was committed text, Git objects and synthetic fixtures only.
No payload, run-root metadata or index under `CGFED_RUNS` was opened.

| # | Severity | Finding | Provision | Resolution |
|---|---|---|---|---|
| F1 | **material, needs human decision** | Re-entry after an invocation 1 that published nothing. §13.3 says an invocation that could not publish its first record "has published nothing and does not count". But publishing `ledger/invocation_1.json` creates `ledger/` before the link, and a hard kill also leaves a temporary that §13.4 counts and forbids deleting. §13.1 then refuses invocation 1 for any root with a `ledger/` directory, and invocation 2 needs `ledger/invocation_1.json`. The root can never be entered again without deleting something, which §13.4 forbids. | v2 §13.1 (invocation 1 entry), §13.3 (first record not published), §13.4 (temporaries counted, not removed) | Not silently repaired. The code now applies §13.1 literally: it refuses any `ledger/` or control directory, not only files, and it names the conflict (`FIRST_RECORD_CONFLICT`). It deletes nothing. Test: `test_invocation_one_that_published_nothing_blocks_re_entry_pending_review`. Options for a reviewed forward revision are in section 12. Preflight 1 has the same `mkdir`-before-link order (§13.1 "root does not exist"). There the frozen text names only files ("holds only earlier preflight records"); the code refuses a root without a preflight record. |
| F2 | defect, fixed | At a resource closure, `close_run` relabelled a `STARTED` / `ENDED` control (a training that had started) as `INCOMPLETE (RESOURCE STOP)` instead of its §12 recovery status. §7 and §13.4 give `RESOURCE STOP` at closure only to controls that were "not trained", and §12 assigns `INTERRUPTED` / `PUBLICATION` to recovered controls. Only a resource closure was affected; at a normal closure the recovery statuses were already correct. | v2 §12 recovery, §13.5 closure | `recover()` now always uses the §12 statuses; the closure statuses apply only to controls that never started. Test: `test_resource_closure_never_relabels_a_trained_control`. |
| F3 | defect, fixed | The closure-only step did not check that the provenance matched invocation 1. §13.1 requires recovery to name the same protocol, config and launch record. | v2 §13.1, §13.5 | `check_same_run()` is shared by invocations 2-3 and `closure_only()`. Test: `test_fourth_invocation_and_closure_only_step` (a mismatching config is refused; no training and no invocation 4 are granted). |
| F4 | test defect, fixed | Three crash tests used `raise SystemExit`, which runs every `finally` block (including the store's own-temporary cleanup) and the runner's handlers. They did not reproduce a hard kill. | review requirement; v2 §12, §13.4 | New child process `tests/unit/p5a_hard_kill.py` ends with `os._exit` at three real boundaries: during the start-record write, inside `ae.to(device)` after the start record, and after the end record while linking the checkpoint. Its tests show that the surviving temporary is counted and never deleted, that no start is counted before the start record, that a started control is recovered as `INTERRUPTED` with `unknown_process_death` / 1,800 s charged and is never retrained, and that an ended control keeps its completion and becomes `PUBLICATION`. |
| F5 | deviation disclosed; process fixed | **Earlier CPU suite runs initialised CUDA.** `import peft` (peft 0.21.1, imported by `models/lora.py` through the `tiny_bundle` fixture) calls `torch.cuda._lazy_init` while CUDA is visible. The first test to do so was `tests/integration/test_ae_training.py::test_ae_trains_on_snapshots_and_learns_structured_control`. Every full CPU run in this environment before this review, including the 308-pass runs of 2026-10-04 and the first full run of this session, therefore created a CUDA context on the GPU. Those runs did no GPU computation and accessed no data; this is inherited test behaviour, not P5-A code. No P5-A module imports `peft` or `transformers` (verified with `sys.modules` after import). | session boundary (no CUDA initialisation) | All final gates ran with `CUDA_VISIBLE_DEVICES=-1`. A read-only session probe (`torch.cuda.is_initialized()` before and after every test) reported `false` throughout. The repository was not changed for this; whether CI or the default command should hide CUDA is a separate decision. |
| F6 | test gap, fixed | The 1,800 s rule at the end of evaluation was tested only with a subclassed clock directly on `train_and_evaluate`, never through the runner with the real trainer. | v2 §12 evaluation overrun | `test_real_evaluation_overrun_path_publishes_no_checkpoint`: real trainer, real runner, a clock that turns slow when the evaluation codec is built. Result: `RESOURCE STOP`, measured (not clamped) time, no checkpoint, and the invocation continues. |
| F7 | observation, no change | §8.3 makes a deterministic-backend refusal `INCOMPLETE (INTERRUPTED)`. PyTorch documents `ReflectionPad2d` backward on CUDA as an operation that is non-deterministic under `use_deterministic_algorithms(True)`. Whether it warns or raises on the pinned torch 2.14 cannot be checked without CUDA. F6 trained the same architecture with `run.deterministic: true` on this GPU (its records have 61 curve points up to iteration 3,000), which suggests training does not raise. That is historical evidence, not a check of this commit. | v2 §8.3 | Listed as a launch prerequisite: the allowed ≤ 50-iteration GPU smoke test on random tensors (§15), only after explicit authorisation. |
| F9 | defect, fixed | The production preflight did not call `configure_determinism`, while the training invocation calls `configure_determinism(True)` before rebuilding the same inputs. The training invocation requires **exact** equality with the preflight evidence (N3), so differing torch settings across the two processes could refuse every representation spuriously. | v2 §5.4.1 (evidence must agree) | `production_preflight` calls `configure_determinism(True)`, which sets flags only and initialises no CUDA context. Test: `test_production_preflight_route_on_an_empty_synthetic_root` (the real CLI route on an empty temporary `CGFED_RUNS`; fails closed, publishes one failing record, no AE or CUDA). |
| F8 | observation, no change | Measured preflight record size: 141,283 bytes with 80 / 80 / 5 records, production-length paths and hashes, 50 dirty-file entries (synthetic; cap 300,000). The CPU preflight holds the full fp32 populations in memory (about 2.0 GB for R2 or R3: 80 x 12,582,912 bytes), plus the stacked copy the RMS check makes. | v2 §13.2 | Recorded in the launch draft as a resource note. |

Verified without change (selected): R2/R3 select time 0-15 and R4 time 0 from index metadata before any payload is
opened; R4 opens only the 14 frozen files, never `load_client_round` and never a state file. The allowlist,
both index hashes, `m_r`, `s_r` and `train_rms` equal the committed v1/v2 text and the F6 JSON at `d3c4eb3…`
byte for byte. The F6 records' runtime provenance is `e8100e9…` with `dirty: true`. `d3c4eb3…` is only the evidence
commit; its provenance is not repaired or restated here. Other checks: selection; exact max over the whole population;
the frozen scale applied; the 1e-6 and 2e-6 rules; the per-control thresholds; RSE as SSE / SIG; pooled ratios as sums
of sums; C4 as the plain inequality; the 9 x 9 outcome table; caps and reservation; no-clobber publication; the
production locks.

## 10. Production path (traced)

`cgfed p5a-preflight` -> `cli.cmd_p5a_preflight` -> `p5a_preflight.production_preflight`: refuses `--set`; runs
`load_config` and `p5a.check_config` (locks); requires an absolute existing `CGFED_RUNS`; refuses an existing v1 root;
cites the launch record by path and SHA-256; records `git_info`. `run_preflight` -> `check_root_for_preflight` ->
for R2, R3, R4: `build_population` (`training_records`: index SHA-256, uniqueness, count, geometry, layout; then
`load_states(verify=True)` for R2/R3, or `AllowlistedGradientReader` + `load_pre_clip_round` for R4; then
`layout.forward`, shape, dtype, finiteness, Phi SHA-256) -> `prepare_representation` (`frozen_normalizer`: exact
max, 1e-6, frozen scale, `range_check`; `secondary_statistics`; metadata-only selection; `sig = 0`) -> `evidence` ->
publish `preflight/preflight_<k>.json` with the training worst case reserved. No AE, CUDA or start record.

`cgfed p5a` -> `cmd_p5a` -> `production_main` -> `_production_context` (the same refusals; requires CUDA;
`configure_determinism(True)`; VRAM guard; `collect_environment`; provenance set) -> `run_invocation`:
`next_invocation` (§13.1, `check_same_run`) -> publish `invocation_<k>` (capability check) -> `recover` ->
per representation with a `NOT_STARTED` control: `build_population` + `prepare_representation` again, then
`validate_against_preflight` (schema, protocol, execution commit, config, pass, exact evidence equality) ->
`run_control` per control in plan order (anomalous files, `sig = 0`, C0, `resource_state`, reservation, start
record, then `train_and_evaluate`; end record, checkpoint, provenance set, record) -> `finish_invocation` (closure if
due: `close_run`, `outcomes`) -> `summary_<k>`. `--closure-only` -> `production_closure` -> `closure_only`
(no CUDA, no payload, `check_same_run`, recovery, closure, `summary_3`).

The test-only paths (`synthetic_profile`, injected `trainer`, `clock`, `writer`, `linker`, `frozen`) are not
reachable from the CLI. `Profile` refuses an altered `PRODUCTION`, and the production entry points take only
`config_path`, `overrides`, `runs_root`, `command` and `launch_record` (tested).

## 11. The five engineering choices of section 5 (assessment; none approved by this review)

| # | Choice and rationale | Controlling v2 text | Code | Tests | Classification | Recommendation |
|---|---|---|---|---|---|---|
| 1 | A pre-start `PROVENANCE` or `NOT INFORMATIVE` keeps the control `NOT_STARTED`; its terminal record is written at closure from the last pre-start status, so such a run closes only after invocation 3 | §12 "Before a start … not as a terminal record"; §13.5 closure rules | `run_invocation` (`pre`), `_earlier_pre_start`, `close_run` | `test_c0_is_an_integrity_check…` (3 invocations, closure), `test_zero_signal…` | compliant (literal reading) | Accept. Operational cost: a permanently blocked control forces 3 invocations, and every re-entry reads all payloads again. |
| 2 | Over-cap vs reservation refusal while publishing a started control: over its own cap -> `RESOURCE STOP`; reservation refusal of a later file -> `PUBLICATION`; both close the run | §13.3 "artifact larger than its cap … RESOURCE STOP"; §13.4 partial publication -> `PUBLICATION`; "if the cause is the byte ceiling, the run is closed" | `_publication_failed` | `test_checkpoint_is_not_published_unless_remaining_evidence_is_reserved` (reservation -> `PUBLICATION`); the over-cap branch is covered only at store level (`test_cap_and_reservation_are_enforced_before_writing`) | material interpretation | **Human decision.** §13.3 and §13.4 both apply to a refused checkpoint after the end record. The code picks by cause; the alternative is `RESOURCE STOP` for every byte-policy refusal. The scientific meaning is the same either way: both are `INCOMPLETE`, and the outcome is `INCONCLUSIVE` unless the other control fails. |
| 3 | Preflight / training identity compared on `execution_commit` and `config_sha256`, not on the launch record, because the launch record is updated after the preflight to cite it | §5.4.1 "must name the same protocol commit, implementation commit and configuration SHA-256"; §13.1 launch-record identity between training invocations | `validate_against_preflight`, `check_same_run` | `test_training_validation_refuses…`, `test_entry_rules…` | compliant | Accept. Note: a launch-record edit between invocations 1 and 2 is refused unless the implementation commit also changes. That is the literal §13.1 rule. |
| 4 | `cuda_initialized` in the preflight record reports the process state truthfully; the preflight code itself never initialises CUDA | §5.4.1 "allocates no CUDA context"; §14 preflight record field | `_cuda_initialized`, `run_preflight` | `test_preflight_builds_no_ae_starts_nothing_and_touches_no_cuda` | compliant | Accept. Finding F5 confirms the distinction matters: an inherited import elsewhere in the same process can set it. For the real preflight, launch with `CUDA_VISIBLE_DEVICES=-1` (launch draft). |
| 5 | The provenance set (`config.resolved.yaml`, `config.sha256.json`, `run_metadata.json`) is serialised once per invocation and copied into each control directory, sharing one 150,000-byte cap per control | §13.2 "provenance set … 150,000 per set" | `_production_context` (`run_files`), `run_control`, `ArtifactStore.check` | `test_provenance_set_shares_one_cap` | compliant | Accept. `run_metadata.json` holds the invocation's environment, not a per-control capture time. |

**Choices identified in this review** (also pending human decision):

* N1 (= F1): the §13.1 / §13.3 / §13.4 conflict. Blocks only the re-entry path after an invocation-1 death before its
  first record. Proposed resolutions in section 12.
* N2: training uses only the **most recent** preflight record (§5.4.1 says so); an earlier passing record does not
  rescue a later failing one. Compliant.
* N3: `validate_against_preflight` requires exact equality of the whole evidence dictionary, including the float64
  recomputed maximum, normalised maximum and RMS. With the same code, inputs and pinned environment this is bitwise
  reproducible (the synthetic tests rely on it). A different thread count could change the RMS last bits and refuse
  training. In the tests, a child process had to apply the same `configure_determinism(True, 2)` as the parent; that
  is the general form of this sensitivity. Recommendation: accept the strict comparison, and fix the thread count and
  launch environment in the launch record.

## 12. Validation of 2026-10-05, limitations and the F1 options

**Environment (verified this session).** `D:\conda_envs\cgfedllm\python.exe`, Python 3.11.16, Windows-10-10.0.26100.
The editable install points to this worktree's `src` (`__editable__.cg_fedllm_repro-0.2.0.dev0.pth`). torch
2.14.0+cu130, numpy 2.4.6, safetensors 0.8.0, peft 0.21.1, transformers 5.17.0, ruff 0.16.9, pytest 9.1.1. All tests
ran with `HF_HUB_OFFLINE=1` and `CUDA_VISIBLE_DEVICES=-1`.

| Command | Exit | Result |
|---|---|---|
| `python -m ruff check src tests scripts` | 0 | all checks passed |
| `python -m ruff format --check src tests scripts` | 0 | 118 files already formatted |
| `python -m pytest -q -m "not gpu and not model"` | 0 | 313 passed, 2 deselected; session probe: CUDA never initialised |
| `git diff --check` | 0 | clean |

Count change from 308: +5. Three hard-kill subprocess tests replaced two `SystemExit` tests, which nets +1, and four
tests were added (F1, F2, F6, F9). Each hard-kill test starts one child process.

**Limitations.** CPU and synthetic only. Real input identity, the frozen `m_r` / `s_r`, GPU determinism (F7), timing
and memory are unverified. The production entry points are tested only up to their refusal paths. A hard kill is
exercised at three boundaries, not at every instruction.

**F1, options for a separately reviewed forward revision (the protocol is not changed here):**
1. Amend §13.1: invocation 1 is permitted when the root holds no **finalized** non-preflight file. Empty directories
   and temporaries are allowed; temporaries stay counted and are never deleted.
2. Amend the implementation order so that no directory is created before the capability check, e.g. publish
   `invocation_1` into the run root itself. This changes the §13.2 artifact paths and also needs a protocol revision.
3. Keep v2 as written. An invocation-1 death before its first record then ends the run in a non-enterable state that
   is recorded as a failed launch. A new run would need a new protocol revision and run root.

Option 1 is the smallest change and keeps every no-deletion and accounting rule. Until a decision is recorded, the
code refuses that state. *(Historical text of the first review. Superseded on 2026-10-05: a restricted option 1 was
selected and frozen as v2.1 A2; see section 13.)*

## 13. Reconciliation with the v2.1 addendum and the F7 diagnostic (2026-10-05)

**Decisions (written human instruction of 2026-10-05), frozen in `33e51fd63c6cd0679e76e12380ae5d418158d698`.**
F1: a restricted option 1 (v2.1 A2, A3). Choice 2: control status separated from the stop reason (v2.1 A4). N2, N3
retained. CPU preprocessing environment pinned and compared (v2.1 A5 P1). F5 kept as a historical deviation. One
bounded synthetic GPU diagnostic for F7. The instruction did not approve choices 1, 3, 4 and 5 by number; they are
reproduced in section 13.4 and stay pending.

### 13.1 Source and test mapping

| Provision | Code | Tests (synthetic, CPU) |
|---|---|---|
| A1 protocol identity (pair v2, v2.1) | `p5a.P5A_ADDENDUM_*`, `p5a.protocol_identity()`; `_header`, `run_preflight`, `check_same_run`, `validate_against_preflight` compare the whole pair | `test_p5a::test_protocol_provenance_is_the_full_v2_commit` (both blobs vs. `git hash-object`); `test_p5a_preflight::test_training_validation_refuses…` (a v2-only record is stale) |
| A2 invocation-1 eligibility | `p5a_run.first_slot_entry` (called by `next_invocation` when no finalized `invocation_<k>` exists); records `entry` in `invocation_1.json`; `FIRST_RECORD_CONFLICT` and the literal directory rule removed | `test_hard_kill_during_the_first_record_permits_the_invocation_one_slot` (real `os._exit` child, mode `during_first_record_write`); `test_invocation_one_slot_refuses_started_foreign_or_unknown_states` (start temporary, finalized start/summary, control file, empty control directory, nested directory, unknown files, foreign or later-invocation temporaries, malformed preflight record; each refusal leaves the tree byte-identical); `test_invocation_one_slot_accepts_empty_ledger_and_preflight_temporaries` |
| A2 concurrent claims | unchanged no-clobber `ArtifactStore.publish` (`os.link`); a lost race raises "capability check failed" and publishes nothing further | `test_concurrent_first_record_claims_finalize_exactly_one` |
| A3 retained temporaries | unchanged `footprint()`; a byte refusal of the first record is reported with its cause and publishes nothing | `test_retained_temporary_can_legitimately_refuse_the_first_record` (F = 14,510,000 accepted, 14,510,001 refused as `evidence_reservation`; the temporary is kept and counted) |
| A4.1 stop causes | `p5a_artifacts.refusal_cause` (in every `check` result); `InvocationStop.cause` and `to_dict()`; summaries, failure records, control records and closure records carry `resource_stop_cause` | `test_p5a_artifacts::test_refusal_cause_is_classified_separately_from_any_status` (cap, aggregate and reservation at their exact boundaries, with a retained temporary) |
| A4.2/A4.3 status precedence | `p5a_run.started_status`, `_publication_failed` (no `RESOURCE STOP` for a started control; S1 kept; S3 for every cause) | `test_over_cap_checkpoint_of_a_completed_training_is_publication_not_resource_stop` (computed `FIT FAIL` kept as reported-only evidence), `test_checkpoint_is_not_published_unless_remaining_evidence_is_reserved` (reservation cause), `test_training_stop_status_survives_a_later_byte_refusal`, `test_status_precedence_helpers` |
| A4.2 never-started controls | `close_run` unchanged statuses, plus `run_closed_by` with the cause | the two over-cap/reservation tests above; `test_byte_reservation_refusal_closes_the_run_before_training` |
| A4.4 recovery | `p5a_run.ended_status` in `recover`; failure-record facts cited, not used for the status or charge | `test_recovered_ended_training_stop_keeps_its_status`; existing `test_hard_kill_after_the_end_record…`, `test_an_ended_control_without_record…` |
| A4.5 outcomes | `outcomes()` adds `reported_only_gate_results` beside each outcome; the §11.2 table is unchanged | the over-cap and reservation tests above |
| A5 N2 latest only | `latest_preflight` unchanged | `test_training_uses_only_the_latest_preflight_and_never_falls_back` |
| A5 N3 exact | `validate_against_preflight` unchanged equality | `test_training_validation_refuses…` (one ulp in the RMS refuses) |
| A5 P1 environment | `p5a_preflight.cpu_environment`, `ENV_VARS`; both `production_preflight` and `_production_context` call `configure_determinism(cfg.run.deterministic, cfg.run.num_threads)`; `RUN_LOCK["num_threads"] = None`; training re-reads the fingerprint before each rebuild (`RunContext.cpu_env`) and compares it as part of the preflight identity | `test_cpu_environment_fingerprint_must_match_exactly`; `test_production_preflight_route_on_an_empty_synthetic_root` (full fingerprint recorded) |

Preserved without change: F2 (`close_run` never relabels a started control), F3 (`check_same_run` in the
closure-only step), F4 (hard-kill children; one mode added), F6 (real evaluation-overrun path), F9 (now the
configured instead of the literal `configure_determinism(True)`, with the same values for the locked config).
Existing negative evidence and historical counts are kept as written.

### 13.2 Pinned CPU preprocessing settings

The locked configuration gives `run.deterministic: true`, `run.num_threads: null` (`configs/base/defaults.yaml`, not
overridden). Both stages therefore leave the torch thread count at the machine default. Measured in this
environment with no threading variables set: `torch.get_num_threads() = 24`, `get_num_interop_threads() = 24`,
float32 matmul precision `highest`, default dtype `torch.float32`; after `configure_determinism`: deterministic
algorithms on, warn-only off, cuDNN deterministic on, benchmark off, both TF32 flags off,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`. No thread count is invented: the requirement is that the preflight and every
training invocation reproduce the same fingerprint, which is checked and recorded in full. The tests run with the
suite's own `configure_determinism(True, 2)` (`tests/conftest.py`).

### 13.3 F7: synthetic-only GPU compatibility diagnostic (measured)

Authorised once on 2026-10-05; not a P5-A experiment and not a §15 launch-record smoke test. Helper and log are
outside the repository and outside every run root (`D:\p5a_f7_diag\`). No real index, metadata, snapshot, gradient,
model or dataset was read; no checkpoint, control, attempt, ledger or preflight record was written.

* **Pre-check.** The PyTorch 2.14 documentation of `torch.use_deterministic_algorithms` lists
  "torch.nn.ReflectionPad2d when attempting to differentiate a CUDA tensor" among the operations that throw a
  `RuntimeError` when `mode=True`, `warn_only` defaults to `False`. The AE has 14 `ReflectionPad2d` modules and its
  backward graph contains 13 `ReflectionPad2DBackward0` nodes (CPU graph walk; the first stem pad acts on the
  input, which needs no gradient). Source evidence from the pinned binary: `torch_cuda.dll` contains both
  `reflection_pad2d_backward_out_kernel` and `reflection_pad2d_backward_det_out_kernel` (a separate deterministic
  kernel); only the non-deterministic kernel exists for 1-D and 3-D. This suggests that the pinned build has a
  deterministic 2-D path that the documentation page does not reflect. That is an inference from symbol names, not a
  reading of the source.
* **Command.** `timeout 180 D:\conda_envs\cgfedllm\python.exe D:\p5a_f7_diag\f7_gpu_diag.py`, `HF_HUB_OFFLINE=1`,
  `CUDA_VISIBLE_DEVICES` unset. Helper SHA-256 `1d273aea6996ae26cb60a2255e6a416f7c91900efb0066aba728fb6056b0bcfa`.
  It runs the real `p5a_run.train_and_evaluate` after the production order `load_config`, `check_config`,
  `configure_determinism(cfg.run.deterministic, cfg.run.num_threads)`, `apply_vram_guard`, `build_ae`.
* **Code identity.** Worktree HEAD `33e51fd…` plus the uncommitted implementation; config SHA-256
  `e5ac614c…bb122`; the SHA-256 of every `phase5/*.py` file and of `autoencoder.py`, `codecs.py`,
  `normalization.py`, `seeding.py` and `pipeline.py` is in the log. The `phase5` hashes equal section 13.5.
* **Environment.** Python 3.11.16, torch 2.14.0+cu130 (git `08187d9e…`, CUDA 13.0, cuDNN 92400), RTX 4060 Laptop GPU
  (compute 8.9, 8,585,216,000 bytes), driver 595.97; allocator cap 7,182,745,600 bytes (free 7,451,181,056 minus
  256 MiB).
* **Flags immediately before each of the 4 backward calls** (recorded by a pass-through wrapper of
  `Tensor.backward`): deterministic algorithms `True`, warn-only `False`, cuDNN deterministic `True`, benchmark
  `False`, `cuda.matmul.allow_tf32` `False`, `cudnn.allow_tf32` `False`, float32 matmul precision `highest`,
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
* **Geometry.** Synthetic uniform(-0.9, 0.9) fp32 tensors `[1, 2048, 1536]`, generator seed 20261005; the frozen
  ResNet-3 (493,483 parameters), Adam lr 2e-4, (0.9, 0.999), eps 1e-8, wd 0, normaliser `global_exact_maxabs_train`
  with scale 1.0.
* **Result.** Exit 0 in 6.97 s. Batch 1 (k = 1): 2 optimizer steps, eval-mode evaluation, finite reconstructions,
  peak allocated 203,850,752 bytes. Batch 4 (k = 4): 2 steps, evaluation, finite, peak allocated 789,693,952 bytes,
  reserved 916,455,424. 4 optimizer steps in total. No error, warning or OOM; stderr empty.
* **Meaning.** With the strict settings of v2 §8.3, the pinned build ran forward, backward (including
  `ReflectionPad2d` backward on CUDA) and the optimizer step for both batch paths without a deterministic-kernel
  error. The measured behaviour differs from the documentation page. This establishes compatibility for these
  synthetic paths only. It does not establish real-data fit, full-run time or memory, bitwise repeatability across
  runs, seed stability or generalisation; repeatability was not tested.

### 13.4 The engineering choices, as described in section 5 and their status now

| # | Actual description (section 5) | Status after 2026-10-05 | Recommendation |
|---|---|---|---|
| 1 | Pre-start `PROVENANCE` and `NOT INFORMATIVE` stay `NOT_STARTED` until closure; terminal records are issued at closure from the last recorded pre-start status, so a run with a permanently pre-start-blocked control closes only after invocation 3 | unchanged; compliant with v2 §12, §13.5 and v2.1 A4.2 (never-started rules retained) | accept (pending) |
| 2 | Over-cap → `RESOURCE STOP`, reservation refusal → `PUBLICATION` during a started control's publication | **superseded** by v2.1 A4: a started control is never `RESOURCE STOP` because of bytes; S3 `INCOMPLETE (PUBLICATION)` for every cause, S1 kept; cause recorded as `resource_stop_cause` | decided (frozen) |
| 3 | Preflight/training identity compared on `execution_commit` and `config_sha256`, not on the launch record (updated after the preflight to cite it) | unchanged; the comparison now also includes the protocol pair and `cpu_environment` (v2.1 A1, A5 P1). A launch-record edit between invocations 1 and 2 is still refused unless the implementation commit also changes | accept (pending) |
| 4 | `cuda_initialized` in the preflight record reports the process state truthfully; the preflight never initialises CUDA | unchanged; F5 shows why it matters; the real preflight runs with `CUDA_VISIBLE_DEVICES=-1` | accept (pending) |
| 5 | The provenance set (`config.resolved.yaml`, `config.sha256.json`, `run_metadata.json`) is serialised once per invocation and copied into each control directory under one 150,000-byte cap | unchanged; `run_metadata.json` now also holds `cpu_environment` | accept (pending) |

### 13.5 Validation (2026-10-05, after the reconciliation)

Pinned interpreter, worktree imports (`cg_fedllm.__file__` under this worktree's `src`), `HF_HUB_OFFLINE=1`,
`CUDA_VISIBLE_DEVICES=-1`:

| Command | Exit | Result |
|---|---|---|
| `python -m ruff check src tests scripts` | 0 | all checks passed |
| `python -m ruff format --check src tests scripts` | 0 | 118 files already formatted |
| `python -m pytest -q -m "not gpu and not model" -p cuda_init_probe` | 0 | **324 passed, 2 deselected** in 88.8 s; probe: 324 tests, CUDA never initialised, `false` at the end |
| `git diff --check` | 0 | clean |

The probe is a read-only pytest plugin outside the repository (`D:\p5a_cpu_probe\cuda_init_probe.py`, on
`PYTHONPATH`) that records `torch.cuda.is_initialized()` around every test. Count change from 313: +11 (the F1 test
replaced by five A2/A3 tests: +4; in `test_p5a_run` also four status and recovery tests: +4; in
`test_p5a_preflight` one environment and one N2 test: +2; in `test_p5a_artifacts` one cause test: +1). Five P5-A files
were reformatted by `ruff format` after the edits; no other file was reformatted.

**Refreshed fingerprints** (the pre-audit working copy; section 14 supersedes changed-file values). Unchanged
files keep their earlier values.

| Path | Git blob | SHA-256 |
|---|---|---|
| `configs/phase5/p5a_v2_seed1.yaml` | `f6385eb0c3a5ac710e102cd94a808f06b4096e26` | `dde9b89a479363a1bbe879b967f89d291345cefdf480c04767c04e042f5513d6` |
| `src/cg_fedllm/phase5/p5a.py` | `b6b0477beef4a70d1aa6766624765134a9bfe382` | `bd709b2e80037e03cdd03ce048a5222f18c5fd63043a5ea997ce70ce5fd21265` |
| `src/cg_fedllm/phase5/p5a_artifacts.py` | `1184b0ea617159411e42b7d80436beaa4032160b` | `93634c152c26bcbb77a47ea1cf9e13d8a127b555afc80b387dc31514c022773b` |
| `src/cg_fedllm/phase5/p5a_preflight.py` | `49c265766d534e160a30f1822dca7336c9cfd9ec` | `b8c0140452445b19a0a46fa5f76eff71585b11a624bd88cfa20b141507dcba44` |
| `src/cg_fedllm/phase5/p5a_run.py` | `fd6b66668b508488598855c6e1617ff011e5cd46` | `01465b0bebf55c199a10386da51275c8e70142bc71d8dd345bf2e384dd660534` |

Resolved `config.sha256()` and identity hash: `e5ac614c5f176f39b40ddd96cfe311f50c8209151578a1535794cd55d87bb122`
(unchanged; `num_threads: null` was already the resolved value).

**Limitations.** CPU and synthetic, except the one synthetic GPU diagnostic of section 13.3. Real input identity, the
frozen `m_r` / `s_r`, real-data timing and memory and run-to-run GPU repeatability are unverified. A hard kill is
exercised at four boundaries. The concurrent-claim test injects the rival inside the link step of one process; two
independent processes were not raced.

## 14. External diff audit and proposed forward fixes (2026-10-05)

The submitted transcript contains the complete staged implementation diff (26 files). Its seven tracked-file
patches apply to the GitHub baseline `afc9e33`; all 26 reconstructed final files match the diff's Git blob
identifiers. This audit does not include the complete local v2/v2.1 documents, the untracked launch draft or the
ZIP bundle. No real input, preflight or training was accessed or run.

* **F10, failure reporting.** `finish_invocation` retains `summary_publication_error`, but `cmd_p5a` dropped it
  and `main` returned 0. An isolated execution of the original function bodies reproduced a missing summary with
  CLI `closed: true`, no visible error and exit 0. The proposed fix raises `P5AProtocolError` when a summary or
  closure record cannot be published, preserving the error details and causing a failing CLI exit. It does not
  delete, overwrite or republish any run artifact, or change a control status or outcome table.
* **F11, closure execution environment.** `_production_context(cuda=False)` still imports `pipeline` and calls
  `collect_environment`; with CUDA visible, the latter calls `get_device_properties`, whose PyTorch 2.14 source
  invokes `_lazy_init`. The proposed fix refuses `production_closure` before shared context construction if CUDA
  is available or already initialised. Use a fresh process with `CUDA_VISIBLE_DEVICES=-1` for closure-only; the
  launch draft must record that precondition. No diagnostic using a real GPU was run for this audit.
* **Regression coverage.** Four CLI cases cover both publication-error fields and both command modes; two cases
  verify that closure refuses a visible GPU or an existing CUDA context before context capture or output. These
  six cases are proposals until the pinned Windows suite is run. The earlier 324-pass record remains historical
  evidence for the pre-audit code bytes, not validation of the proposed fixes.

**Post-fix Windows validation:** PASSED (Phase-5 scoped checks). Ruff check and format check: exit 0; pytest: 188 passed in 54.19s; git diff --check: exit 0. Earlier-phase suites were not rerun.

The launch draft remains outside the implementation commit. Its fingerprint table must be refreshed after any
code formatting, its implementation SHA filled after the implementation commit, and its actual execution HEAD
distinguished from that SHA. Preflight and training must use the same execution HEAD; a docs-only commit between
them would make the stored preflight stale. All real-step authorisations remain separate and pending.

Post-audit source fingerprints after formatting:

- src/cg_fedllm/cli.py: Git blob cf2a1511c0159cb28c80ed40f1e24a1a59d09706; SHA-256 20cac35ce5d6e992ebbdd4a9bc1e3c35d0eb57c4ca703b110470b004c7116603.
- src/cg_fedllm/phase5/p5a_run.py: Git blob 412547f0c0dcdbc28c1ee890617fa6eef8b51473; SHA-256 5aa4f3be27e18bb92083287452b2df5ad9f900fba357aabb43e34f7e5689ce53.
- tests/unit/test_p5a_run.py: Git blob a303742a82e65ea1752e5c55e7af07455d8de2e6; SHA-256 900aba6d050cc858d28c21051d669846e07c64eb099383862c9f6e60e73b74e6.
