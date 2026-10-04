# Phase 5 P5-A: audit response and v2 implementation impact

**Status: v2 APPROVED FOR IMPLEMENTATION — NOT AUTHORIZED FOR EXECUTION.** This document accompanies
`docs/phase5_preregistration_v2.md`. It records how the audit was answered, the approval of v2 and the consistency
corrections made before the protocol commit. It approves no real preflight, payload access, GPU use or run.

| Item | Value |
|---|---|
| Baseline | `afc9e333b5b865b32d47f5634aa7db3ddbfa97ef` (`main`, remote and local) |
| v1 preregistration | `52a0dd4e7bb8521752ac80ff8f77072dc0e0761c`, blob `00f2477d164b795c4f35f717a3017888baee9b4e`, APPROVED for implementation only; local branch `worktree-p5a-engineering-support`, not pushed; unchanged |
| v2 approval | written human instruction, 2026-10-04: the six design choices of v2 §21 approved for implementation; one protocol-only commit of v2 and this document authorised |
| Implementation under review | uncommitted in `.claude/worktrees/p5a-engineering-support`; reconciled to v1 at the time of this document; to be reconciled to the v2 commit afterwards (`docs/phase5_implementation_review.md`) |
| Previously reported software evidence | Ruff lint/format clean; `pytest -q -m "not gpu and not model"` 246 passed, 2 deselected (rerun at the start of the approval session in the pinned environment: same result) |

## 0. Approval and consistency corrections (2026-10-04)

**Approved choices** (v2 §21): (1) `global_exact_maxabs_train`; (2) the strict single-snapshot memorisation standard
RSE <= 0.01, cos >= 0.99; (3) R2/R3 primary, R4 a prespecified secondary that is always scheduled; (4) C4 evaluated as
the inequality at a zero subset-mean error; (5) one run-wide 20,000,000-byte ceiling with evidence reservation, the
old 2 MB / 18 MB split removed; (6) a separately invocable CPU-only input preflight before any GPU training. The
approval covers these choices, not every sentence of the draft. The following consistency corrections were made
before the protocol commit. None changes a threshold, a scale, a selection, a population, a budget ceiling or an
outcome rule.

| # | Check | Finding in the draft | Correction |
|---|---|---|---|
| A1 | Byte accounting | The 18,090,000 total was correct for the artifacts listed, but omitted the preflight records that choice 6 creates | Added `preflight/preflight_<k>.json` (cap 300,000, at most 3). Estimated worst case 18,990,000 <= 20,000,000. Ceiling, other caps and reservation rule unchanged. Estimate, caps, reservation and measured footprint are defined as four separate quantities (v2 §13.2) |
| A2 | Reservation scope | The reservation counted only one further invocation | Every still-permitted invocation is reserved; a preflight reserves the whole training worst case; a checkpoint is never published without the remaining evidence reserved (v2 §13.3) |
| A3 | Run-level cap failure | Not stated for invocation records and summaries | Not published; the invocation stops; a first record that cannot be published means nothing was published (v2 §13.3) |
| A4 | Closure | A process death during invocation 3 before its summary left the run neither closed nor re-enterable | A closure-only step (no payload, no AE, no control, no invocation record) publishes recovery and closure records within reserved caps, at most once (v2 §13.5) |
| A5 | Time-ceiling wording | "time-ceiling stop" could be read as including one training's 1,800 s limit | Defined as the start rule refusing a start; a single overrun only ends that control (v2 §13.5), as in v1 §12 |
| B | R4 scheduling | "always runs" could be read as exempting R4 from prerequisites | "Always scheduled": independent of earlier outcomes, subject to the same provenance, prerequisite, resource-stop and no-retry rules (v2 §2, §19 D03, §20, §21) |
| C1 | Provenance | `d3c4eb3` and other evidence commits abbreviated | Full SHAs, paths, blob ids and fields verified from local Git, all ancestors of `52a0dd4` (v2 §3) |
| C2 | Scale arithmetic | R2 `s_r` written as `0.034553585083861103`, while Python's shortest form is `0.0345535850838611` | Same float64 (`0x1.1b101ebca1af3p-5`); stated explicitly (v2 §5.3). All three `m_r` / `s_r` / `float32(s_r)` values re-derived from the committed JSON fields by float64 arithmetic |
| C3 | Scale semantics | Tolerance, range allowance, NaN ordering and no-substitution partly implicit | Inclusive float64 bound on `m_r`; non-finite fails; finiteness checked before the maximum; the 2e-6 is an allowance on normalised values, not on the scale; a mismatch never replaces the frozen value (v2 §5.2, §5.4, §5.5, §5.6) |
| C4 | Preflight form | Two forms left open | Choice 6 written into v2 §5.4.1, with re-validation of preflight evidence by every training invocation, stale-evidence and changed-identity rules, and at most 3 preflights |
| D | Interpretation | Conditioning and single-seed limits stated in places only | v2 §0 and §1 state: old 0.01 / 0.99 was UNSOURCED scaffolding; both changes are new v2 decisions; one seed establishes no stability; exact max does not remove optimisation or conditioning effects, especially for R4; changing scaling and threshold together isolates no cause |
| E | Status labels | DRAFT / PROPOSED labels throughout | Replaced by APPROVED (v2) for CHANGED/NEW decisions; INHERITED kept; "approved for implementation, not execution" stated |

**Evidence terms.** VERIFIED HERE: checked on 2026-10-04 (drafting or approval session) from the repository text, git
objects or committed `results/` JSON. No payload was read in either session. REPORTED EARLIER: stated by an earlier
session and not re-checked. DERIVED: arithmetic on committed records.

## 1. Audit requests and dispositions

| # | Audit request | Evidence before v2 | Disposition |
|---|---|---|---|
| 1 | Prefer exact training-population max-abs normalisation; if p99.9 is kept, state its limits and do not attribute failure to capacity | v1 uses p99.9 -> 0.95 with no clipping. Under v1, normalised maxima would reach 1.61 / 1.66 / 6.57 (v1 §5, DERIVED from F6), so a Tanh decoder cannot represent the largest values | **Changed.** v2 adopts `global_exact_maxabs_train` (P5v2-D08/D09/D11). The interpretation limits of v2 §1 apply in either case |
| 2 | Stricter single-snapshot gate, or qualify RSE <= 0.50 as a minimum standard; one seed does not establish stability | v1: C2 RSE <= 0.50, C3 cos >= 0.90 for both controls; seed {1}; no stability claim was made, but none was ruled out explicitly | **Changed.** `single_snapshot`: RSE <= 0.01, cos >= 0.99 (P5v2-D29). The subset gate is unchanged but named as a weaker standard (P5v2-D30). The single-seed limit is stated (P5v2-D26, v2 §1) |
| 3 | Verify the complete R4 SHA-256 manifest and reproducible normalisation semantics | see §2 | **Already satisfied** for the manifest (declared in v1 and implemented). The normalisation specification is **rewritten** for exact max (v2 §5). Payload verification is **pending** a later authorised preflight |
| 4 | R4 secondary, R2/R3 prioritised; keep the subset control after a single-snapshot failure | v1 §6 already forbids skipping `fixed_subset4` after a `single_snapshot` outcome, and the implementation tests it. v1 gives R4 equal standing | **Changed.** No-pruning is already approved and implemented. R4 is now a prespecified secondary that is always scheduled and subject to the common prerequisite rules (P5v2-D03); R2/R3 are listed first |
| 5 | Make metrics, zero-norm handling, resource limits, gate execution and failure-evidence preservation explicit | v1 §§9-12 define most of these. Gaps: C4 at a zero subset-mean error; JSON of undefined values; evaluation overrun; crash timing fields; summary bounds; a reserve split not enforceable as written | **Changed** where gaps existed (P5v2-D31-D33, D35, D40-D47). Otherwise written out in full |
| 6 | Freeze the revised preregistration in a separate approved commit before reconciling code; no real experiment without review and authorisation | the v1 approval record already requires a launch record and written authorisation | **Adopted** as the v2 approval path (v2 §18). The protocol-only commit contains v2 and this document only; the implementation is reconciled afterwards and is not part of it |

## 2. R4 manifest and provenance (audit item 3)

| Question | Finding | Basis |
|---|---|---|
| Is a manifest absent? | No | — |
| Is it present but incomplete? | No: 14 files, 5 `meta.json` and 9 pre-clip gradients; step counts 2, 3, 1, 1, 2; no clipped or delta file | VERIFIED HERE by parsing the v1 table at `52a0dd4` |
| Is it complete in v1 and implemented? | Yes. `phase5/p5a_inputs.py` `R4_GRADIENT_FILES` equals the v1 table entry by entry (path, bytes, SHA-256) | VERIFIED HERE (tuple equality) |
| Do the step counts match committed evidence? | Yes. `f5_gradient_forensics.json` (`3cc0c10`) `optimizer_steps.per_client_round` begins 2, 3, 1, 1, 2 for the time-index-0 clients 2, 26, 55, 75, 86 of its `schedule` | VERIFIED HERE |
| Are the index hashes frozen? | Yes. Both appear in v1 §3 and in `INDEX_SHA256`. The Phase-3 hash also appears in the committed Phase-3/4 records | VERIFIED HERE |
| Does the reader avoid forbidden files? | The implementation's restricted reader opens only allowlisted files. A synthetic test with clipped, delta and validation-round files present shows that each allowlisted file is opened exactly once and no other file is opened | REPORTED EARLIER (test passed in the implementation session); design re-read here |
| Was payload identity re-verified? | **No.** The 14 hashes were recorded on 2026-10-04 by reading the files (v1 §3). No payload was opened or hashed in the implementation, drafting or approval sessions | — |

The audit's request is therefore satisfied for the declared manifest and the access design. What remains is the
authorised preflight, which re-verifies identity on the real files (v2 §5.4, §16).

## 3. New scientific decisions and rationale

| ID | Decision | Rationale | Cost / limit |
|---|---|---|---|
| P5v2-D08, D09, D11 | Exact maxabs `s_r = m_r / 0.95` over the whole population, a new mode identifier | Every training value lies within Tanh's range, so the range cannot by itself prevent a fit. This answers audit item 1 directly | Smaller bulk scale: normalised RMS 0.117 / 0.130 / 0.015 vs v1's 0.199 / 0.227 / 0.105 (DERIVED). This deepens the small-scale problem of P4-D5 for R4. Optimisation difficulty is not removed |
| P5v2-D29 | `single_snapshot`: RSE <= 0.01, cos >= 0.99 | Memorising one snapshot is the weakest possible demand on an AE, so a 50 % squared-error allowance is not a meaningful memorisation test | A new number (the earlier scaffolding thresholds were UNSOURCED, never approved). Cosine is implied by the RSE bound and is not binding |
| P5v2-D30 | `fixed_subset4` unchanged, named a weaker subset fit standard | It reuses the Phase-4 numbers; changing it as well would add a second untested number | The two controls test different standards and are reported as such |
| P5v2-D03 | R4 prespecified secondary | Only 5 round-0 snapshots, heavy tail (max/rms 62.9), zero A parts in two records | R4 is always scheduled (subject to the common prerequisite rules) and always reported |
| P5v2-D32 | C4 at subset-mean SSE = 0: the inequality itself | v1 failed C4 unconditionally there, which is a shortcut, not the inequality | Cannot arise for the selected records (four distinct snapshots) |
| P5v2-D36 | Bounded meaning of the outcome phrases | The single-snapshot and subset standards differ | Phrases are kept for continuity with v1 |

Changing the normalisation and the single-snapshot threshold together means a later reader cannot attribute any
difference from the v1 design, or from F6, to either change alone, and no outcome isolates a causal explanation of the
F6 failures. v1 was never run, so no such comparison exists. One seed establishes no stability. Exact max-abs
scaling removes the Tanh-range obstruction but not optimisation or conditioning effects, which may be larger,
especially for R4.

## 4. Frozen scales (DERIVED FROM COMMITTED EVIDENCE; NOT RECOMPUTED FROM PAYLOADS)

`input_stats.train_max_abs` in the committed F6 records is the exact fp32 maximum of |x| over the same 80 / 80 / 5
training tensors: F6 computed `float(max(xs[i].abs().max() for i in train_idx))` in `cmd_forensic_screen` at
`e8100e9dd19ed4df98777646d6907d718790014e`, the commit recorded in each record's `provenance.git.commit`. The three
records were committed in `d3c4eb3d631e0280533f35eb652fb24f68870079`, the only commit touching their paths; their
blobs are identical at `52a0dd4`. Each record's `split.train` and `snapshot_index_sha256` match the frozen populations
and indexes. These checks were VERIFIED HERE, re-run at approval from `git show d3c4eb3:<path>` and float64
arithmetic only.

| R | File (blob) | `train_max_abs` | `s_r = m_r / 0.95` |
|---|---|---|---|
| R2 | `results/phase4/screen/f6_r2_balanced_effective_state.json` (`39f9c20`, `d3c4eb3`) | 0.032825905829668045 | 0.034553585083861103 |
| R3 | `results/phase4/screen/f6_r3_balanced_effective_delta_r8.json` (`98e8ce2`, `d3c4eb3`) | 0.014964050613343716 | 0.015751632224572334 |
| R4 | `results/phase4/screen/f6_r4_mean_step_gradient.json` (`1edf73b`, `d3c4eb3`) | 0.0643031895160675 | 0.06768756791165001 |

Each `m_r` is exactly representable in fp32, consistent with an fp32 maximum. The R2 literal `0.034553585083861103`
and Python's shortest form `0.0345535850838611` are the same float64. These values are frozen by the v2 approval.
They are DERIVED from committed evidence and have not been confirmed against the payloads. The later authorised
preflight (v2 §5.4.1) confirms them: a mismatch beyond rel. 1e-6 stops that representation, and the value is never
replaced. The v1 p99.9 scales are not reused.

## 5. Revised engineering rules

| Topic | Implementation at drafting (v1) | v2 rule | Implication |
|---|---|---|---|
| Byte budget | ceiling 20,000,000 on all files, plus a check that the cumulative bytes of all non-`ledger/` files stay <= 18,000,000 (`p5a_artifacts.py` `preflight`). The cap is cumulative across trainings, not per training | one run-wide 20,000,000 ceiling on all files, per-artifact caps (v2 §13.2), and a reservation check `F + Σ caps(Q) <= 20,000,000` before each training and publication; estimated worst case 18,990,000 including preflight records (section 0, A1) | removes the 2 MB reserve split. Required failure, record and summary evidence is always reserved before a checkpoint is allowed |
| Evidence after a ceiling stop | a failure record is published into `ledger/`, which may use the full ceiling | records and summaries were reserved in advance, so they fit by construction | the 2 MB reserve has no remaining role |
| Crash charging | 1,800 s charged for a start record without an end record; not reported as a measured field | separate `gpu_wall_s_measured`, `timing_status`, `gpu_wall_s_charged`, `charge_reason` (v2 §12) | a charge is never reported as runtime |
| Attempt states | `attempt_*` and `end_*` files; a re-entry raises on existing paths | `NOT_STARTED` / `STARTED` / `ENDED` / `RECORDED`; start record before `ae.to(device)`; later invocations recover evidence without retraining | an uncertain start always counts; recovery is defined, not an exception |
| Summaries | one new time-stamped summary per runner entry (unbounded) | at most 3 invocations, `invocation_<k>` and `summary_<k>`; closure rules (v2 §13.5) | bounded artifact count; outcomes only at closure |
| Evaluation overrun | `INCOMPLETE (RESOURCE STOP)` if the measured time >= 1,800 s after evaluation | the same rule, now in the protocol | none (already implemented) |
| Publication failure | the control re-raises a `PublicationError`; finalized files are kept | `INCOMPLETE (PUBLICATION)`; failure record; invocation stops | separates completion from publication |
| Undefined values | `None` in Python, written as `"nan"` / `"inf"` strings by `json_safe` | `null` with a reason map; `allow_nan = false` | record schema change |
| C4 at subset-mean RSE 0 | fails | the inequality | one branch in `criterion_subset_mean` |
| Wording | `check_wording` blocks forbidden words in any record | structured fields; fixed texts; human review of findings | the filter would block correct limitation statements |
| Run root | `phase5_p5a_seed1/` | `phase5_p5a_v2_seed1/` | v1 output can never be produced by the v2 code |
| Input preflight | part of the run | separate CPU-only route, `preflight/preflight_<k>.json`, re-validated by every training invocation (v2 §5.4.1) | new route; no AE or CUDA in the preflight |
| No-clobber primitive | `os.link`; the earlier session reported `FileExistsError` on NTFS | the same, re-checked at every invocation start | REPORTED EARLIER evidence only |

## 6. v1-to-v2 differences

See `docs/phase5_preregistration_v2.md` §20 (the full table) and §19 (the decision register, with INHERITED /
PROPOSED status for every decision).

## 7. Implementation changes planned at drafting (made after the v2 commit)

Paths are those of the uncommitted implementation at drafting time, verified by reading the source. The final
implementation and its mapping are recorded in `docs/phase5_implementation_review.md`.

| Area | File(s) | Change |
|---|---|---|
| Normalisation mode | `compression/normalization.py`, `config.py` (`NormalizationMode`) | Add `global_exact_maxabs_train`: one global scale, the same `_scale_like` / finite-positive check. `global_maxabs_train` is unchanged. No P5-A-only fit function that bypasses frozen values |
| Frozen scale and preflight | `phase5/p5a_inputs.py` | Replace `FROZEN_SCALE` / `FROZEN_QUANTILE` / `quantile_scale` use with frozen `m_r` and `s_r` (v2 §5.3). `frozen_normalizer` compares the recomputed exact max with `1e-6 * m_r`, applies the frozen scale and runs the §5.5 range check. `secondary_statistics` checks `train_rms` only. Optionally a CPU-only preflight entry point that stops before any AE is built |
| Inputs / provenance | `phase5/p5a_inputs.py` | No change to the allowlist, the reader or the R2/R3 path. Keep the R4 start/end states unread. Add the population-count, geometry and dtype checks if they are not already exhaustive |
| C0 | `phase5/p5a_run.py` `control_c0`, `phase5/p5a_metrics.py` | No logic change. The rationale and record field note that under v2, C0 is an integrity check |
| Single-snapshot gate | `phase5/p5a.py` constants; `p5a_metrics.gate` / `ae_decision` | Per-control thresholds: `single_snapshot` 0.01 / 0.99, `fixed_subset4` 0.50 / 0.90 / 0.8. C0 uses the same per-control thresholds |
| C4 edge case | `p5a_metrics.criterion_subset_mean` | Drop the `pooled_mean_rse == 0 -> False` shortcut and evaluate `rse <= 0.8 * mean_rse` as written. Undefined or non-finite still fails |
| Undefined values | `p5a_metrics`, `p5a_run` (`json_safe` use) | `null` plus an `undefined` reason map; no string encodings in metric fields; margins and `near_threshold` |
| Order / R4 designation | `phase5/p5a.py`, `p5a_run.run_p5a` | Order unchanged. Add `designation` (`primary` R2/R3, `secondary` R4) to records and summaries; primary first |
| Statuses and outcomes | `phase5/p5a.py` | Add `INCOMPLETE (PUBLICATION)` and `INCOMPLETE (NOT STARTED)`. Extend `representation_outcome` to 9 statuses. v2 qualifier, standards and `does_not_establish` fields. Remove `check_wording` from record writing; keep `FORBIDDEN_WORDS` only for a test or a reviewer aid |
| Attempt states / timing | `phase5/p5a_run.py` (`Ledger`, `run_control`) | Rename `attempt_` to `start_`. Timing fields measured / status / charged / reason. Recovery of `STARTED` / `ENDED` controls at invocation start. A pre-start failure keeps the control `NOT_STARTED` and goes into the summary |
| Invocations and closure | `phase5/p5a_run.py` (`run_p5a`, `production_main`) | `invocation_<k>` records with capability check first; at most 3; closure rules and closure terminal records; `summary_<k>` replaces the time-stamped summary; outcomes only at closure |
| Byte budget | `phase5/p5a_artifacts.py`, `p5a_run.py` | Remove `reserve` / `training_limit`. Add the artifact-cap table, a per-artifact cap check and the reservation check `F + Σ caps(Q) <= 20,000,000` before each training start and publication. `retained_bytes_before_this_file` |
| Publication failure | `phase5/p5a_run.py` `_finish` | `INCOMPLETE (PUBLICATION)` record plus failure record; invocation stop; finalized files kept (unchanged) |
| Run root / config lock | `phase5/p5a.py` `P5A_RUN_NAME`, `p5a_run.RUN_LOCK`, `AE_LOCK.normalization`, `configs/phase5/p5a_seed1.yaml` | Name `phase5_p5a_v2_seed1`; mode `global_exact_maxabs_train`. Either a new `configs/phase5/p5a_v2_seed1.yaml` (recommended; the v1 config is kept unused) or an edit of the existing one. Production refuses the v1 root name |
| Provenance constants | `phase5/p5a.py` | Point `P5A_PREREGISTRATION_*` at the approved v2 commit, path and blob; record v1 as superseded-for-execution |
| CLI | `cli.py` `cmd_p5a` | The default config path changes if a new config file is used; otherwise unchanged |
| Docs | `README.md`, `docs/deviations.md` (rows 44-47), `docs/architecture.md`, `docs/phase5_implementation_review.md` | Rows 45 and 47 change to exact maxabs and the two standards; add rows for `INCOMPLETE (PUBLICATION)` and the run-wide byte policy; rewrite the review record against v2 |

## 8. Tests to add or revise

| Test file | Change |
|---|---|
| `tests/unit/test_p5a_normalization.py` | Replace the p99.9 quantile tests for P5-A with exact-max tests: last-element and negative maxima, the float64 scale, fp32 application, the `1e-6 * m` boundary on both sides, the range-check boundary `0.95 (1 ± 2e-6)` on both sides, all-zero population, non-finite input, no clipping. Keep one test that `global_maxabs_train` is unchanged; add the frozen-value table test against the committed F6 JSON |
| `tests/unit/test_p5a_metrics.py` | Strict single-snapshot thresholds at equality and adjacent float64 values; C4 at subset-mean SSE = 0 (passes only at AE SSE = 0); `null` / reason JSON; margins and `near_threshold` |
| `tests/unit/test_p5a.py` | 81-combination outcome table with the two new statuses; designation fields; qualifier text; no word filter on records |
| `tests/unit/test_p5a_run.py` | Per-control thresholds through the runner; timing fields after a death; start/end/record recovery; invocation count and closure; closure terminal records; outcomes only at closure; the v1 root refused; the reservation check before a start; `INCOMPLETE (PUBLICATION)` after a record-write failure with the checkpoint kept; config lock for the v2 mode and name |
| `tests/unit/test_p5a_artifacts.py` | Remove the reserve-split tests; add per-artifact caps, the reservation check, temporaries counted, and the estimated worst case 18,990,000 |
| `tests/unit/test_p5a_inputs.py` | Unchanged; the recomputed exact max replaces the p99.9 secondary statistic |
| `tests/unit/test_config.py` | The new normalisation literal is accepted; old labels and modes are still accepted |

Afterwards, run the full pinned gates: Ruff check and format, `pytest -q -m "not gpu and not model"`, and
`git diff --check`.

## 9. Human approval items (resolved)

All six items of v2 §21 were approved on 2026-10-04 (section 0): exact maxabs normalisation; the strict
single-snapshot memorisation standard; R4 as a prespecified secondary that is always scheduled; C4 as the inequality
at a zero subset-mean error; one run-wide byte ceiling with evidence reservation; a separately invocable CPU-only
preflight before any GPU training. No open design choice remains.

## 10. Sequence

1. Done: human approval of the six choices.
2. Protocol-only commit of v2 and this document. Its full SHA is recorded outside the documents.
3. Reconcile the implementation with that SHA and run the full pinned software gates.
4. Human review of the implementation diff, and an authorised implementation commit.
5. Launch record (v2 §16) with the actual protocol and implementation SHAs.
6. Separate written authorisation for the real CPU preflight; then, citing its record, for the training run.
7. Execute only the frozen v2 plan, then commit the evidence in a separate reviewed change.

## 11. Sessions

* **Drafting session.** Read-only inspection of git state, v1, the implementation, tests, configs and committed
  `results/` JSON. No snapshot, start-state or gradient payload was opened or hashed. No external `index.jsonl` was
  read. Small read-only Python checks on repository text and committed JSON covered R4 table equality, step counts,
  the selection re-derived from committed metadata, the scale arithmetic, outcome-table coverage and the byte
  arithmetic. One check used random synthetic CPU tensors (Tanh-range ceiling RSE about 5e-16). No AE was trained.
* **Approval session (2026-10-04).** The six choices were written into v2 and the corrections of section 0 were
  made. Full SHAs, blob ids, ancestry, F6 fields and the scale arithmetic were re-verified from local Git objects.
  One in-memory serialisation of a freshly initialised, untrained AE measured the checkpoint size (1,996,628 bytes,
  synthetic). No payload was read, no preflight was run, no AE was trained, and no GPU was used.
