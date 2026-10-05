# Phase 5 preregistration v2.1: P5-A addendum on invocation-1 recovery and publication states

**Status: APPROVED FOR IMPLEMENTATION — NOT AUTHORIZED FOR EXECUTION**

This is a narrow normative addendum to the frozen P5-A v2 protocol. It replaces only the provisions named in
section A1. Every other v2 rule is unchanged. It authorises reconciling the implementation with v2 plus this
addendum and synthetic software validation. It does **not** authorise the real CPU input preflight, any read of a
real payload, a real GPU training or a P5-A run; each still needs a reviewed launch record and a separate written
human authorisation (v2 sections 16 and 17).

| Field | Value |
|---|---|
| Base protocol (frozen, unchanged) | `docs/phase5_preregistration_v2.md`, commit `dd4bf3f4348fd3b2b1b2c4ad23e7204fbef8369c`, git blob `e80d21aeda653ab6d1339d34f7b88118588855df` |
| Companion (unchanged) | `docs/phase5_v2_audit_response.md`, blob `7164034fe7a652f204e7120d59f638f2bb831ffd` |
| v1 (historical, never executed) | `52a0dd4e7bb8521752ac80ff8f77072dc0e0761c`, blob `00f2477d164b795c4f35f717a3017888baee9b4e` |
| Approval | written human instruction of 2026-10-05 selecting the F1 and choice-2 resolutions below and the review decisions of section A5. No reviewer identity is recorded here |
| This addendum's commit | not recorded here (a commit cannot contain its own SHA). The implementation, the launch record and every P5-A record cite it together with the base commit |
| Origin | implementation review of 2026-10-05 (`docs/phase5_implementation_review.md`, findings F1, F5, F7 and engineering choice 2). No P5-A output exists; nothing here responds to a result |

## A1. Precedence

The provisions of sections A2 and A4 **supersede** the conflicting v2 text listed below, and only that text:

| v2 text superseded | Replaced by |
|---|---|
| §13.1, bullet "Training invocation 1 requires a root that holds at least one preflight record and no `ledger/` or control directory" | A2 |
| §13.3, the sentence "For a control artifact that is an engineering failure: the control is `INCOMPLETE (RESOURCE STOP)` …" | A4.3 |
| §13.4, "Partial publication", first sub-bullet "The control is `INCOMPLETE (PUBLICATION)`" | A4.3 |
| §12, "Recovery of evidence", the sentence "If it finds `ENDED` without `RECORDED`, it publishes the control record with `INCOMPLETE (PUBLICATION)` …" | A4.4 |
| §11.1, the "Meaning" cells of `INCOMPLETE (RESOURCE STOP)` and `INCOMPLETE (PUBLICATION)` | A4.2 |

**Protocol identity.** From this addendum on, "the protocol commit" of v2 §5.4.1, §13.1, §14 and §16 means the pair
(base v2 commit, this addendum's commit). Every P5-A record cites both, and preflight evidence or an earlier
invocation that names a different pair is stale or foreign under the unchanged v2 rules.

Section A3 applies the unchanged v2 accounting rules to retained temporaries. Section A5 records review decisions; its
item P1 adds a recorded environment fingerprint and makes a fingerprint mismatch a case of the stale preflight
evidence that v2 §5.4.1 already refuses. Neither changes a v2 value. Where this addendum is silent, v2 governs. No v2
rule that is not listed above is relaxed.

## A2. Training invocation 1: eligibility (replaces the v2 §13.1 invocation-1 bullet; resolves F1)

**Why.** Publishing `ledger/invocation_1.json` creates `ledger/` before the no-clobber link, and a process death
during that publication leaves the directory and possibly a temporary file, which v2 §13.4 counts and forbids
deleting. v2 §13.3 says such an invocation "has published nothing and does not count", while the v2 §13.1 bullet
refused any root with a `ledger/` directory. The root could then never be entered without a deletion. The mere
existence of `ledger/` is not evidence that invocation 1 published its first record.

**Rule.** A training invocation that finds no finalized `ledger/invocation_<k>.json` takes the invocation-1 slot only
if every condition below holds. They are checked before anything is published; a refusal publishes, renames and
deletes nothing.

1. **Preflight evidence.** At least one finalized preflight record exists, the records are numbered consecutively
   from `preflight_1`, and each is strict JSON of the P5-A v2 preflight schema, with a record number equal to its file
   name and `trainings_started: 0`. A record that fails this is foreign or ambiguous provenance, and entry is
   refused. Whether the most recent record may be *used* (protocol identity, implementation commit, configuration,
   environment, pass, exact evidence) is decided afterwards, per representation, by the v2 §5.4.1 rules.
2. **No finalized training-stage file.** Apart from the preflight records, no finalized file exists under the run
   root: no invocation record, summary, start, end or failure record, provenance file, control record or checkpoint.
3. **No evidence that training started or may have started.** No control directory exists (empty or not), and no
   start, end, failure, record, checkpoint or provenance temporary exists. A surviving start-record temporary is an
   uncertain start and is never treated as a fresh run.
4. **Only recognized residue.** Every other path under the run root is one of:
   * the directory `preflight/` and the directory `ledger/` (each may be empty), with no subdirectory inside either;
   * a temporary of an interrupted invocation-1 publication: `ledger/.invocation_1.json.<token>.p5a-tmp`;
   * a temporary of an interrupted preflight publication: `preflight/.preflight_<k>.json.<token>.p5a-tmp`,
     k in {1, 2, 3} (the replaced v2 bullet did not prohibit these).

   `<token>` is the 12-hex-digit per-invocation token of the publishing process. Any other file or directory, at any
   depth, is an unknown state, and entry is refused.

Temporaries are recognized by name only. They are never opened, parsed, renamed, presented as artifacts or used as
evidence of anything.

**Preflight reuse.** Existing preflight records are kept. A training invocation that takes the invocation-1 slot
applies every v2 §5.4.1 rule to the most recent finalized preflight record before it is used; missing, failed, stale
or disagreeing evidence makes the affected representation `INCOMPLETE (PROVENANCE)` exactly as for any other
invocation. This addendum does not change the preflight entry rules of v2 §13.1 or the preflight count, and it
creates no new preflight allowance.

**Publication.** The first record is published with the unchanged v2 §13.4 primitive: an exclusive temporary
followed by `os.link` to `ledger/invocation_1.json`, which fails if that name exists. Two claimants that both pass
the eligibility check race only at that link. Exactly one can finalize the record. Every other claimant gets the
no-clobber failure, removes only its own temporary, publishes nothing further and stops as a capability failure
(v2 §13.5). The finalized record of the winner is never touched. This primitive arbitrates only the first-record
claim. v2's handling of later invocations is not changed: a process that starts after `ledger/invocation_1.json` is
finalized is a candidate for invocation 2 under the unchanged rules, so one training process at a time remains an
operational requirement of the launch record.

**Recorded evidence.** The invocation-1 record lists the entry state (`fresh` or `abandoned_first_publication`), every
retained temporary with its exact size, and whether `ledger/` existed before the invocation.

**Interactions.**
* *§13.1.* Invocations 2 and 3, the closure-only step and the three-invocation limit are unchanged. A root whose
  `ledger/invocation_1.json` is finalized is never eligible for the invocation-1 slot; its next invocation is 2 under
  the unchanged rules, even if a temporary of that record survives as a second hard link.
* *§13.3.* An invocation that could not publish its first record has published nothing and does not count; this
  sentence is unchanged and is what makes re-entry into the slot consistent with the three-invocation limit.
* *§13.4.* Retained temporaries stay where they are, keep their bytes and are counted (section A3). Own-temporary
  cleanup is unchanged: an invocation removes only temporaries carrying its own token.
* *No-retry state machine (§12).* A control can be `STARTED` only after a finalized start record, and a start record
  can be published only after `ledger/invocation_1.json` is finalized. Under conditions 2-4 every control is therefore
  `NOT_STARTED`, `trainings_started = 0` and the charged GPU time is 0. Re-entry into the slot is not a retry of any
  training: nothing started, no counter is reset, no training, time or byte allowance is added, and a started or
  possibly started control is never trained again.

## A3. Byte accounting of retained temporaries (application of unchanged v2 §13.2-13.4)

* The ceiling is 20,000,000 decimal bytes over the exact sizes of **all regular files** under the run root, temporaries
  included, in every subdirectory (v2 §13.3, unchanged). A retained temporary is a regular file. It is counted in the
  measured footprint `F` at its exact size, in every later reservation check and in every reported
  `retained_bytes_before_this_file`, for the life of the run. A temporary name that survives as a second hard link of
  a finalized file is counted again (v2 §13.2).
* A temporary is not an artifact of the §13.2 table. It has no cap, is never in `Q`, is never reserved and can never be
  finalized or presented.
* An invocation-1 temporary is written only after the cap and reservation check of that record passed, so it holds at
  most the serialised record, which is at most 50,000 bytes. Each abandoned attempt can leave one. Abandoned attempts
  publish nothing and do not count (v2 §13.3); their residue accumulates in F and is bounded by the unchanged byte
  rule below.
* **Estimate vs. counted bytes.** The 18,990,000 bytes of v2 §13.2 are a planning bound: every cap at its maximum
  count, with no temporary. It is never reported as usage and is not enforced. The bytes counted after a crash are
  the measured footprint: the finalized files plus every retained temporary. The 1,010,000-byte difference to the
  ceiling is not a reserve; only the reservation check is enforced.
* **Before the first record** of the invocation-1 slot, the unchanged check is
  `F + 50,000 + Σ caps(Q) <= 20,000,000`, where Q holds summaries 1-3 (600,000), invocation records 2-3 (100,000) and
  the evidence of all six controls (6 x 790,000), so Σ caps(Q) = 5,440,000 and the record is publishable iff
  `F <= 14,510,000`. Every later publication and training start applies the unchanged check with F including the
  retained temporaries. A retained temporary may therefore legitimately refuse the first record (the invocation
  stops, having published nothing) or a later start (a byte-ceiling stop under v2 §13.3 and §13.5). No temporary is
  deleted to avoid this, and no cap or ceiling is raised.

## A4. Control status vs. stop reason (resolves engineering choice 2)

**Why.** v2 §13.3 gave `INCOMPLETE (RESOURCE STOP)` to a started control whose artifact exceeded its own cap, while
§13.4 gave `INCOMPLETE (PUBLICATION)` to a started control whose later file could not be finalized. A refused
checkpoint after the end record fell under both, and the implementation picked by cause, so two byte refusals of the
same event gave different control statuses. This addendum separates the control's **completion status** from the
**reason the invocation or run stops**.

### A4.1 Literals and stop causes

The nine status literals of v2 §11.1 are unchanged; the publication status is exactly `INCOMPLETE (PUBLICATION)`.
Every byte-policy refusal records one cause in a separate field, never as a status:

| `resource_stop_cause` | Condition (the v2 §13.3 check is unchanged; the classification is reporting only) |
|---|---|
| `per_file_cap` | the serialised artifact is larger than its own §13.2 cap (for a provenance file: than what is left of its set's cap) |
| `aggregate_byte_cap` | within its cap, but the file itself would take the measured footprint over the ceiling: `F + bytes(this) > 20,000,000` |
| `evidence_reservation` | `F + bytes(this) <= 20,000,000`, but `F + cap(this) + Σ caps(Q) > 20,000,000` |

A refusal before a training start uses the same causes. A byte-policy refusal of a training start or of a control
artifact is a byte-ceiling stop as in v2 (§13.3-13.5): the invocation stops and the run is closed. A refused run-level
artifact (invocation record, summary) keeps the unchanged v2 §13.3 handling: it is not published, the invocation stops
and the cause is reported. A filesystem failure (error, collision, missing no-clobber primitive) is a publication
failure without a `resource_stop_cause`: the invocation stops and the run stays open, as in v2.

### A4.2 Status precedence

For a control whose start record is finalized (`STARTED` or later), the status is the first row that applies:

| # | Condition | Status |
|---|---|---|
| S1 | the training ended without a completed evaluation, with a v2 §10 training-stop status: non-finite loss; the per-training 1,800 s limit or the evaluation overrun of v2 §12; crash, OOM, backend refusal, operator interrupt, or process death after the start record (including the death of a hung or timed-out training, recovered under v2 §12) | that status, unchanged: `INCOMPLETE (NON-FINITE LOSS)`, `INCOMPLETE (RESOURCE STOP)` or `INCOMPLETE (INTERRUPTED)` |
| S2 | training and evaluation completed, and the end record, the checkpoint, the provenance set and the control record are all finalized | `FIT PASS` or `FIT FAIL` (v2 §10) |
| S3 | training and evaluation completed, and any of those four cannot be finalized, for any cause: filesystem error, collision, process death before the record, `per_file_cap`, `aggregate_byte_cap` or `evidence_reservation` | `INCOMPLETE (PUBLICATION)` |

Consequences:
* A status established by S1 is never relabelled by a later publication failure, recovery or closure. If one of that
  control's own files (end record, provenance file or record) is then refused or fails, the failure and its cause are
  recorded in the failure record and, where it can still be published, in the control record; the S1 status stays.
* For a started control, `INCOMPLETE (RESOURCE STOP)` arises only from S1 (a time limit). A byte-policy refusal never
  makes a started control `INCOMPLETE (RESOURCE STOP)`. It stops the invocation and closes the run, which affects
  only controls that have not started.
* Finalized files are never overwritten to simplify reporting. The observed metrics and the computed gate result of
  an S3 control are kept as `reported_only_gate_result` evidence; they are not a gate result (v2 §11.1).
* "Completed" in S2/S3 means observed by the publishing process or stated by the finalized end record. If neither the
  end record nor the control record of a completed training can be finalized, the control stays `STARTED` and the
  unchanged, conservative v2 §12 recovery applies to it (section A4.4). The failure record and the invocation summary,
  where they can be published, report the in-process status and the failure, so the evidence is not lost.

Replacement §11.1 "Meaning" cells (the other columns are unchanged):

| Status | Meaning |
|---|---|
| `INCOMPLETE (RESOURCE STOP)` | a time or byte rule prevented the start of a control that never started, or closed the run before it started; or a started training reached its 1,800 s limit, including the evaluation overrun. A byte-policy refusal during the publication of a started control never gives this status |
| `INCOMPLETE (PUBLICATION)` | the training and evaluation completed, but its checkpoint, provenance set or record could not be finalized: filesystem error, collision, process death between the end record and the control record, or a byte-policy refusal (the `resource_stop_cause` is recorded separately). Computed metrics are kept as reported-only evidence when a record can still be written, but are not a gate result |

For a control that **never started**, the frozen v2 rules are unchanged (§7, §12, §13.5): a pre-start
`INCOMPLETE (PROVENANCE)` or `NOT INFORMATIVE (TANH RANGE)` keeps it `NOT_STARTED` until closure; at closure it
receives `INCOMPLETE (RESOURCE STOP)` if a resource ceiling closed the run, otherwise its last recorded pre-start
status, otherwise `INCOMPLETE (NOT STARTED)`. When a byte-policy refusal closed the run, those closure records also
cite that refusal's `resource_stop_cause`.

### A4.3 Publication of a started control (replaces v2 §13.3 control sentence and §13.4 first sub-bullet)

An artifact of a started control that the byte policy refuses is never published. The control receives its status
from A4.2 (S1, otherwise S3). The failure record (within its own cap, if it can still be published) lists the
finalized and the failed files with their exact byte counts and the `resource_stop_cause`. A byte-policy refusal is a
byte-ceiling stop: the invocation stops and the run is closed (v2 §13.5). Every other v2 §13.4 rule is unchanged.

### A4.4 Recovery and closure (replaces the v2 §12 `ENDED`-without-`RECORDED` sentence)

A later or closing invocation that finds `ENDED` without `RECORDED` publishes the control record with the status
implied by the finalized end record: `INCOMPLETE (PUBLICATION)` if the end record states a completed training and
evaluation (S3), otherwise the S1 status it records; an end record published by a v2 §12 recovery
(`training_end: unknown_process_death`) gives `INCOMPLETE (INTERRUPTED)`. The record cites any finalized checkpoint
and copies the end record's timing. `STARTED` without `ENDED` is recovered exactly as in v2 §12
(`INCOMPLETE (INTERRUPTED)`, `unknown_process_death`, 1,800 s charged unless the recovering process itself measured
that training); a finalized failure record of that control is cited as evidence and changes neither the status nor
the charge. Closure never relabels a started control; the v2 §13.5 closure statuses apply only to controls that never
started.

### A4.5 Outcomes

The v2 §11.2 table, the §11.3 meanings, the qualifier and the §11.5 authorisations are unchanged.
`INCOMPLETE (PUBLICATION)` is never a gate result, so a reported-only `FIT PASS` or `FIT FAIL` does not enter the
outcome table. The closing summary lists every reported-only gate result next to its representation's outcome,
labelled as not a gate result, so that no observed failure evidence is hidden.

## A5. Other review decisions recorded (no v2 value changes)

| Item | Decision | Basis |
|---|---|---|
| N2 | Training uses only the most recent preflight record and fails closed (`INCOMPLETE (PROVENANCE)`) if that record is failed, stale, missing or mismatched. It never falls back to an older passing record | v2 §5.4.1 "Training uses only the most recent preflight record" |
| N3 | Preflight evidence must agree **exactly**. No `allclose` and no new tolerance | v2 §5.4.1 "the two agree exactly" |
| P1 | **CPU preprocessing environment.** Both the preflight and every training invocation apply `configure_determinism(run.deterministic, run.num_threads)` from the locked configuration before building inputs (`num_threads: null` means the machine's torch default; this addendum sets no thread count). The preflight records, and each training invocation recomputes immediately before it rebuilds the inputs, one fingerprint: interpreter path; Python, torch (version, git version, CUDA build), numpy and safetensors versions; the deterministic-algorithm and warn-only flags, cuDNN deterministic and benchmark flags, both TF32 flags, float32 matmul precision and the default dtype; the torch intra-op and inter-op thread counts; and the values (or absence) of `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `MKL_DYNAMIC`, `OMP_DYNAMIC`, `MKL_CBWR`, `KMP_AFFINITY`, `KMP_BLOCKTIME` and `CUBLAS_WORKSPACE_CONFIG`. The fingerprints must be equal. A difference makes the preflight evidence stale, giving `INCOMPLETE (PROVENANCE)` before any start. The full values are written into the preflight and invocation records and cited by the launch record. `CUDA_VISIBLE_DEVICES` is not part of the fingerprint: the preflight hides CUDA and training needs it. Rationale: thread counts and backend flags can change the last bits of float64 reductions such as the RMS, so an unpinned environment could refuse training spuriously under N3; recording it makes any refusal diagnosable instead of silent | v2 §5.4.1 (stale evidence; exact agreement); review N3 |
| F5 | Historical execution-scope deviation, preserved. Every full CPU suite run before 2026-10-05, including the 308-pass runs of 2026-10-04, created a CUDA context through an inherited `peft` import. Those runs did no GPU computation and accessed no data. They are not restated as CUDA-free. CPU validation runs with `CUDA_VISIBLE_DEVICES=-1` | implementation review F5 |
| F7 | Open when this addendum was frozen. v2 §8.3 is unchanged: determinism is never disabled, and a backend refusal of a deterministic kernel makes the training `INCOMPLETE (INTERRUPTED)`. The PyTorch 2.14 documentation of `torch.use_deterministic_algorithms` lists `torch.nn.ReflectionPad2d` "when attempting to differentiate a CUDA tensor" among the operations that raise a `RuntimeError`. A separately authorised, synthetic-only GPU compatibility diagnostic (no real input, no P5-A record) checks the pinned build. If it fails, real GPU training is blocked. Any change of architecture, padding, determinism setting, software version or device would be a scientific revision that needs separate human review | implementation review F7; v2 §8.3, §15 |

## A6. Unchanged; approval

Unchanged: the scope and designations; the populations and selections; the frozen inputs, `m_r` and `s_r`; the
normalisation and its checks; the thresholds and gates; the seed; the architecture, optimiser, iterations and
checkpoint rule; the 6-training, 1,800 s and 10,800 s ceilings; the 20,000,000-byte ceiling, the §13.2 caps and the
reservation rule; the preflight and invocation limits; the no-retry rule; the record contents, apart from the fields
added by sections A2, A4 and A5; the launch-record and authorisation requirements. The preflight entry rules of v2
§13.1 are not changed by this addendum.

Approval: the written human instruction of 2026-10-05 approves sections A2 and A4 and the decisions of section A5 for
implementation. The real preflight, real training and every later P5-A step remain **not authorised**. Each requires a
launch record that cites the v2 commit, this addendum's commit and the reviewed implementation commit, and a separate
written authorisation.
