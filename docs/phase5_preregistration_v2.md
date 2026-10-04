# Phase 5 preregistration v2: P5-A AutoEncoder fit control for R2-R4

**Status: APPROVED FOR IMPLEMENTATION — NOT AUTHORIZED FOR EXECUTION**

This document is the revised, self-contained P5-A protocol. It is the active successor of v1 for every future P5-A
step (section 0). Its approval authorises only (a) reconciling the P5-A implementation with this document and (b)
synthetic software validation of that implementation (section 15). It does **not** authorise the real CPU input
preflight, any read of a real snapshot or gradient payload, any GPU use, or a P5-A run. Each of those needs a
reviewed launch record and a separate written human authorisation (sections 16 and 17).

The approved v1 preregistration is preserved unchanged as the historical approved record. It is never executed.

| Field | Value |
|---|---|
| Repository | https://github.com/zindigo321/cg-fedllm-reproduction |
| Baseline | `afc9e333b5b865b32d47f5634aa7db3ddbfa97ef` (`main`) |
| v1 preregistration | `docs/phase5_preregistration.md`, commit `52a0dd4e7bb8521752ac80ff8f77072dc0e0761c` (parent `afc9e33`; local branch `worktree-p5a-engineering-support`, not pushed), git blob `00f2477d164b795c4f35f717a3017888baee9b4e`; APPROVED 2026-10-04 for implementation only |
| Approval of v2 | written human instruction of 2026-10-04 approving the six design choices of section 21 for implementation (section 18). No reviewer identity or review event is recorded here |
| This document's commit | not recorded here (a commit cannot contain its own SHA); the implementation, the launch record and every P5-A record cite it |
| Scope | P5-A only (section 2) |
| Result labels | `PHASE5-DIAGNOSTIC`; `DERIVED` for arithmetic on recorded results |
| Companion | `docs/phase5_v2_audit_response.md`: audit dispositions and implementation impact |

**Status terms.** INHERITED: unchanged from v1 (`52a0dd4`; v1 section given). CHANGED: replaces a v1 value or rule.
NEW: has no v1 counterpart. Every CHANGED and NEW item is **APPROVED (v2)**: it is a new v2 rule, approved on
2026-10-04 for implementation, and it is not a v1 rule. DERIVED: arithmetic on committed records, done before any
P5-A result exists. Section 19 is the decision register.

## 0. Why v2 exists

* v1 was approved on 2026-10-04 for implementation only. A later human audit withheld approval to execute it and
  asked for revisions (`docs/phase5_v2_audit_response.md`, section 1).
* No P5-A training exists. No AE forward pass has been run on a real input, and no P5-A gate value or output exists.
  v2 was therefore written and approved before any result, and none of its changes responds to an observed outcome.
* v2 supersedes v1 **for execution**: it is the only P5-A protocol that may ever be executed. v1 is never executed,
  and its run root `<CGFED_RUNS>/phase5_p5a_seed1/` is never created (P5v2-D01). v1 remains the historical approved
  record, and the earlier Phase-4 negative results (F6) stand unchanged.
* **Historical correction.** The single-snapshot thresholds RSE <= 0.01 and cosine >= 0.99 appeared earlier only in
  unapproved scaffolding, classified UNSOURCED. They were never part of an approved protocol. Their use in v2
  (P5v2-D29) is a new v2 decision motivated by the current audit, not a restoration of an earlier gate.
* v2 changes the normalisation (exact max-abs) and the single-snapshot gate (0.01 / 0.99) together. Both are newly
  approved v2 changes. v1 was never run, so no outcome comparison exists, and any later comparison with v1's design
  or with F6 cannot attribute a difference to either change alone. No v2 outcome isolates a causal explanation of
  the F6 failures.
* Phase 5 is not complete. v2 approval completes no experiment.

## 1. Research question and interpretation limits (P5v2-D02, CHANGED wording)

**Question.** For each of R2, R3 and R4 separately: can the fixed ResNet-3 AutoEncoder with the exact 1/64
bottleneck meet the fit standards of section 10 on (a) one fixed training snapshot and (b) four fixed training
snapshots? Training uses the unchanged Phase-3/4 optimiser for exactly 3,000 iterations. Before training, every
input of the representation's training population is divided by one frozen scalar that places all of those inputs
inside the decoder's Tanh range.

**What P5-A is.** A bounded diagnostic of the fixed instrument on the snapshots it is trained on. Removing the Tanh
range obstruction does not isolate "capacity" from optimisation, architecture, conditioning, precision or BatchNorm
effects. All of these are held fixed; none is separated. In particular, exact max-abs scaling makes the bulk of
each input smaller (section 5.3), most strongly for R4, so optimisation and conditioning effects remain and may be
larger than under v1's scaling. A single AE seed cannot establish stability of any outcome.

**What an outcome can establish** (section 11):
* a control that meets its gate shows that, under this protocol, the fixed AE reached that control's fit standard on
  those snapshots;
* a control that does not meet its gate shows only that this AE, scaling, budget, seed and checkpoint did not reach
  that standard.

**What no outcome establishes:** compressibility or incompressibility of a representation at 1/64; sufficient or
insufficient AE capacity in general; generalisation to unseen snapshots; stability across seeds; the cause of the F6
failures; anything about FAF, operational server semantics or the paper's compression claims.

**Smallest informative design.** One seed, two controls, three representations, no validation data and no
hyperparameter variation: at most six trainings.

## 2. Scope and exclusions

| Item | Rule | Status |
|---|---|---|
| Representations | R2 `balanced_effective_state`, R3 `balanced_effective_delta_r8`, R4 `mean_step_gradient`, constructed as in Phase-4 F6 | INHERITED (v1 §2) |
| Designation | R2 and R3 are **primary**. R4 is a **prespecified secondary** diagnostic. It is always **scheduled** in the fixed plan (section 7), independently of any earlier outcome; like every control, it remains subject to the frozen provenance, prerequisite, resource-stop and no-retry rules. Its designation does not depend on any result | NEW (P5v2-D03) |
| Excluded | R0 `adapter_state`, R1 `adapter_delta`, any new representation | INHERITED |
| Data | frozen artifacts of section 3 only; 0 new client rounds; no TGAP collection | INHERITED |
| Held-out data | no validation-split payload (R2/R3 time indices 16-19; R4 time index 1) is opened, hashed or read; no D2, MMLU, C-Eval or held-out loss | INHERITED |
| Excluded work | FAF; P5-B/C/D; any re-screen; downstream evaluation; operational server semantics | INHERITED |
| Compute | local RTX 4060 Laptop GPU only | INHERITED |

**Why R4 is secondary** (P5v2-D03). R4 has only five training snapshots, all from round 0. Two of them, (0, 55) and
(0, 75), took a single optimizer step at B = 0, so their A part is exactly zero. Its value distribution is extremely
heavy-tailed (committed F6 `train_max_abs / train_rms` = 62.9, against 8.1 for R2 and 7.3 for R3). These properties
were known before P5-A (`phase4_findings.md`, `203be49`). They make R4 less representative of the federated
regime, but they are not grounds to drop it. R4 is scheduled in the fixed plan whatever R2 and R3 yield, and its
outcome (including an `INCOMPLETE` or `INCONCLUSIVE` one) is always reported. "Always scheduled" does not exempt R4
from the section-7 prerequisite rules: a provenance failure, a resource stop or a run closure affects R4 exactly as
it would affect any other control.

## 3. Frozen inputs and provenance

No artifact is regenerated or modified. Payloads stay outside Git under `CGFED_RUNS` (`D:\cgfed-runs` on the
reference machine). Paths are relative to the named root.

**Roots and indexes** (INHERITED, v1 §3; P5v2-D05).

| Root (under `CGFED_RUNS`) | `index.jsonl` SHA-256 | Records | Used for |
|---|---|---|---|
| `phase3_tierb_qwen15_1p8b_seed1/tgap_federated_state` | `b6834e476f21f3d6b07f6489b5113576e566ab3987a2f4593405664476d07240` | 100; `federated_pretrain`; schema `cg_fedllm.tgap_snapshot/v1` | R2, R3 |
| `phase4_gradient_forensics_seed1/gradient_forensics` | `560cb0f8153fc0a44e2910ae119ebdccdf228271f963969ff4352401833af9b2` | 10; `federated_pretrain`; same schema | R4 |

The Phase-3 index hash also appears in the committed `results/phase3/tgap/federated_state_seed1_stats.json` and
`results/phase4/screen/f6_r{2,3}_*.json`. The F5 index hash appears in `results/phase4/screen/f6_r4_*.json`.

**R2/R3 payloads** (INHERITED). Each of the 80 training records names an end-state `file` and a start-state
`start_file`, with file SHA-256 (`file_sha256`, `start_file_sha256`) and tensor-content hashes (`end_adapter_hash`,
`start_adapter_hash`). These hashes are frozen by the hashed index. Payloads are read only through
`load_states(root, record, verify=True)` (`tgap/snapshots.py`), which checks both kinds of hash. `verify=False` is
forbidden.

**R4 payloads** (INHERITED, v1 §3 table; P5v2-D06). P5-A may open, under `<F5 root>/gradients/`, exactly the 14 files
below and no other file of that directory. They are `meta.json` plus each pre-clip gradient
`step{k:03d}_grad.safetensors` of the five time-index-0 client rounds; the step counts are 2, 3, 1, 1, 2.

| File | Bytes | SHA-256 |
|---|---|---|
| `t0000_c0002/meta.json` | 280 | `796d4c1c3b81040fdd2150d3556093b75d1f1ca4eca42e77fe0454960ae3b0ee` |
| `t0000_c0002/step000_grad.safetensors` | 12,612,736 | `79a309a868ed8428fe7ab2040a67825abf37aff353fcc2bca1904deef0f89f99` |
| `t0000_c0002/step001_grad.safetensors` | 12,612,736 | `0cfdfc3bc9a05d9105ac04d08b9fd9a8ead3852d3c13c3575edb2e53bd1a4370` |
| `t0000_c0026/meta.json` | 393 | `2f475d8a86f8c337e111414c0d68e6a0d9718271c2d631dfe76860b03ba94e4e` |
| `t0000_c0026/step000_grad.safetensors` | 12,612,736 | `4f3da4a516c76f629f5e7e6b417eb95de97bb638ece738689d45431fab2b5892` |
| `t0000_c0026/step001_grad.safetensors` | 12,612,736 | `2b2fccb15830b444dae70f47c0c580f9f3eb9140ec925adaa45d4ee9c9fa28b1` |
| `t0000_c0026/step002_grad.safetensors` | 12,612,736 | `d8cee48f6a0278728a63325a1687bb0541f4683265fd53c6a081970302e2e9b6` |
| `t0000_c0055/meta.json` | 170 | `6da7293551ad7eade1fd8d7846d6f4298ca08164ea9bda34c20369bc125b1884` |
| `t0000_c0055/step000_grad.safetensors` | 12,612,736 | `a271c7d09bfe38a2b6347088ba95dfb189443b05912f8e3cc095e1ab7c5cb808` |
| `t0000_c0075/meta.json` | 168 | `43eacf9cd7cbb5b1dfb7fc5af6c031ca818de6a475895c1213edbe1bbdcc9ac0` |
| `t0000_c0075/step000_grad.safetensors` | 12,612,736 | `d9f5a9f12f8bb181d5fab1118ce2c2ee619b80678165b8751eee10c6eaeab5d9` |
| `t0000_c0086/meta.json` | 282 | `5bd53055f71e2ed906f2b1f3cb3ceccdffb840fb00864f7663df520fce062fa7` |
| `t0000_c0086/step000_grad.safetensors` | 12,612,736 | `6167d489cf105cf77e71dab957a13ac313eea68c79d8724598a77b8b7d8998a7` |
| `t0000_c0086/step001_grad.safetensors` | 12,612,736 | `236d29f16d13369cd2e9792ba469ce716c3aadce1b48748134fac1108fd1001e` |

The step counts 2, 3, 1, 1, 2 agree with the committed `optimizer_steps.per_client_round` of
`results/phase4/forensics/f5_gradient_forensics.json` (`3cc0c10`), whose schedule lists the time-index-0 clients
2, 26, 55, 75 and 86. The table was recorded on 2026-10-04 by reading the files only (v1 §3). It has not been
re-verified against the payloads since then.

**R4 access boundary** (CHANGED from v1 §4's construction text; P5v2-D06). The F6 path
`mean_state(load_client_round(dir)["grads"])` is **not** used: `forensics.gradients.load_client_round` also opens
every `step{k:03d}_grad_clipped.safetensors` and `step{k:03d}_delta.safetensors`. P5-A instead uses a restricted reader
with these properties:
1. it opens a file only if its relative path is in the table above;
2. it refuses absolute paths, `..` components, backslashes, duplicate allowlist entries and a second read of the same
   file;
3. it reads the whole file once, checks the byte count and the SHA-256 of exactly those bytes, and deserialises those
   same bytes;
4. after construction, every allowlisted file must have been read exactly once;
5. it never enumerates the directory.

The clipped and delta files, and the time-index-1 directories, coexist in the collection. Their presence is not an
error. They are never opened, hashed or listed.

The math is unchanged: `mean_state` (float64 sum over steps, divided by the step count, stored as fp32) is applied to
the pre-clip gradients only. The R4 start/end state files are frozen by the R4 index. P5-A does not need them: the R4
input depends only on the gradient files, so P5-A does not open them.

**Verification before any training** (fail closed; P5v2-D07, CHANGED secondary check). Any failure stops every
control of the affected representation before training, with status `INCOMPLETE (PROVENANCE)` (section 11).
1. **Primary, identity.** Both `index.jsonl` files match the hashes above. Every R2/R3 payload read passes
   `load_states(verify=True)`, and every R4 file read matches the table. Opening any other file under the R4
   `gradients/` directory is a protocol violation.
2. **Primary, record keys.** Within each whole index, `(time_index, client_id)` is unique, compared as integers.
   Otherwise abort.
3. **Primary, population.** The training population (section 4) has exactly 80 / 80 / 5 records. Every record has
   geometry `{num_layers 24, modules [q_proj, k_proj, v_proj, o_proj], rank 8, hidden 2048}` and layout
   `layer_major_qkvo_AtB`. Every constructed input is fp32 with shape `[1, 2048, 1536]`, and every element is finite.
4. **Primary, R4 metadata.** Each `meta.json` names its own `(time_index, client_id)`, and its steps are numbered
   `0..n-1` with n = 2, 3, 1, 1, 2.
5. **Secondary, semantic integrity.** The recomputed population RMS, computed with the F6 expression
   `float(torch.stack(xs).pow(2).mean().sqrt())` over the fp32 tensors of `E`, agrees with the committed F6
   `input_stats.train_rms`: `|recomputed - committed| <= 1e-6 * |committed|`. Committed values: R2
   0.004050884395837784, R3 0.0020410344004631042, R4 0.0010228796163573861 (`d3c4eb3`). This check guards the
   construction code path, not file identity, and does not replace the hash checks. The maximum is checked in section
   5.4. The F6 p99.9 quantile is **not** recomputed in v2 (CHANGED from v1, which also compared it).

**Code identity** (INHERITED, v1 §3). F6 ran at `e8100e9`. The construction functions are AST-identical at `afc9e33`.

**Committed-evidence provenance** (VERIFIED from local Git objects on 2026-10-04; no payload read). Every commit below
is an ancestor of the v1 commit `52a0dd4`.

| Short | Full SHA | Content used by v2 |
|---|---|---|
| `d3c4eb3` | `d3c4eb3d631e0280533f35eb652fb24f68870079` | F6 records `results/phase4/screen/f6_r2_balanced_effective_state.json` (blob `39f9c2082b7716f96308bf533649e120759f3205`), `f6_r3_balanced_effective_delta_r8.json` (blob `98e8ce23b32a1827889bacd869c52bc6e9a1fcc6`), `f6_r4_mean_step_gradient.json` (blob `1edf73bb3e321fbd5983b9fc4a6ccf6f30ab8caf`). This commit is the only commit that touches these three paths; the blobs are identical at `52a0dd4`. Fields used: `input_stats.train_max_abs`, `input_stats.train_rms`, `split.train`, `snapshot_index_sha256`, `provenance.git.commit` |
| `e8100e9` | `e8100e9dd19ed4df98777646d6907d718790014e` | the F6 execution commit named by `provenance.git.commit` of all three records; `cmd_forensic_screen` computes `train_max_abs = float(max(xs[i].abs().max() for i in train_idx))` and `train_rms = float(torch.stack([xs[i] for i in train_idx]).pow(2).mean().sqrt())` |
| `eb771bb` | `eb771bb907dbccfd3cc1e9a6dd2ee8a0322aaaa1` | `results/phase3/tgap/federated_state_seed1_stats.json` `per_snapshot` (selection re-derivation, section 6) |
| `3cc0c10` | `3cc0c10004989651932be66b9ebdbc1528821088` | `results/phase4/forensics/f5_gradient_forensics.json` `schedule` and `optimizer_steps.per_client_round` |
| `203be49` | `203be49e6a709f3fc296fb0c76a4a5172bde44f2` | `docs/phase4_findings.md` (P4-D5, known R4 properties) |
| `5cc2324` | `5cc2324baa5a9df14e70f20e7afc7b57c7c6b7e8` | `docs/phase4_preregistration.md` (S2/S3/S7 numbers) |
| `ca0a17f` | `ca0a17f7fe172cc4eb0bdbac0392d1997207eb1f` | `docs/reproduction_protocol.md` §6 (launch-record rules) |

The committed records give: `split.train` = 80 / 80 / 5; `snapshot_index_sha256` = the Phase-3 hash for R2 and R3 and
the F5 hash for R4 (table above); `provenance.git.commit` = `e8100e9…` for all three.

## 4. Representation construction and training populations (INHERITED, v1 §4; P5v2-D04, P5v2-D05)

Common to all three: Qwen1.5-1.8B LoRA r = 8 on q/k/v/o, 24 layers, LoRA scale s = alpha / r = 16 / 8 = 2; Phi
layout `layer_major_qkvo_AtB`; AE input `[1, 2048, 1536]` fp32 (n = 3,145,728 elements); latent `[64, 32, 24]`
(1/64 of the elements).

| ID | Definition per record | Construction |
|---|---|---|
| R2 | balanced canonical factors of `M = s B_end A_end` per module (`B_c = Q_B U sqrt(S)`, `A_c = sqrt(S) Vᵀ Q_Aᵀ`; the largest-magnitude entry of each left vector is positive; components with σ <= 1e-12 σ_max are set to 0) | `balanced_effective_state(end, s)` |
| R3 | balanced factors of the best rank-8 approximation of `dM = s (B_end A_end - B_start A_start)` | `balanced_effective_delta(start, end, s, rank=8)` |
| R4 | mean over the client round's optimizer steps of the pre-clip factor gradients (float64 sum, divided by the step count, stored fp32) | `mean_state(pre-clip gradients read by the restricted reader)` |

Then `x = layout.forward(rep, geometry)`. No other transformation is applied.

**Training populations** = the F6 temporal training split (`split_indices(records, "temporal", 0.2, ...)`):
* R2, R3: time indices 0-15 of the Phase-3 federated set, 80 records;
* R4: time index 0 of the F5 collection, 5 records.

The population is selected from index metadata **before any payload is opened**. A validation-split record is never
passed to a payload reader.

## 5. Normalisation: exact training-population maximum (CHANGED; P5v2-D08 to P5v2-D14)

### 5.1 Rule

For representation r with training population `T_r` (80 / 80 / 5 records, section 4), in the order `E` of section 6:

```text
m_r = max over x in T_r, over all elements j, of |x_j|        (exact; every element of every snapshot)
s_r = m_r / 0.95                                               (one frozen scalar per representation)
encode:  x_n = x / s_r
decode:  x_hat = s_r * dec(enc(x_n))
```

* **Mode identifier:** `global_exact_maxabs_train` (NEW). The existing mode `global_maxabs_train` keeps its historical
  meaning, `p99.9(|x|) / 0.95` (Phase 4, v1), and is not redefined. The unapproved scaffolding mode
  `global_train_maxabs` is not resurrected; it was removed and has no status.
* **Population:** all 80 / 80 / 5 records. The scale is fitted once, before either control is selected, and both
  controls of a representation share it. It is never fitted per control or on the selected snapshots.
* **Unconditional:** the scale is always applied, whatever its value.
* **Exact:** no quantile, no subsampling, no stride.
* **No clipping:** inputs are never clamped.
* **Loss:** MSE in the normalised space. **Gates and metrics:** in the original space, after de-normalisation.
* **Uplink:** 0 bytes. The frozen scalar is known to both ends.

### 5.2 Arithmetic and storage conventions

| Item | Convention |
|---|---|
| Inputs | the constructed fp32 Phi tensors of section 4 |
| Maximum | `m_r = max_i max_j |x_ij|`, evaluated on the fp32 values of every element of every one of the 80 / 80 / 5 tensors (no subsampling). The absolute value and the maximum of fp32 values are exact in fp32; the result is converted to a Python float64 without change, so `m_r` is always an fp32-representable number |
| Scale | `s_r = m_r / 0.95` in float64 (IEEE 754 double, round to nearest). The frozen value is that float64, written as a decimal literal that parses to exactly that float64 |
| Application | `x / float32(s_r)` and `y * float32(s_r)` in fp32. The frozen float64 scalar is cast once to fp32 (existing `Normalizer._scale_like`). The frozen scalar of section 5.3 is applied; the recomputed maximum is never used to form the applied scale |
| Metrics | float64 sums on the CPU over the fp32 values of truth and prediction (section 9) |

### 5.3 Frozen values (DERIVED FROM COMMITTED EVIDENCE; NOT RECOMPUTED FROM PAYLOADS)

The committed F6 records hold `input_stats.train_max_abs`. F6 computed it as
`float(max(xs[i].abs().max() for i in train_idx))` over the same fp32 Phi tensors and the same temporal training split
(`cmd_forensic_screen` at `e8100e9dd19ed4df98777646d6907d718790014e`). That is exactly `m_r` of section 5.2. Each
record's `split.train` is 80 / 80 / 5, and its `snapshot_index_sha256` equals the frozen index hash of section 3. All
three records were committed in `d3c4eb3d631e0280533f35eb652fb24f68870079` and are unchanged since (section 3,
committed-evidence provenance).

| R | Source file (git blob at `52a0dd4`) | Field | Stored value `m_r` | `s_r = m_r / 0.95` (float64) | `float32(s_r)` |
|---|---|---|---|---|---|
| R2 | `results/phase4/screen/f6_r2_balanced_effective_state.json` (`39f9c20`) | `input_stats.train_max_abs` | 0.032825905829668045 | **0.034553585083861103** | 0.03455358371138573 |
| R3 | `results/phase4/screen/f6_r3_balanced_effective_delta_r8.json` (`98e8ce2`) | `input_stats.train_max_abs` | 0.014964050613343716 | **0.015751632224572334** | 0.015751631930470467 |
| R4 | `results/phase4/screen/f6_r4_mean_step_gradient.json` (`1edf73b`) | `input_stats.train_max_abs` | 0.0643031895160675 | **0.06768756791165001** | 0.06768757104873657 |

Each stored `m_r` is exactly representable in fp32 (R2 `0x3d067475`, R3 `0x3c752bc7`, R4 `0x3d83b164`), as an fp32 maximum
must be. With these scales, the fp32-normalised maximum is 0.95000005 (R2), 0.94999999 (R3) and 0.94999993 (R4), all
within 8e-8 relative of 0.95. These values are DERIVED by float64 arithmetic on the committed JSON fields only (no
payload was read) and are **frozen** by the approval of v2. They are not yet confirmed against the payloads; sections
5.4 and 5.4.1 state how a later, separately authorised preflight confirms them.

**Literal note.** The R2 literal `0.034553585083861103` and the shorter Python representation `0.0345535850838611`
parse to the same float64 (`0x1.1b101ebca1af3p-5`). The frozen value is that float64; either literal denotes it. The
R3 and R4 literals are the shortest round-trip representations.

**Conditioning tradeoff** (DERIVED from the same committed records). Under exact maxabs, the normalised RMS is
`train_rms / s_r` = 0.117 (R2), 0.130 (R3) and 0.0151 (R4). The v1 p99.9 scaling gave 0.199 / 0.227 / 0.105. Exact
maxabs scales the bulk of R2/R3 down by a factor of about 0.58 relative to v1, and the bulk of R4 down by 0.145, because a
few extreme values set the scale. Removing the range obstruction therefore makes the bulk smaller, not easier to fit.
P4-D5 identified small input scale as a fit problem under `none`. Exact maxabs trades part of that gain for range
coverage, most strongly for R4. This tradeoff is deliberate and is the main reason R4 is secondary (P5v2-D03).

### 5.4 Input preflight: verification of the frozen scale (NEW; P5v2-D11)

The input preflight (section 5.4.1) and, again, every training invocation recompute `m_r` from the hash-verified
training population (section 3) and require:

```text
|m_r(recomputed) - m_r(frozen)| <= 1e-6 * m_r(frozen)
```

The comparison is made on `m_r`, the measured quantity, not on `s_r`. Both sides are float64; the bound is inclusive;
the check fails if the recomputed value is not finite. Bitwise equality is expected, because F6 used the same
construction code and the same fp32 tensors, and is recorded (`bitwise_equal`). If the check fails, that
representation stops before training with `INCOMPLETE (PROVENANCE)` for both controls, and the frozen value is
**never replaced**, neither by the recomputed value nor by any other; adopting a different value would require a
separately reviewed protocol revision. If it passes, the run applies the **frozen** `s_r` of section 5.3, not a
scale formed from the recomputed maximum.

### 5.4.1 Preflight form and its relation to the run (APPROVED choice 6 of section 21; P5v2-D11a)

The input checks run first as a **separately invocable CPU-only input preflight**, executed only after the
implementation commit has been reviewed and a separate written authorisation names the preflight (section 16), and
before any GPU training. The preflight:

* builds no AE, allocates no CUDA context, starts no training and publishes no start record;
* opens only the training payloads permitted by section 3 (R2/R3 through `load_states(verify=True)`, R4 through the
  restricted reader); no validation-split, clipped-gradient or delta file is opened;
* performs every check of section 3 (items 1-5), section 5.4, section 5.5 and section 5.6, re-derives the section-6
  selection from metadata and records the identities of the selected records and that each has `sig > 0`;
* publishes one record `preflight/preflight_<k>.json` under the run root (section 13), with the protocol, execution
  and configuration identities and every recomputed value. A mismatch is recorded, stops that representation, and
  never substitutes a value.

**A passing preflight is never a bypass.** Every later training invocation repeats every check of sections 3, 5.4,
5.5 and 5.6 on the bytes it reads itself. It then compares its own results with the most recent preflight record:
protocol commit, configuration SHA-256, frozen `m_r` / `s_r`, index hashes, population keys, per-record payload
identities, the SHA-256 of every constructed Phi tensor, the recomputed `m_r` and RMS. A representation trains only
if the preflight passed for it, the invocation's own checks pass, and the two agree exactly. Any difference (stale
evidence, a changed input identity, a preflight failure) makes both controls of that representation
`INCOMPLETE (PROVENANCE)` as a pre-start status (section 12). The preflight record and the training invocation must
name the same protocol commit, implementation commit and configuration SHA-256; otherwise the preflight evidence is
stale, and a new separately authorised preflight is required before training.

**Repetition.** At most 3 preflights exist (`preflight_1` to `preflight_3`), and none after the first training
invocation record (section 13.1). A preflight may be repeated only after an engineering or launch error, never with a
changed value of this document. Every preflight record, passing or failing, is retained and cited by the launch
record. Training uses only the most recent preflight record.

### 5.5 Post-normalisation range check (NEW; P5v2-D12)

After the scale is applied, the run checks, for every snapshot of `T_r`:

```text
max_j |x_n,j| <= 0.95 * (1 + 2e-6)
```

and, over the whole population, `max_i max_j |x_n,ij| >= 0.95 * (1 - 2e-6)`.

**Meaning.** `x_n` are the fp32 values produced by the same application that training uses, `x / float32(s_r)` with
the frozen `s_r`. Their absolute maxima are converted exactly to float64 and compared with the float64 bounds
`0.95 * (1 + 2e-6)` and `0.95 * (1 - 2e-6)`, inclusive. The 2e-6 is a relative allowance on the **normalised
values**, around the target 0.95. It is not a tolerance on the scale, not a fit criterion and not a clipping
threshold.

**Allowance.** Two roundings separate `x / float32(s_r)` from the exact `x / s_r`: casting `s_r` to fp32, and the
fp32 division. Each is at most u = 2^-24 (about 6.0e-8) relative, so together at most about 1.2e-7. If the preflight
passes with a non-zero difference, the recomputed maximum differs from the frozen `m_r` by at most 1e-6 relative,
in either direction. Both bounds of the range check therefore hold whenever section 5.4 passes: 2e-6 covers
1e-6 + 1.2e-7 with margin. A range-check failure after a passing preflight indicates a defect, not a data property.
With the expected bitwise agreement, the normalised population maximum lies within 1.2e-7 of 0.95; section 5.3
lists the frozen values.

A failure is `INCOMPLETE (PROVENANCE)`. The check is about numerical integrity, not a pass criterion, and it never
clips or adjusts anything. Values with `|x_n| >= 1` cannot occur within the allowance. The existing `range_report`
statistics (fraction with `|x_n| >= 1`, quantiles of `|x_n|` per factor) are reported for every control and never
gated.

### 5.6 Degenerate and non-finite inputs (NEW; P5v2-D13)

* **Empty population:** impossible after section 3 item 3, which requires 80 / 80 / 5 records. If it occurs, the
  result is `INCOMPLETE (PROVENANCE)`.
* **Non-finite input element:** section 3 item 3 fails, giving `INCOMPLETE (PROVENANCE)`. Finiteness of every
  element is verified before `m_r` is computed, so a NaN can neither enter nor be skipped by the maximum.
* **All-zero population (`m_r = 0`):** `s_r` would be 0, which is not finite and positive. The result is
  `INCOMPLETE (PROVENANCE)` and no division is performed. The existing `Normalizer` check rejects a scale that is
  not finite or is <= 0.
* **A selected snapshot with zero signal (`sig_i = 0`):** the control is `INCOMPLETE (PROVENANCE)` (section 9.2).
  Such a snapshot can exist inside a population with `m_r > 0`.

### 5.7 Relation to v1 (historical provenance)

v1 used `global_maxabs_train` with the frozen p99.9 scales 0.020353251623087806 / 0.009008480041827019 /
0.009784620783634835, a per-tensor strided quantile with stride 4 / 4 / 1 for 2**26 elements, and NumPy 2.4.6
`"linear"` interpolation (v1 §5). Those values and that algorithm remain the historical record of v1. v2 neither uses
nor recomputes them. The committed F6 p99.9 values are cited only to state the conditioning tradeoff (section 5.3).

## 6. Controls and selection (INHERITED, v1 §6; P5v2-D15 to P5v2-D18)

**Ordering.** `E` = the training records of section 4, sorted by `(int(time_index), int(client_id))` ascending.
Uniqueness is verified (section 3 item 2), so no tie-break exists or is needed. `N = |E|`; positions are 0-based.
`round_half_up(v) = floor(v + 0.5)`, evaluated exactly in integer arithmetic as `floor((2a + b) / (2b))` for
`v = a / b` with `a, b >= 0`.

| Control | k | Positions |
|---|---|---|
| `single_snapshot` | 1 | `floor((N - 1) / 2)` |
| `fixed_subset4` | 4 | `round_half_up(j (N - 1) / 3)`, j = 0, 1, 2, 3 |

The rule uses record metadata only, never tensor values or AE outputs. If the positions of `fixed_subset4` were not
distinct, the result would be `INCOMPLETE (PROVENANCE)`. This cannot happen for N = 80 or N = 5.

**Selected records** (DERIVED; reproduced on 2026-10-04 from committed metadata only:
`results/phase3/tgap/federated_state_seed1_stats.json` `per_snapshot`, committed in `eb771bb`, whose
`snapshot_index_sha256` is the frozen Phase-3 hash; and the F5 `schedule`, `3cc0c10`):

| R | N | `single_snapshot` (t, client) | `fixed_subset4` positions -> (t, client) |
|---|---|---|---|
| R2, R3 | 80 | position 39 -> (7, 91) | 0, 26, 53, 79 -> (0, 2), (5, 32), (10, 43), (15, 84) |
| R4 | 5 | position 2 -> (0, 55) | 0, 1, 3, 4 -> (0, 2), (0, 26), (0, 75), (0, 86) |

For N = 80, `j (N - 1) / 3` = 0, 26.33, 52.67, 79 rounds to 0, 26, 53, 79. For N = 5 it is 0, 1.33, 2.67, 4, which
rounds to 0, 1, 3, 4. No value is exactly half-integral, so the half-up convention is stated for completeness but
does not change the result.

**Known properties** (disclosed, `phase4_findings.md` §§6-7, `203be49`). R2/R3 (0, 2) is a round-0 record: its start
state has B = 0, so its R3 increment has rank <= 8. R4 (0, 55) and (0, 75) took one optimizer step at B = 0: their
R4 A part is exactly zero, so half of the Phi columns of the R4 `single_snapshot` input are zero. That snapshot still
has `sig > 0`.

**Selected subset vs normalisation population.** The scale of section 5 is fitted on all of `E` (80 / 80 / 5). The
controls train and evaluate only on their 1 or 4 selected snapshots. The subset-mean predictor (section 8) uses only
the 4 selected snapshots of `fixed_subset4`.

## 7. Execution plan, order and prerequisites (CHANGED wording; P5v2-D19, P5v2-D20)

**Fixed order** (positions 1-6):

| # | Control | Designation |
|---|---|---|
| 1 | R2 `single_snapshot` | primary |
| 2 | R2 `fixed_subset4` | primary |
| 3 | R3 `single_snapshot` | primary |
| 4 | R3 `fixed_subset4` | primary |
| 5 | R4 `single_snapshot` | secondary |
| 6 | R4 `fixed_subset4` | secondary |

**No result-dependent pruning.** A control's outcome never skips, reorders or changes another control. In particular,
a `single_snapshot` failure never skips the `fixed_subset4` that follows it. A training starts unless exactly one of
these frozen rules prevents it:

| Rule | Effect | Section |
|---|---|---|
| provenance or input check of the representation fails, in the invocation itself or by disagreement with the most recent preflight record (section 5.4.1) | both controls of that representation: `INCOMPLETE (PROVENANCE)`, not trained; other representations still run | 3, 5 |
| a selected snapshot has `sig = 0` | that control: `INCOMPLETE (PROVENANCE)`, not trained | 9.3 |
| C0 fails for that control | that control: `NOT INFORMATIVE (TANH RANGE)`, not trained | 8.2 |
| the training/time ceiling has no full 1,800 s allocation left | the run is closed (13.5): this and every later control without a terminal record become `INCOMPLETE (RESOURCE STOP)`, not trained | 12 |
| the byte allowance cannot hold that control's maximum outputs | the run is closed (13.5): this and every later control without a terminal record become `INCOMPLETE (RESOURCE STOP)`, not trained | 13 |
| a `NOT_STARTED` control already has a finalized file under its control directory or ledger names (an anomalous state) | that control: `INCOMPLETE (PROVENANCE)`, not trained | 12, 13 |

No other skip, retry or fallback exists. A control that has already started is never started again, and its persistent
state is recovered instead (section 12). The invocation stops after a byte-ceiling stop, a publication failure, an
operator interrupt or a capability failure (section 13.5). The controls it did not reach stay `NOT_STARTED`.

## 8. Predictors, C0 and training

### 8.1 Predictors and their roles (P5v2-D21)

All predictions are evaluated in the original space, on the control's unique selected snapshots.

| Predictor | Definition | Role |
|---|---|---|
| AutoEncoder | `p = s_r * dec(enc(x / s_r))` at iteration 3,000, eval mode | **gated** (C1-C4) |
| zero | `p = 0` | reported only |
| Tanh-range ceiling | `p = s_r * clamp(x / s_r, -1, 1)` | gated **only as C0** (before training); reported after training |
| subset mean (`fixed_subset4` only) | `p = m = (1/4) Σ_{j in S} x_j`, float64 | enters C4 as the comparator; its own metrics are reported |

### 8.2 C0: Tanh-range feasibility (INHERITED rule, CHANGED rationale; P5v2-D22)

For each control, before training, C0 applies that control's gate criteria (section 10) to the Tanh-range ceiling in
place of the AE: C2 and C3 for `single_snapshot`; C2, C3 and C4 for `fixed_subset4`. If any criterion fails, no
Tanh-headed decoder can pass that control at this scale. The control is then `NOT INFORMATIVE (TANH RANGE)` and is
not trained.

**Relation to exact maxabs.** Under section 5, every training element satisfies
`|x / s_r| <= 0.95 (1 + 2e-6) < 1`, so the clamp is the identity on the training population. The ceiling equals `x`
up to fp32 rounding (`p = float32(s_r) * (x / float32(s_r))`). The relative error per element is at most about 2 fp32
ulps, so the ceiling's RSE is below 1e-12 and its cosine equals 1 to within rounding. A synthetic CPU check on random
tensors (2026-10-04, before approval; no payload) gave RSE of about 5e-16. C0 therefore passes on the C2/C3 side for every control with `sig > 0`,
including under the strict `single_snapshot` standard.

On the C4 side of `fixed_subset4`, the ceiling's pooled RSE (below 1e-12) must not exceed 0.8 x the subset-mean
pooled RSE. This can fail only if the four selected snapshots are identical or nearly identical. For four identical
snapshots the subset-mean SSE is exactly 0, so C4 (section 10, P5v2-D32) holds for the ceiling only if the ceiling's
SSE is exactly 0, i.e. if `float32(s_r) * (x / float32(s_r))` reproduces every element bitwise. No such case is known
for the selected records.

C0 is kept as an executed, recorded integrity check of the scale and selection: a C0 failure under v2 would reveal a
broken invariant. The v1 branch for genuine range infeasibility is now unreachable by construction. It is kept only so
that an impossible state cannot be trained silently.

### 8.3 Architecture and optimisation (INHERITED, v1 §7; P5v2-D23 to P5v2-D27)

| Item | Value |
|---|---|
| Architecture | reconstructed ResNet-3: 3x3 stem 1 -> 1 channel (no bias) + BN + ReLU; 6 stride-2 conv stages 1 -> 2 -> 4 -> 8 -> 16 -> 32 -> 64 (no bias) + BN + ReLU; 3 + 3 residual blocks (64); 6 ConvT stages; reflection pad + 7x7 conv (bias) + Tanh; 493,483 parameters |
| Input / latent | `[1, 2048, 1536]` fp32 -> `[64, 32, 24]`, 1/64 of the elements |
| Loss | MSE (`F.mse_loss`, mean over elements) in the normalised space |
| Optimiser | Adam, lr 2e-4, betas (0.9, 0.999), eps 1e-8, weight decay 0; no scheduler, no gradient clipping, no AMP |
| Iterations | exactly 3,000 for a completed training; no early stopping, no extension |
| Batching | `bs = min(4, k)`. Each iteration takes the first `bs` entries of a permutation of the k selected snapshots, drawn from `numpy_rng(1, "ae_batches")`, and draws a new permutation when fewer than `bs` remain. `single_snapshot`: k = 1, bs = 1. `fixed_subset4`: k = 4, bs = 4. No snapshot is duplicated |
| AE seed | {1} only; no extra seeds, no fallback trials; no stability claim (P5v2-D26) |
| Seed derivation | init: `torch.manual_seed(derive_seed(1, "ae_init"))` inside `torch.random.fork_rng`; data order: `numpy_rng(1, "ae_batches")` |
| Determinism | `configure_determinism(True)`: deterministic algorithms, cuDNN deterministic, benchmark off, TF32 off, `CUBLAS_WORKSPACE_CONFIG=:4096:8`. If a backend refuses a deterministic kernel, the training is `INCOMPLETE (INTERRUPTED)`; determinism is never disabled |
| Device | the local CUDA GPU; allocator cap = free VRAM - 256 MiB |
| Training curve | train-mode batch MSE (normalised space) at iterations 1, 50, 100, ..., 3,000; reported, never gated |

### 8.4 Checkpoint and evaluation (INHERITED, v1 §8; P5v2-D27)

Only the iteration-3,000 checkpoint is saved, evaluated and gated. There is no best-loss or validation selection and
no BatchNorm recalibration. After iteration 3,000: `ae.eval()`, no gradients. Each unique selected snapshot `x` (fp32)
is passed through the AE individually as a batch of 1 using the BatchNorm running statistics:
`p = s_r * dec(enc(x / s_r))` on the GPU in fp32. Both tensors are then moved to the CPU, and section 9 is computed in
float64. Replicated training batches are never evaluated.

## 9. Metrics (P5v2-D28; edge-case rules P5v2-D31 to P5v2-D33)

### 9.1 Definitions

Let snapshot i have truth `x_i` and prediction `p_i` from any predictor, both in the original space, each with
n = 3,145,728 elements. All quantities are float64 sums over the fp32 element values:

```text
sig_i = Σ_j x_ij²                 (signal energy)
sse_i = Σ_j (p_ij - x_ij)²        (squared reconstruction error)
hat_i = Σ_j p_ij²                 (prediction energy)
dot_i = Σ_j p_ij x_ij

MSE_i        = sse_i / n
zero-MSE_i   = sig_i / n           (MSE of the zero predictor)
RSE_i        = sse_i / sig_i       (relative squared error; defined iff sig_i > 0)
cos_i        = dot_i / sqrt(sig_i * hat_i)   (defined iff sig_i > 0 and hat_i > 0)
normratio_i  = sqrt(hat_i / sig_i)          (defined iff sig_i > 0)
```

Pooled over a set S of snapshots: `SSE = Σ_{i in S} sse_i`; `SIG`, `HAT` and `DOT` likewise;
`RSE_S = SSE / SIG`; `cos_S = DOT / sqrt(SIG * HAT)`; `MSE_S = SSE / (|S| n)`. Pooled values are ratios of summed
sums. They are **never** means or medians of per-snapshot ratios, and no snapshot is excluded from a pool.

Per-snapshot sums use the float64 reduction of the numerical library. Pooled sums use the correctly rounded float64
sum (`math.fsum`) of the per-snapshot sums. Agreement at the last-ulp level is not guaranteed across library
versions. A gate value within a few ulps of its threshold is therefore reported as such (section 11.4).

**Subset-mean predictor** (`fixed_subset4` only): `m = (1/4) Σ_{j in S} x_j`, computed in float64 over the four
selected snapshots S. It is not the normalisation population and not all 80 / 5 records. The same m is the prediction
for each of the four snapshots.

### 9.2 Undefined and non-finite values

| Case | Rule | v1 / implementation | Status |
|---|---|---|---|
| `sig_i = 0` for a selected snapshot | the control is `INCOMPLETE (PROVENANCE)`, checked before C0; not trained | same rule (v1 §9) | INHERITED |
| `hat_i = 0` (e.g. AE output identically 0) | `cos_i` undefined. If gated, C1 and C3 fail. Never replaced by 0 or 1 | same | INHERITED |
| non-finite AE output element | C1 fails, so the control is `FIT FAIL` (the training completed; evaluation shows the failure) | same | INHERITED |
| non-finite sum or ratio (from inf/NaN outputs) | the value is not finite. If gated, C1 and the criterion fail. It is kept in every pool, never filtered | same | INHERITED |
| non-finite training loss | the training stops: `INCOMPLETE (NON-FINITE LOSS)`; no checkpoint, no rerun | same (v1 §7) | INHERITED |
| zero predictor, `cos` undefined (`hat = 0`) | reported as undefined with reason `hat_zero`; never gated | same | INHERITED |
| subset mean `m = 0` | its `cos` undefined, reported only; C4 uses `RSE`, which stays defined because `SIG > 0` | not stated in v1 | NEW (P5v2-D31) |
| subset-mean pooled `SSE = 0` (four identical snapshots) | C4 keeps its mathematical meaning: `RSE_AE <= 0.8 x 0` holds iff `RSE_AE = 0` exactly | v1 §10 and the implementation **fail** C4 whenever the subset-mean RSE is 0 | CHANGED (P5v2-D32) |
| denominators | no epsilon is ever added to a denominator or a threshold | same | INHERITED |

### 9.3 JSON representation (NEW; P5v2-D33)

Every metric field is a finite JSON number or `null`. For each `null`, a sibling map `undefined` gives the reason:
`"sig_zero"`, `"hat_zero"`, `"non_finite:nan"`, `"non_finite:+inf"` or `"non_finite:-inf"`. JSON is written with
`allow_nan = false`. Non-finite values are never written as strings in metric fields. (The current implementation
writes `"nan"`/`"inf"` strings through `json_safe`; this changes.)

Every criterion record holds:
* the compared value(s);
* the threshold literal;
* the comparison operator (`<=` or `>=`);
* `pass` (boolean);
* if the result follows from an undefined or non-finite value, `fail_reason`.

## 10. Gates (P5v2-D29, P5v2-D30, P5v2-D34)

All comparisons are IEEE 754 float64 comparisons against the literal thresholds below. Every bound is inclusive, and
no epsilon relaxes or tightens any threshold. A required criterion with an undefined or non-finite value fails.

| # | Criterion | `single_snapshot` | `fixed_subset4` | Status |
|---|---|---|---|---|
| C0 | feasibility: the Tanh-range ceiling meets that control's C2, C3 (and C4), before training | C2s, C3s | C2, C3, C4 | INHERITED rule (section 8.2) |
| C1 | every element of every AE reconstruction is finite, and every gated value is defined and finite | required | required | INHERITED |
| C2 | per-snapshot RSE | **`RSE_1 <= 0.01`** (C2s) | `RSE_i <= 0.50` for **every** i in S | single: CHANGED (P5v2-D29); subset: INHERITED |
| C3 | per-snapshot cosine | **`cos_1 >= 0.99`** (C3s) | `cos_i >= 0.90` for **every** i in S | single: CHANGED (P5v2-D29); subset: INHERITED |
| C4 | better than the subset mean | not applicable | `RSE_S(AE) <= 0.8 * RSE_S(m)`, the right side evaluated in float64 | INHERITED rule; zero-SSE edge CHANGED (P5v2-D32) |

A control is `FIT PASS` iff every required criterion holds: C1, C2s, C3s for `single_snapshot`; C1, C2, C3, C4 for
`fixed_subset4`. Otherwise it is `FIT FAIL`.

**Two different standards** (P5v2-D30).
* `single_snapshot` is a **memorisation standard**: one fixed snapshot is reconstructed to within 1 % squared error.
  RSE <= 0.01 and cosine >= 0.99 are new numbers, approved for v2 (choice 2 of section 21). They are neither a paper
  requirement nor a previously approved threshold (section 0).
* `fixed_subset4` is a weaker, separately named **subset fit standard**. It reuses the Phase-4 S2/S3/S7 numbers
  (`phase4_preregistration.md` §4, `5cc2324`) on training snapshots, per snapshot, plus input dependence (C4).
* Meeting either standard says nothing about generalisation, because the AE is evaluated on the snapshots it was
  trained on.

**Arithmetic note** (DERIVED). For RSE < 1, `cos >= sqrt(1 - RSE)`: the prediction's distance to the ray of `x` is
at least `|x| sin θ`. So C2s (RSE <= 0.01) implies cos >= 0.99499 > 0.99, and C3s can never be the binding criterion
of `single_snapshot`. It is kept as a recorded criterion. For the subset standard, C2 (RSE <= 0.50) implies only
cos >= 0.7071, so C3 (>= 0.90) is binding there.

```text
control_status(R, c):                                    # c in plan order (section 7)
    if a provenance or input check fails:     INCOMPLETE (PROVENANCE)        # not trained
    if some selected sig_i = 0:               INCOMPLETE (PROVENANCE)        # not trained
    if not C0(R, c):                          NOT INFORMATIVE (TANH RANGE)   # not trained
    if a resource rule forbids the start:     INCOMPLETE (RESOURCE STOP)     # not trained (sections 12, 13)
    publish the start record; train (section 8.3)
        non-finite loss:                      INCOMPLETE (NON-FINITE LOSS)
        GPU time reaches 1,800 s:             INCOMPLETE (RESOURCE STOP)
        crash, OOM, backend refusal:          INCOMPLETE (INTERRUPTED)    # the invocation continues
        operator interrupt (e.g. Ctrl+C):     INCOMPLETE (INTERRUPTED)    # records published, invocation stops
    evaluate the iteration-3,000 checkpoint (section 8.4); GPU time must still be < 1,800 s
    gate (C1 + C2/C3 [+ C4]):                 FIT PASS or FIT FAIL
    if the checkpoint or record cannot be finalized: INCOMPLETE (PUBLICATION)   # section 13
```

## 11. Control statuses, outcomes and interpretation (P5v2-D35 to P5v2-D38)

### 11.1 Control statuses

| Status | Meaning | Trained? | Kind |
|---|---|---|---|
| `FIT PASS` | completed, evaluated, every required criterion holds, checkpoint and record finalized | yes | scientific (gate met) |
| `FIT FAIL` | completed, evaluated, at least one required criterion fails, checkpoint and record finalized | yes | scientific (gate not met) |
| `NOT INFORMATIVE (TANH RANGE)` | C0 failed: the gate is unattainable for any Tanh-headed decoder at this scale. Under v2 this reveals a broken invariant (section 8.2) | no | precondition |
| `INCOMPLETE (PROVENANCE)` | an identity, input, scale, range, `sig = 0` or existing-path check failed | no | could not run |
| `INCOMPLETE (RESOURCE STOP)` | a time or byte rule prevented the start, stopped the training at 1,800 s, or closed the run | either | could not complete |
| `INCOMPLETE (NON-FINITE LOSS)` | the training loss became non-finite | yes | could not complete |
| `INCOMPLETE (INTERRUPTED)` | crash, OOM, interruption, deterministic-backend refusal, or process death after the start record | yes | could not complete |
| `INCOMPLETE (PUBLICATION)` | the training completed, but its checkpoint or record could not be finalized (filesystem error, or a process death between the end record and the control record). Computed metrics are kept as reported-only evidence when a record can still be written, but are not a gate result | yes | could not complete (NEW, P5v2-D35) |
| `INCOMPLETE (NOT STARTED)` | the run was closed (section 13.5) before this control started, and it has no other recorded pre-start status | no | could not run (NEW, P5v2-D35) |

A gate failure (`FIT FAIL`) is a scientific outcome. Every other non-pass status is an inability to complete the
diagnostic and is never reported as a gate failure. Outcomes (section 11.2) are computed only from terminal control
records at run closure. Intermediate summaries report persistent states, not outcomes.

### 11.2 Per-representation outcome (INHERITED table; CHANGED status set)

| `single_snapshot` | `fixed_subset4` | Outcome for that representation |
|---|---|---|
| `FIT PASS` | `FIT PASS` | `CAPACITY FIT DEMONSTRATED` |
| `FIT FAIL` | any status | `CAPACITY FIT NOT DEMONSTRATED` |
| any status | `FIT FAIL` | `CAPACITY FIT NOT DEMONSTRATED` |
| every other combination: at least one of `NOT INFORMATIVE`, `INCOMPLETE (...)`, and no `FIT FAIL` | | `INCONCLUSIVE` |

Examples: (`FIT PASS`, `INCOMPLETE (PUBLICATION)`) gives `INCONCLUSIVE`; (`INCOMPLETE (PROVENANCE)`, `FIT FAIL`)
gives `CAPACITY FIT NOT DEMONSTRATED`; (`NOT INFORMATIVE`, `NOT INFORMATIVE`) gives `INCONCLUSIVE`. Exactly one
outcome applies to each of the 9 x 9 combinations; rows 2 and 3 overlap only where both give the same outcome. The
required test covers all 81.

### 11.3 Bounded meaning of the outcome phrases (CHANGED definition; P5v2-D36)

The historical phrases are retained. Under v2 they mean exactly:
* `CAPACITY FIT DEMONSTRATED`: under this protocol, the fixed AE met the **memorisation standard** on the one
  `single_snapshot` record **and** the **subset fit standard** on the four `fixed_subset4` records of that
  representation's training population.
* `CAPACITY FIT NOT DEMONSTRATED`: under this protocol, at least one completed control did not meet its standard. It
  does not show that the representation is incompressible or that the AE has insufficient capacity.
* `INCONCLUSIVE`: no completed control failed, but at least one control did not complete or was not informative.
  It is not evidence either way.

**Required qualifier** in every report of an outcome: "under the P5-A v2 protocol (fixed ResNet-3, exact
training-population max-abs scaling to 0.95, Adam 2e-4, 3,000 iterations, final checkpoint, AE seed 1, training
snapshots only)".

### 11.4 Structured outcome record (CHANGED; P5v2-D37)

Each representation's outcome is a structured object:

* `representation`, `designation` (`primary` or `secondary`);
* both control statuses;
* the outcome enum, and the qualifier above;
* for each criterion: its value, threshold, operator and `pass`, and the margin `value - threshold`. Any gate value
  within 4 float64 ulps of its threshold is flagged `near_threshold: true`;
* `standards`: `{"single_snapshot": "memorisation (RSE <= 0.01, cos >= 0.99)", "fixed_subset4": "subset fit (RSE <= 0.50, cos >= 0.90 per snapshot; C4)"}`;
* `does_not_establish`: the fixed list of section 1;
* `authorises` (section 11.5).

The run summary lists R2 and R3 (primary) before R4 (secondary). There is **no combined pass** across
representations, and no field aggregates them.

A blanket forbidden-word filter is not used (CHANGED from the v1 implementation). Machine-written records contain
only the enumerated statuses and outcomes, the fixed qualifier and the fixed limitation texts of this document.
Exception messages appear only in `detail` fields. Claims in human-written reports (findings) are reviewed against
section 1 and this section. These reports must not describe an outcome as proof of compressibility or
incompressibility, of sufficient or insufficient capacity, of stability, of a causal explanation of the F6
failures, or of reproduction of the paper. They may, and should, state these limitations explicitly.

### 11.5 What an outcome authorises (INHERITED; P5v2-D38)

`CAPACITY FIT DEMONSTRATED` for a representation authorises only **drafting** a separate scale-normalised re-screen
preregistration for it. It does not authorise that re-screen, FAF, operational server semantics, P5-B/C/D or any
downstream experiment. Other outcomes authorise nothing; the reviewer then decides between a new, separately
preregistered experiment and stopping this line of work. A primary-representation outcome is not changed by the
secondary one, and vice versa.

## 12. Time ceilings and training accounting (P5v2-D39 to P5v2-D42)

| Ceiling | Value | Status |
|---|---|---|
| Trainings | at most 6 = 3 representations x 2 controls x 1 seed; every started training counts | INHERITED |
| Per-training GPU wall time | 1,800 s hard limit | INHERITED |
| Aggregate charged GPU time | 10,800 s; a training starts only if `trainings_started < 6` **and** `charged_total + 1,800 <= 10,800` | INHERITED |
| Retries | none after a training has started | INHERITED |
| New client rounds | 0 | INHERITED |

**GPU wall time of a training** (INHERITED boundary, CHANGED evaluation-overrun rule; P5v2-D40). The clock is
monotonic. It starts immediately before `ae.to(device)`, after `torch.cuda.synchronize`, and stops after the end of
the section-8.4 evaluation, after `torch.cuda.synchronize`. It is stopped on every exit path. The limit is checked
before every iteration and before evaluation; reaching 1,800 s stops the training at once with
`INCOMPLETE (RESOURCE STOP)`. If the measured time at the end of evaluation is >= 1,800 s, the training is also
`INCOMPLETE (RESOURCE STOP)` and publishes no checkpoint (NEW explicit rule; v1 did not state the evaluation case).
Measured time is recorded as measured, never clamped, and may exceed 1,800 s when one iteration crosses the limit.

**Setup time** (CPU: hash checks, construction, scale verification, C0) is recorded per invocation as
`setup_wall_s`. It is not ceilinged and is never added to GPU time.

**Timing fields** (NEW; P5v2-D41). Each started training has:

| Field | Meaning |
|---|---|
| `gpu_wall_s_measured` | the measured GPU wall time, or `null` if unknown |
| `timing_status` | `measured` or `unknown_process_death` |
| `gpu_wall_s_charged` | the value used by the aggregate ceiling: the measured time if known, otherwise 1,800 |
| `charge_reason` | `measured` or `conservative_full_allocation_after_unobserved_process_death` |

A charge is never reported as measured runtime. The 1,800 s charge after an unobserved process death is an
accounting convention. The real elapsed time is unknown and may exceed it: a hang inside a single kernel is not
interrupted by the per-iteration check.

**Persistent attempt states** (P5v2-D42). Each control `(R, c)` has these persistent states, derived only from
finalized files under the run root (section 13):

```text
NOT_STARTED  -- no finalized start record
STARTED      -- ledger/start_<R>_<c>.json finalized
ENDED        -- ledger/end_<R>_<c>.json finalized (training and evaluation over; timing recorded)
RECORDED     -- <control dir>/p5a_record.json finalized (terminal)
```

* **Start boundary.** `ae.to(device)` is called only after the start record has been finalized. A training without a
  finalized start record is therefore impossible. If the process dies before the start record is finalized, the
  control is still `NOT_STARTED`, and any surviving temporary file is counted (section 13). If it dies at any point
  after the start record is finalized, including before or during `ae.to(device)`, the control has started.
* **No retry.** A control in `STARTED`, `ENDED` or `RECORDED` is never trained again by any invocation. An uncertain
  start therefore always counts as a start.
* **Recovery of evidence** (no re-training, no re-evaluation, no payload read). A later invocation that finds `STARTED`
  without `ENDED` publishes the end record with `timing_status: unknown_process_death`, charged 1,800 s. It then
  publishes the control record with `INCOMPLETE (INTERRUPTED)`. If it finds `ENDED` without `RECORDED`, it publishes
  the control record with `INCOMPLETE (PUBLICATION)`, citing any finalized checkpoint. Recovery happens first in
  every later invocation.
* **Before a start** (provenance, path, environment or launch error), the control stays `NOT_STARTED`. Its pre-start
  status is recorded in that invocation's summary, not as a terminal record. The cause may be fixed and the control
  attempted again by a later permitted invocation (section 13.5), provided no value of this document changes. A code
  fix requires a newly reviewed implementation commit and a reviewed update of the launch record. Every invocation is
  recorded. A representation whose preflight (section 5.4.1) failed cannot train in any training invocation, because
  no new preflight is permitted once `ledger/invocation_1.json` exists; such a failure must be resolved before the
  first training invocation.

## 13. Artifacts, byte budget and publication (P5v2-D43 to P5v2-D47)

### 13.1 Run root and invocations (CHANGED; P5v2-D43)

* Run root: `<CGFED_RUNS>/phase5_p5a_v2_seed1/`. This is a new name, so v2 output can never mix with the v1 path
  `phase5_p5a_seed1/`, which must never be created.
* **Preflight invocations** (section 5.4.1). Preflight 1 requires that the run root does not exist; it creates the
  root and publishes `preflight/preflight_1.json` only. Preflights 2 and 3 require a root that holds only earlier
  preflight records. At most 3 preflights; none after `ledger/invocation_1.json` exists. Each preflight record's
  publication is its own no-clobber capability check.
* **Training invocation 1** requires a root that holds at least one preflight record and no `ledger/` or control
  directory. Without a preflight record it is refused before it publishes anything.
* **Training invocations 2 and 3** (re-entry after a pre-start failure, an invocation stop, or a process death)
  require an existing, not yet closed root (section 13.5). Its `ledger/invocation_1.json` must name the same protocol
  commit, the same configuration SHA-256 and the same launch-record identity (path and SHA-256), and the same
  implementation commit unless a reviewed launch-record update names the new one.
* At most **3 training invocations**. A fourth, or any invocation after closure, is refused before it publishes
  anything.
* A training invocation k publishes `ledger/invocation_<k>.json` first. That publication is also the filesystem
  capability check: if the no-clobber primitive fails, the invocation stops before any control.
* At its end, a training invocation publishes `ledger/summary_<k>.json` with every control's state. At most 3
  invocation records and 3 summaries exist.
* The launch-record identity recorded by an invocation is a citation for provenance. The software cannot verify a
  human authorisation and does not claim to; authorisation exists only in the written launch record itself.

### 13.2 Retained artifacts and fixed size caps (NEW; P5v2-D44)

| Artifact | Per | Count (max) | Cap (bytes, decimal) |
|---|---|---|---|
| `preflight/preflight_<k>.json` (section 5.4.1) | preflight | 3 | 300,000 |
| `ledger/invocation_<k>.json` | training invocation | 3 | 50,000 |
| `ledger/summary_<k>.json` | training invocation | 3 | 200,000 |
| `ledger/start_<R>_<c>.json` | control | 6 | 20,000 |
| `ledger/end_<R>_<c>.json` | control | 6 | 20,000 |
| `ledger/failure_<R>_<c>.json` (only after a cap or publication failure) | control | 6 | 100,000 |
| `<R>_<kind>_<c>/config.resolved.yaml` + `config.sha256.json` + `run_metadata.json` | control | 6 sets | 150,000 per set |
| `<R>_<kind>_<c>/p5a_record.json` | control | 6 | 500,000 |
| `<R>_<kind>_<c>/autoencoder_p5a_final.safetensors` (only `FIT PASS` / `FIT FAIL`) | control | 6 | 2,100,000 |

Every finalized P5-A artifact is in this table. Checkpoint metadata is inside the checkpoint file and its cap.

**Estimated worst-case footprint** (sum of all caps at their maximum counts):

```text
preflight            3 x 300,000                                                    =    900,000
invocation records   3 x  50,000                                                    =    150,000
summaries            3 x 200,000                                                    =    600,000
per control          6 x (20,000 + 20,000 + 100,000 + 150,000 + 500,000 + 2,100,000) = 17,340,000
total                                                                               = 18,990,000 <= 20,000,000
```

The remaining 1,010,000 bytes absorb surviving temporary files, which are counted at their full size (a temporary
name that survives as a second hard link of a finalized file is counted twice). The estimate is a planning bound;
the reservation check of section 13.3 is what is enforced. (Engineering correction made at
approval: the draft's total of 18,090,000 = 750,000 + 17,340,000 was arithmetically correct for the artifacts it
listed, but it did not list the preflight records required by choice 6. Adding them changes only this estimate. The
20,000,000-byte ceiling, the per-artifact caps of the draft and the reservation rule are unchanged.)

Four different quantities are kept apart: the **estimated worst-case footprint** above (a planning bound, never
reported as usage); the **per-artifact caps** (hard limits on each file); the **reservation** of section 13.3 (caps
of still-required evidence, used only in the check); and the **measured footprint** (the exact bytes of all regular
files under the run root, the only byte count ever reported as usage).

The checkpoint cap allows for 1,981,796 bytes of fp32 AE state (493,483 parameters plus BatchNorm buffers, 152
tensors) plus the safetensors header and metadata. An in-memory serialisation of a freshly initialised AE with
representative metadata measured 1,996,628 bytes (2026-10-04, synthetic software evidence; no input, no training). No
tensor dump (one Phi input is 12,582,912 bytes) can fit any cap.

**Not retained:** reconstructions, normalised tensors, latents, optimiser state and copies of inputs.

### 13.3 One run-wide budget with reservation (CHANGED; P5v2-D45)

* **Ceiling:** 20,000,000 decimal bytes = the sum of the exact sizes of **all regular files** under the run root:
  finalized artifacts and any surviving temporary file, in every subdirectory. It replaces v1's split into
  "6 x 3,000,000 + 2,000,000 run-level reserve" and the implementation's cumulative 18,000,000-byte limit on
  non-ledger files. There is no separate reserve.
* **Exact sizes.** Each artifact is serialised to bytes before publication, so its size is known exactly. An artifact
  larger than its cap is never published. For a control artifact that is an engineering failure: the control is
  `INCOMPLETE (RESOURCE STOP)`, P5-A stops (a byte-ceiling stop, section 13.5), and the failure record (within its own
  cap) records the byte counts. For a run-level artifact (preflight record, invocation record, summary) the artifact is
  not published and the invocation stops with the error reported to the operator; a preflight or invocation that
  could not publish its first record has published nothing and does not count.
* **Reservation check.** Before publishing any artifact, and before starting any training, the run requires:

  ```text
  F + Σ caps(Q) <= 20,000,000
  ```

  F is the measured footprint. Q is every artifact that may still be required, excluding files already finalized:
  * the artifact about to be published;
  * this training invocation's summary, and the invocation record and summary of every later training invocation
    that could still be permitted (section 13.1);
  * the evidence artifacts (start, end, failure, provenance set, record) of every control not yet `RECORDED`, which
    includes the terminal records issued at closure;
  * at a training start, that control's checkpoint; when publishing a checkpoint, that checkpoint.

  During a preflight, Q is the preflight record plus every training-stage artifact of the worst case (all invocation
  records, summaries and per-control artifacts), so a preflight can never consume bytes needed by the run.
  (Engineering clarification at approval: the draft reserved only one further invocation; reserving every still
  possible invocation is the conservative reading of "every artifact that may still be required".)

  Because the §13.2 caps already include every possible artifact and the worst case fits, the check can fail only if a
  measured file exceeds expectations, e.g. a surviving temporary file. It is still enforced. A checkpoint is never
  published unless the remaining evidence of every control and every summary is still reserved.
* If the check fails before a training, the run is closed by a byte-ceiling stop (section 13.5): that control and every
  later control without a terminal record become `INCOMPLETE (RESOURCE STOP)` without training. Their records and
  the summary are published within the caps already reserved for them. Caps are used only in this check. **Reported
  byte counts are always the measured footprint**, never a reservation.

  A record cannot contain its own size. A file reports the footprint measured immediately before it is published, as
  `retained_bytes_before_this_file`, and the summary reports the footprint before the summary.

### 13.4 Publication (INHERITED v1 §12 semantics, specified; P5v2-D46, P5v2-D47)

* **Temporary vs finalized.** An artifact is written to a temporary file owned by this invocation, in the same
  directory: the name carries a per-invocation random token, it is created exclusively, and it is flushed and fsynced.
  It is then published to its final name with an operation that fails if the final name exists (no-clobber, atomic on
  the same filesystem). On Windows/NTFS and POSIX this is `os.link(temporary, final)` followed by removing the
  temporary name. A file is finalized only once that publication completes. The previous implementation session
  reported that `os.link` raised `FileExistsError` on an existing target on the reference NTFS volume. That is
  earlier software evidence; it is re-checked by the capability check of section 13.1.
* **Temporary cleanup.** Only temporary files created by this invocation may be removed, after their own failed or
  incomplete write. A temporary file that survives (e.g. a process death) is counted in the footprint and is never
  presented as an artifact.
* **No overwrite, no deletion.** A pre-existing file is never overwritten or deleted. A finalized P5-A artifact is
  never deleted, including to recover budget or to undo a partial publication. An existing output path of a control,
  found before its start, makes that control `INCOMPLETE (PROVENANCE)` (section 7).
* **Filesystem capability failure.** If the no-clobber primitive is unavailable (e.g. hard links unsupported), the
  invocation stops at the capability check. There is no fallback to overwriting semantics.
* **Order within a completed control:** end record, checkpoint, provenance set, record. Each is published
  individually, after the reservation check.
* **Partial publication.** If a later file fails after an earlier one was finalized (e.g. the checkpoint is finalized
  and the record write fails), the finalized files are kept. Deleting a finalized checkpoint would destroy the only
  evidence of a completed training and would break the no-deletion rule.
  * The control is `INCOMPLETE (PUBLICATION)`.
  * A failure record lists the finalized and failed files with their exact byte counts, if it can still be
    published.
  * The invocation stops (section 13.5). If the cause is the byte ceiling, the run is closed and every later control
    without a terminal record becomes `INCOMPLETE (RESOURCE STOP)`. Otherwise the later controls stay `NOT_STARTED`
    for a later permitted invocation.
* **`INCOMPLETE` trainings** publish their start record, end record, provenance set and `p5a_record.json`: status,
  partial training curve and timing fields. They publish **no checkpoint**. Their bytes, time and training count stay
  charged.
* Training completion (the end record) is recorded separately from the control's outcome (the record), so a
  publication failure never relabels a completed training as not completed.

### 13.5 Invocation stops and run closure (NEW; P5v2-D43)

**A training invocation stops** (publishes its summary, starts no further control) after:
* a byte-ceiling stop;
* a publication failure;
* an operator interrupt;
* a capability failure. If the capability check itself cannot publish, nothing is published.

A crash, OOM, non-finite loss or backend refusal ends that control `INCOMPLETE`. The invocation continues with the
next control in plan order.

**The run is closed** when any one of these holds:
* every control has a terminal record;
* a byte-ceiling stop or a time-ceiling stop occurred;
* the third invocation has ended.

A **time-ceiling stop** is the section-12 start rule refusing a start (`trainings_started = 6`, or no full 1,800 s
allocation left). A single training that reaches its own 1,800 s limit is not a time-ceiling stop: that control is
`INCOMPLETE (RESOURCE STOP)`, its time stays charged, and the invocation continues with the next control, whose start
is then decided by the section-12 rule (as in v1 §12).

At closure, every control without a terminal record receives one, published by the closing invocation within the
reserved caps (section 13.3):
* `INCOMPLETE (RESOURCE STOP)` if a resource ceiling closed the run;
* otherwise, its last recorded pre-start status (`INCOMPLETE (PROVENANCE)` or `NOT INFORMATIVE (TANH RANGE)`);
* otherwise, `INCOMPLETE (NOT STARTED)`.

If a terminal record cannot be published because its path is already occupied by a file that P5-A did not finalize
(the anomalous state of section 7), that file is left untouched, and the closing summary records the control's
status, the occupied path and its size.

The closing summary computes the outcomes of section 11.2, and it is the only summary that does so. No invocation
follows closure.

**Death of the third training invocation** (engineering completion of the closure rule, made at approval). If the
process of training invocation 3 dies before it publishes `ledger/summary_3.json`, the run is neither closed nor
re-enterable as written. A **closure-only step** then completes it. It is not a fourth invocation: it requires a root
with `ledger/invocation_3.json` and without `ledger/summary_3.json`; it reads no payload, builds no AE, starts no
control and publishes no invocation record. It performs the evidence recovery of section 12 (no retraining), publishes
the closure terminal records above and publishes `ledger/summary_3.json` as the closing summary. All of these files
are already counted in the section-13.2 caps and reserved by section 13.3, so no budget changes. It runs at most once.

**Other cases.** A run that is not yet closed after invocation 1 or 2 stays open until a later permitted training
invocation; at most three exist, so every run that is continued reaches closure. Each re-entry first recovers
evidence, then re-checks inputs (section 5.4.1) and attempts only controls that are still `NOT_STARTED`.

## 14. Per-control record and provenance (CHANGED fields; P5v2-D48)

`<R>_<kind>_<c>/p5a_record.json`, label `PHASE5-DIAGNOSTIC`, contains:

* the protocol commit SHA (from the launch record), path and blob; the execution commit, branch and tree state; the
  exact command; and the launch-record identity;
* representation, designation, control and plan position;
* input root, `index.jsonl` SHA-256 and population count;
* every selected record with its position, `(time_index, client_id)`, `file_sha256`, `start_file_sha256` and adapter
  hashes (R2/R3), or the SHA-256 of every R4 gradient file read (R4);
* the SHA-256 of every selected Phi input tensor;
* normalisation: mode `global_exact_maxabs_train`, frozen `m_r` and `s_r`, recomputed `m_r`, relative difference,
  tolerance, `bitwise_equal`; the range check (section 5.5) and the `range_report`;
* the secondary RMS check;
* C0 values and booleans; seed and derived seed values;
* the training curve; the final metrics per snapshot and pooled for the AE, zero, Tanh-range ceiling and subset-mean
  predictors, with `undefined` reasons;
* C1-C4 values, thresholds, operators, booleans, margins and `near_threshold`;
* status, and timing fields (section 12);
* `retained_bytes_before_this_file`, trainings started and charged GPU time.

The checkpoint `autoencoder_p5a_final.safetensors` (format `cg_fedllm.resnet_ae/v1`) holds the iteration-3,000 state,
including the BatchNorm buffers. Its metadata holds the normalisation mode and scalar, the control and the protocol
commit.

**Preflight record** `preflight/preflight_<k>.json` (section 5.4.1), label `PHASE5-DIAGNOSTIC`: protocol commit,
execution commit and tree state, configuration SHA-256, exact command and launch-record identity; for each
representation: index SHA-256, population keys and count, per-record payload identities (R2/R3 file and adapter
hashes; R4 gradient-file SHA-256), the SHA-256 of every constructed Phi tensor, geometry/dtype/shape/finiteness
results, the recomputed `m_r` with frozen value, relative difference, tolerance and `bitwise_equal`, the RMS check,
the range check, the selected records with `sig > 0`, and a pass flag with any failure detail; plus
`trainings_started: 0`, `ae_built: false`, `cuda_initialized: false` and `retained_bytes_before_this_file`.

**Committed evidence after a run** (separate reviewed change): `results/phase5/p5a_v2/` with the six
`p5a_record.json` files, every preflight record, every summary and the ledger JSON files; a findings document; and
deviation rows. Negative, inconclusive and incomplete outcomes are committed with the same weight as passes.

## 15. Required software validation before a real run (synthetic only; P5v2-D49)

All tests use synthetic tensors and synthetic `index.jsonl` / gradient trees in temporary directories. They need no
pretrained model, Hub access or CUDA, and they never write into `CGFED_RUNS`. Oracles are written independently of
the code under test.

1. **Construction:** R2/R3/R4 from synthetic states and synthetic `GradientDumper` trees equal the F6 functions
   bitwise; no R0/R1 path exists.
2. **Access boundary:** with clipped, delta and validation-round files present, an instrumented `Path.open`,
   safetensors loader and reader show that only allowlisted files are opened, each exactly once; no directory is
   listed. This also covers allowlist-only R4 construction (no start/end state file is opened) and refusal of path
   escapes, unlisted paths, duplicate entries and repeated reads.
3. **Identity failures** stop before training: index hash, file hash, tensor hash, size, missing file, metadata
   mismatch, step numbering, wrong count, wrong geometry, duplicate keys.
4. **Selection:** exact positions for N = 80 and N = 5; invariance to input order; integer half-up rounding;
   indistinct positions abort.
5. **Exact maxabs:** the maximum over all elements of all tensors (including a maximum in the last element of the
   last tensor, and a negative maximum); the float64 scale; fp32 application; frozen-value application; the
   `1e-6 * m` tolerance at both sides of the boundary; the range check at both sides of `0.95 (1 ± 2e-6)`; the
   all-zero population; non-finite input; no clipping; de-normalisation; the mode identifier is distinct from
   `global_maxabs_train`, which keeps its historical p99.9 behaviour.
6. **C0** under exact maxabs passes for valid inputs, and fails (no trainer call) for four identical snapshots.
7. **Metrics:** hand-computed per-snapshot and pooled values; pooled != mean of ratios; float64 accumulation;
   `sig = 0`, `hat = 0`, NaN, +inf and -inf; `null` with reasons in JSON, with `allow_nan = false`.
8. **Gates:** exact inclusive equality and the adjacent float64 values for 0.01, 0.99, 0.50, 0.90 and 0.8x; the
   single-snapshot strict standard; C4 not applied to `single_snapshot`; C4 with subset-mean SSE = 0 (passes only if
   AE SSE = 0); per-snapshot failure while the pooled values pass.
9. **Training path:** a real small ResNet forward/backward on CPU with a legal geometry; batch size 1 without
   cloning; batch size 4; the permutation-redraw rule; seeded init; final checkpoint in eval mode with BatchNorm
   buffers kept; production locks (3,000 iterations, seed 1, lr, betas, eps, wd, mode, label) and refusal of
   overrides.
10. **Runner:** the six-control order; a `single_snapshot` failure does not skip `fixed_subset4`; a provenance
    failure isolates one representation; C0 skip without a start record.
11. **Aggregation:** all 81 status combinations of section 11.2; outcomes only at closure; structured outcome
    fields; no combined pass; primary/secondary ordering.
12. **Time accounting with a fake clock:** start boundary; per-iteration and before-evaluation limit; evaluation
    overrun gives `INCOMPLETE (RESOURCE STOP)` without a checkpoint; aggregate reservation at exactly 10,800;
    `trainings_started = 6`; measured vs charged fields; unobserved process death charged 1,800 s as
    `unknown_process_death`.
13. **Attempt states:** a death before the start record (no start counted, surviving temporary counted); a death
    after the start record, before or after `ae.to(device)` (counted, never retrained); recovery of `STARTED` and
    `ENDED` controls into records; invocation stops and run closure (13.5), including closure terminal records; a
    fourth invocation, or one after closure, refused; a v1 root name refused.
14. **Bytes:** exact footprint including temporaries and every subdirectory; cap enforcement per artifact; the
    reservation check before a training start and before each publication, including checkpoint publication; the
    estimated worst case of section 13.2 (18,990,000 bytes with preflight records) fits; a measured footprint is
    reported, not caps.
15. **Publication:** a collision introduced after the check leaves the intruding file and earlier finalized files
    intact; a failure after the first publication keeps the checkpoint and yields `INCOMPLETE (PUBLICATION)`;
    interruption cleans only own temporaries; no overwrite fallback when links fail.
16. **History:** the `ResultLabel` schema still accepts every earlier label; Phase-2/3/4 behaviour of the shared
    helpers is unchanged (the existing suite).
17. **Preflight** (section 5.4.1): the standalone CPU route builds no AE, initialises no CUDA context, starts no
    training and writes no start record; it opens only allowed files; a mismatch is recorded and substitutes nothing;
    a later training invocation re-checks inputs and refuses missing, failed, stale or disagreeing preflight
    evidence.

One optional GPU smoke test is allowed by the launch record: random synthetic tensors of the real shape, at most 50
iterations, outputs in a temporary directory that is deleted, nothing recorded or reported. Not allowed before the
launch: any AE forward or backward on a real Phase-3/4 tensor, or any timing on real inputs.

## 16. Launch record (required before any real step; P5v2-D50)

A separate reviewed document (`reproduction_protocol.md` §6, `ca0a17f`), completed before any real P5-A step:

* the full SHA of the approved v2 protocol commit, and of the reviewed implementation commit;
* configuration path, its SHA-256, and the statement that no override is used;
* the exact command(s), including the CPU-only input preflight (section 5.4.1) and the training invocation(s);
* environment: interpreter path, Python version and package versions (pinned lock), CUDA/driver/GPU facts and
  `nvidia-smi`, and `CGFED_RUNS`;
* the frozen input identities (section 3) and the frozen `m_r` / `s_r` (section 5.3);
* the selected records and the fixed control order (sections 6 and 7);
* the time and byte ceilings and caps (sections 12 and 13); the run root `phase5_p5a_v2_seed1/`, confirmed absent
  before the first preflight (section 13.1);
* the decision criteria and outcome table (sections 10 and 11);
* the planned output and committed-evidence locations;
* the preflight record(s) that the training invocation will rely on, once they exist (a launch-record update);
* a separate, dated written human authorisation for each real step, naming whether it covers the CPU preflight only
  or also the training run. Software never treats a flag or file as proof of that authorisation.

## 17. Forbidden without separate written authorisation

* any real P5-A input preflight, training or evaluation (this document authorises none);
* changing any value of this document after any P5-A output exists, including a preflight result or a training curve;
* changing the representation scope, populations, selection, normalisation, thresholds, seed, budget or checkpoint
  rule;
* any hyperparameter, architecture, iteration, learning-rate, seed or normalisation-mode search; any extra seed or
  fallback trial;
* opening validation-split payloads, R4 clipped/delta files, D2, held-out or benchmark data in P5-A;
* FAF, P5-B, P5-C, P5-D, any re-screen, any downstream evaluation;
* new client rounds or TGAP collection; Tier A, 7B or external GPUs;
* executing v1, or creating `<CGFED_RUNS>/phase5_p5a_seed1/`.

## 18. Approval path

**Approval basis.** On 2026-10-04 a written human instruction approved the six design choices of section 21
(exact training-population max-abs scaling; the strict single-snapshot memorisation standard; R2/R3 primary and R4 a
prespecified secondary always scheduled; C4 as the plain inequality at a zero subset-mean error; one run-wide
20,000,000-byte ceiling for all finalized artifacts with evidence reservation; a separately invocable CPU-only input
preflight before any GPU training). The same instruction authorised one protocol-only commit of this document and its
audit response, and reconciliation of the implementation with that commit. It did not authorise any real preflight,
payload access, GPU use or run. It approved those six choices, not every sentence of the draft; the consistency
corrections made before the commit are listed in `docs/phase5_v2_audit_response.md` section 0. This document does
not name a reviewer or record a review event beyond that instruction.

Sequence:
1. Done: human approval of the six choices (above).
2. A protocol-only commit adds this document marked approved for implementation. That commit contains no code. Its
   full SHA is recorded outside the document.
3. The existing uncommitted implementation is reconciled with that SHA, and the synthetic and full pinned software
   gates are run.
4. A human reviews the implementation diff and authorises its commit.
5. The launch record (section 16) is prepared with the actual protocol and implementation SHAs.
6. A separate written authorisation is given for the real CPU input preflight; the preflight runs (section 5.4.1).
7. A separate written authorisation is given for the training run, citing the preflight record.
8. Only the frozen v2 plan is executed.

## 19. Decision register

Status: INHERITED (from v1 `52a0dd4`, section given) or APPROVED (v2) (CHANGED / NEW; approved 2026-10-04 for
implementation only, section 18).

| ID | Decision | Value | v1 | Status |
|---|---|---|---|---|
| P5v2-D00 | Go / no-go | v2 replaces v1 for execution; approved for implementation and synthetic validation only; no real preflight or run without a launch record and written authorisation | D00 | APPROVED (v2, CHANGED) |
| P5v2-D01 | v1 handling | v1 stays an approved historical record, never executed; its run root is never created | — | APPROVED (v2, NEW) |
| P5v2-D02 | Question and limits | §1: fit standards under a range-covering scale; no capacity, compressibility, stability or causal claim | §1 | APPROVED (v2, CHANGED wording) |
| P5v2-D03 | Representations and designation | R2, R3 primary; R4 prespecified secondary, always scheduled independently of earlier outcomes and subject to the same provenance, prerequisite, resource-stop and no-retry rules; R0/R1 excluded | D01 | APPROVED (v2, CHANGED: designation; choice 3) |
| P5v2-D04 | Construction | F6 math, s = 2, rank 8, fp32, `layer_major_qkvo_AtB` | §4 | INHERITED |
| P5v2-D05 | Data and populations | frozen indexes; R2/R3 t 0-15 (80); R4 t 0 (5); selection before payloads; no validation payload | D02, D05 | INHERITED |
| P5v2-D06 | R4 access | the 14 files of §3 only; restricted reader; F6 `load_client_round` not used; clipped/delta files and t1 never opened | D06 | APPROVED (v2, CHANGED: construction path stated) |
| P5v2-D07 | Identity and integrity | primary checks of §3; secondary: F6 `train_rms` to rel. 1e-6 (max checked in §5.4; p99.9 no longer recomputed) | D06 | APPROVED (v2, CHANGED: secondary set) |
| P5v2-D08 | Normalisation mode | `global_exact_maxabs_train`, a new identifier; `global_maxabs_train` keeps its p99.9 meaning | D09 | APPROVED (v2, CHANGED; choice 1) |
| P5v2-D09 | Statistic and target | `m_r` = exact max of |x| over the whole population; `s_r = m_r / 0.95` | D07, D08 | APPROVED (v2, CHANGED statistic; choice 1; target inherited) |
| P5v2-D10 | Fitting population | the whole training population; one scalar per representation, shared by both controls; unconditional; no clipping | D10, D12 | INHERITED |
| P5v2-D11 | Frozen scales and scale check | `m_r` R2 0.032825905829668045, R3 0.014964050613343716, R4 0.0643031895160675; `s_r` R2 0.034553585083861103, R3 0.015751632224572334, R4 0.06768756791165001; recomputed `m_r` must satisfy `|rec - frozen| <= 1e-6 * frozen`; frozen value applied; a mismatch stops and never replaces the value | D11 | APPROVED (v2, CHANGED values; DERIVED from committed F6 `d3c4eb3d631e0280533f35eb652fb24f68870079`; choice 1) |
| P5v2-D11a | Preflight form | separately invocable CPU-only input preflight before any GPU training (§5.4.1); no AE, no CUDA, no training attempt; evidence re-validated by every training invocation, never a bypass | — | APPROVED (v2, NEW; choice 6) |
| P5v2-D12 | Range check | every normalised max <= 0.95 (1 + 2e-6), population max >= 0.95 (1 - 2e-6); `range_report` reported only | D12 | APPROVED (v2, NEW check) |
| P5v2-D13 | Degenerate inputs | empty, non-finite, all-zero population: `INCOMPLETE (PROVENANCE)` | D12 | APPROVED (v2, CHANGED: explicit) |
| P5v2-D14 | Arithmetic conventions | float64 scale from fp32 max; fp32 application; float64 metrics | §5 | APPROVED (v2, CHANGED: restated for exact max) |
| P5v2-D15 | Ordering | sort by integer `(time_index, client_id)`; uniqueness verified, abort on duplicates; no tie-break | D14 | INHERITED |
| P5v2-D16 | `single_snapshot` | `E[floor((N - 1)/2)]`: R2/R3 (7, 91); R4 (0, 55) | D14 | INHERITED |
| P5v2-D17 | `fixed_subset4` | `E[round_half_up(j (N - 1)/3)]`, integer arithmetic: R2/R3 (0,2) (5,32) (10,43) (15,84); R4 (0,2) (0,26) (0,75) (0,86) | D15 | INHERITED |
| P5v2-D18 | Batching | `bs = min(4, k)`; `numpy_rng(1, "ae_batches")` permutations; no duplication | D16 | INHERITED |
| P5v2-D19 | Execution order | R2 s, R2 f, R3 s, R3 f, R4 s, R4 f; no outcome prunes or reorders | D36 | INHERITED (stated as a table) |
| P5v2-D20 | Start prerequisites | only the frozen rules of §7 prevent a start | D36 | APPROVED (v2, CHANGED: enumerated) |
| P5v2-D21 | Predictor roles | AE gated; zero reported; Tanh ceiling gated only as C0; subset mean is the C4 comparator | §8 | INHERITED (stated) |
| P5v2-D22 | C0 | Tanh ceiling meets the control's C2/C3 (+C4); else `NOT INFORMATIVE`; under exact maxabs an integrity check | D13 | APPROVED (v2, CHANGED rationale) |
| P5v2-D23 | Architecture | reconstructed ResNet-3, unchanged | D17 | INHERITED |
| P5v2-D24 | Optimiser and loss | MSE (normalised space); Adam 2e-4, (0.9, 0.999), 1e-8, wd 0 | D18 | INHERITED |
| P5v2-D25 | Iterations | exactly 3,000; no early stop or extension | D19 | INHERITED |
| P5v2-D26 | Seeds | {1}; no extra seeds or fallback trials; no stability claim | D20 | INHERITED (limit stated) |
| P5v2-D27 | Determinism, checkpoint, BatchNorm | `configure_determinism(True)`, never disabled; iteration 3,000 only; eval mode, running statistics, no recalibration | D21-D23 | INHERITED |
| P5v2-D28 | Metric definitions | §9.1, float64, original space, pooled = summed sums | D24 | INHERITED (written out) |
| P5v2-D29 | `single_snapshot` gate | C1, C2s RSE <= 0.01, C3s cos >= 0.99 (strict memorisation standard; new v2 numbers, not a restored earlier gate) | D25, D26 (0.50 / 0.90) | APPROVED (v2, CHANGED; choice 2) |
| P5v2-D30 | `fixed_subset4` gate | C1, C2 RSE <= 0.50 and C3 cos >= 0.90 for every snapshot, C4 pooled RSE <= 0.8 x subset mean; a separately named weaker standard | D25-D28 | INHERITED (standard named) |
| P5v2-D31 | Baseline undefined fields | reported as `null` with reason; never gated | §9 | APPROVED (v2, NEW: explicit) |
| P5v2-D32 | C4 at subset-mean SSE = 0 | the inequality itself: passes iff AE SSE = 0; no epsilon | D27 (fails) | APPROVED (v2, CHANGED; choice 4) |
| P5v2-D33 | JSON of undefined values | `null` + `undefined` reason map; `allow_nan = false`; no strings in metric fields | — | APPROVED (v2, NEW) |
| P5v2-D34 | Comparison semantics | float64, literal thresholds, inclusive, no epsilon; undefined fails | D28 | INHERITED |
| P5v2-D35 | Status set | adds `INCOMPLETE (PUBLICATION)` and `INCOMPLETE (NOT STARTED)`; gate failures separate from incomplete execution; outcomes only at run closure | §10 | APPROVED (v2, NEW statuses) |
| P5v2-D36 | Outcome table and phrases | v1 §11 table over the extended status set; phrases with the bounded meanings of §11.3; v2 qualifier | D28, D30 | APPROVED (v2, CHANGED definitions) |
| P5v2-D37 | Outcome records | structured fields, margins, near-threshold flag, primary first, no combined pass; no blanket word filter | D30 | APPROVED (v2, CHANGED) |
| P5v2-D38 | Authorisation by outcome | `DEMONSTRATED`: drafting a re-screen preregistration only; otherwise nothing | D37 | INHERITED |
| P5v2-D39 | Time ceilings | 6 trainings; 1,800 s each; 10,800 s aggregate; no retry after start | D31-D33, D36 | INHERITED |
| P5v2-D40 | Timing boundary | `ae.to(device)` to the end of evaluation, with synchronisation; an evaluation overrun is `RESOURCE STOP` with no checkpoint | §12 | APPROVED (v2, CHANGED: overrun rule) |
| P5v2-D41 | Timing fields | measured / status / charged / reason; 1,800 s conservative charge after an unobserved death | — | APPROVED (v2, NEW) |
| P5v2-D42 | Attempt states | `NOT_STARTED`, `STARTED`, `ENDED`, `RECORDED`; start record before `ae.to(device)`; evidence recovery without retraining | D36 | APPROVED (v2, NEW) |
| P5v2-D43 | Run root, invocations, closure | `phase5_p5a_v2_seed1/`; at most 3 invocations; capability check first; invocation-stop and run-closure rules of §13.5 | — | APPROVED (v2, NEW) |
| P5v2-D44 | Artifact caps | §13.2 table, including `preflight_<k>.json` 300,000; estimated worst case 18,990,000 bytes | — | APPROVED (v2, NEW; estimate corrected at approval for the preflight records) |
| P5v2-D45 | Byte budget | one run-wide 20,000,000-byte ceiling on all files under the run root; reservation check `F + Σ caps(Q) <= 20,000,000` before every publication and training start; no separate reserve | D34 | APPROVED (v2, CHANGED; choice 5) |
| P5v2-D46 | Publication | exclusive temporary, no-clobber link, own-temporary cleanup, no deletion of finalized files, no overwrite fallback | D35 | INHERITED (specified) |
| P5v2-D47 | Partial publication | keep finalized files; `INCOMPLETE (PUBLICATION)`; failure record; P5-A stops | D35 | APPROVED (v2, CHANGED: explicit) |
| P5v2-D48 | Record contents | §14 | §15 | APPROVED (v2, CHANGED fields) |
| P5v2-D49 | Software validation | §15 synthetic tests | §13 | APPROVED (v2, CHANGED list) |
| P5v2-D50 | Launch record | §16 fields and a separate written authorisation | §15, §17 | INHERITED (extended) |
| P5v2-D51 | Result label | `PHASE5-DIAGNOSTIC` (+ `DERIVED`) | D29 | INHERITED |

## 20. v1-to-v2 changes

Every item not listed here is unchanged from v1.

| Topic | v1 (`52a0dd4`) | v2 (approved) | Why |
|---|---|---|---|
| Normalisation mode | `global_maxabs_train`, `p99.9(|x|)/0.95`, strided quantile | `global_exact_maxabs_train`, `max(|x|)/0.95`, exact | audit item 1: remove the Tanh-range confound for every training value |
| Frozen scales | 0.020353251623087806 / 0.009008480041827019 / 0.009784620783634835 | 0.034553585083861103 / 0.015751632224572334 / 0.06768756791165001 | follows from the mode; DERIVED from committed F6 `train_max_abs` |
| Scale verification | recomputed `s` to rel. 1e-6 | recomputed `m_r` to rel. 1e-6, plus a post-normalisation range check | the exact max is the measured quantity |
| Secondary statistics | `train_rms`, `train_max_abs`, p99.9 | `train_rms` (max is primary in §5.4) | p99.9 is no longer part of the protocol |
| C0 | could skip a range-infeasible control | always reached; an integrity check under exact maxabs | consequence of the mode |
| `single_snapshot` gate | RSE <= 0.50, cos >= 0.90 | RSE <= 0.01, cos >= 0.99 (memorisation standard) | audit item 2; new numbers, not a restoration |
| `fixed_subset4` gate | as v2 | named as a weaker subset fit standard | audit item 2 |
| C4 at subset-mean SSE = 0 | fails | the inequality itself (passes iff AE SSE = 0) | audit item 5: no ratio shortcut |
| Undefined values in JSON | not specified (implementation: strings) | `null` + reason | audit item 5 |
| R4 role | retained, equal standing | prespecified secondary, always scheduled (same prerequisite rules) | audit item 4 |
| R4 construction path | `mean_state(load_client_round(dir)["grads"])` | the restricted 14-file reader | `load_client_round` opens forbidden files |
| Status set | 7 statuses | adds `INCOMPLETE (PUBLICATION)` and `INCOMPLETE (NOT STARTED)`; outcomes only at run closure | audit item 5: separate science from completion |
| Outcome phrases | defined by v1 §1 | bounded meanings stated per standard; v2 qualifier | audit items 2 and 5 |
| Wording control | forbidden-word list (implemented as a filter) | structured fields and review of human-written claims | a filter blocks accurate limitation statements |
| Evaluation overrun | not stated | `INCOMPLETE (RESOURCE STOP)`, no checkpoint | completes the timing rule |
| Timing fields | GPU time only | measured / charged / status / reason | audit item 5 |
| Attempt states and recovery | "no retry after start" | four states, start record before `ae.to(device)`, evidence recovery | audit item 5 |
| Byte budget | 20,000,000 = 6 x 3,000,000 + 2,000,000 reserve | one run-wide ceiling with per-artifact caps and a reservation check | the reserve split was not enforceable as written |
| Summaries / invocations | not bounded (implementation: a new time-stamped summary per entry) | at most 3 invocations, one summary each; explicit closure | bounds the artifact count |
| Run root | `phase5_p5a_seed1/` | `phase5_p5a_v2_seed1/` | v1 and v2 output can never mix |
| Input preflight | part of the run | a separately invocable CPU-only step before training, re-validated by every training invocation | choice 6 |

## 21. Approved human decisions

There are no open design choices. The six choices below were approved on 2026-10-04 (section 18) and are written
into this document. None was chosen after any P5-A result.

| # | Choice | Approved value | Where |
|---|---|---|---|
| 1 | Normalisation | `global_exact_maxabs_train`: one scalar per representation, fitted on the whole training population, `s_r = exact max |x| / 0.95`, applied unconditionally, no clipping. Tradeoff: no Tanh-range confound, but a smaller bulk scale (§5.3), especially for R4 | §5, P5v2-D08 to D14 |
| 2 | Single-snapshot standard | strict memorisation standard: RSE <= 0.01 and cosine >= 0.99, finite outputs and gated values | §10, P5v2-D29 |
| 3 | R4 designation | R2 and R3 primary; R4 a prespecified secondary diagnostic, always scheduled, never skipped on earlier scientific results, subject to the common provenance, prerequisite, resource-stop and no-retry rules | §2, §7, P5v2-D03 |
| 4 | C4 at a zero subset-mean error | the inequality as written (`RSE_AE <= 0.8 x 0` iff `RSE_AE = 0`); no epsilon; no automatic failure | §9.2, §10, P5v2-D32 |
| 5 | Byte policy | one run-wide 20,000,000-decimal-byte ceiling for all files under the run root, per-artifact caps and evidence reservation; the 2 MB reserve / 18 MB subdivision is removed | §13.2, §13.3, P5v2-D44, D45 |
| 6 | Preflight form | a separately invocable CPU-only input preflight, run later after code review and explicit authorisation, before any GPU training; its evidence is re-validated, never a bypass | §5.4.1, P5v2-D11a |
