# Phase 5 preregistration: P5-A AutoEncoder capacity control for R2-R4

**Status: APPROVED** (human review, 2026-10-04, with the six review corrections incorporated; section 17).

This preregistration fixes every protocol, metric, threshold, selection rule and resource ceiling of P5-A. It
was committed before any P5-A training, reconstruction or gate value existed.

**What it authorises.** P5-A engineering implementation only: reconciling the code with this protocol and testing
it on synthetic data (section 13). It does **not** authorise a real P5-A run. A real run additionally needs a
launch record (`reproduction_protocol.md` section 6) that cites this document's commit SHA, and a separate
written run authorisation.

| Field | Value |
|---|---|
| Repository | https://github.com/zindigo321/cg-fedllm-reproduction |
| Parent commit | `afc9e333b5b865b32d47f5634aa7db3ddbfa97ef` (`main`) |
| Commit of this document | not recorded here (a commit cannot contain its own SHA); the launch record cites it |
| Approval | human review, 2026-10-04 |
| Scope | P5-A only (section 2) |
| Result labels | `PHASE5-DIAGNOSTIC`; `DERIVED` for arithmetic on recorded results |

**Terms.** INHERITED: taken unchanged from a committed preregistration, finding or code path that implemented a
preregistered rule (path and commit given). DERIVED: arithmetic on committed records, done before any P5-A result
existed. Every decision is listed with its status in section 16.

## 1. Scientific question

**Uncertainty left by Phase 4.** F6 failed R2-R4 (`phase4_findings.md` section 8, `203be49`). But under the
preregistered `none` scaling, the fixed ResNet-3 never reached the zero-predictor MSE *on its own training data*:
the last training-batch MSE at iteration 3,000 was 4.3x, 11x and 93x the zero predictor for R2, R3 and R4. F6
therefore does not measure 1/64 compressibility (P4-D5, `discrepancies.md`, `203be49`). Phase-4 section 17 item 2
offers a scale-normalised re-screen "with an AE capacity control: fit a single snapshot or a training subset to
below the zero-predictor MSE".

**P5-A question.** For each of R2, R3 and R4 separately: with the AE inputs divided by one frozen global scalar,
can the fixed ResNet-3, trained with the unchanged Phase-3/4 optimiser for 3,000 iterations, reconstruct (a) one
fixed training snapshot and (b) four fixed training snapshots *of its own training set* to the criteria of
section 10?

**What an outcome can establish.**
* `CAPACITY FIT DEMONSTRATED`: under this protocol the P4-D5 scale floor no longer prevents fitting, so a
  re-screen of that representation under the same scaling would be informative. It is **not** evidence that the
  representation is compressible at 1/64, that the AE generalises, that updates survive compression, or that
  CG-FedLLM works. It is a capacity check of the instrument on memorised inputs.
* `CAPACITY FIT NOT DEMONSTRATED`: this fixed AE, scaling, budget and seed did not fit the stated inputs. It is
  **not** evidence that the representation is incompressible.

**Smallest informative design.** One seed, two controls, three representations, no validation data and no
hyperparameter variation: six trainings. A later re-screen is not designed here and needs its own preregistration.

## 2. Scope and exclusions

| Item | Rule |
|---|---|
| Included representations | exactly R2 `balanced_effective_state`, R3 `balanced_effective_delta_r8`, R4 `mean_step_gradient`, constructed as in Phase-4 F6 |
| Excluded representations | R0 `adapter_state`, R1 `adapter_delta`, and any new representation |
| Data | the frozen artifacts of section 3 only; 0 new client rounds; no TGAP collection |
| Held-out data | no Phase-4 validation-split payload (R2/R3 rounds 16-19; R4 round 1) is opened; no D2, MMLU, C-Eval or held-out loss |
| Excluded work | FAF of any kind; P5-B/C/D; any re-screen; any downstream evaluation; operational server semantics; the Phase-4 section 17 items 1, 3, 4 and 5 |
| Compute | the local RTX 4060 Laptop GPU only; no Tier A, no 7B, no external GPU (Phase-4 section 17 item 6, `203be49`) |

**Why R2-R4 and not R0/R1.** Phase-4 section 17 item 2 names R2-R4, and P4-D5 (the scale floor) was diagnosed on
R2-R4. R0/R1 already ran under normalisation in Phase 3 (`global_rms`, `factor_rms`; `phase3_preregistration.md`
section 4, `cabc3d7`; `phase3_findings.md` sections 6-10, `8287811`) and failed differently: they fitted the
shared state but carried no innovation.

**R4 is retained** (review decision Q1). Its five round-0 training snapshots and unusual factor statistics are
known from Phase 4, so they cannot justify removing it; P5-A is a capacity diagnostic and those characteristics
are part of what it tests.

## 3. Frozen inputs and provenance

No artifact is regenerated or modified. Payloads stay outside Git under `CGFED_RUNS` (`D:\cgfed-runs` on the
reference machine). Paths below are relative to the named root.

**Roots and indexes.**

| Root (under `CGFED_RUNS`) | `index.jsonl` SHA-256 | Committed in | Records | Used for |
|---|---|---|---|---|
| `phase3_tierb_qwen15_1p8b_seed1/tgap_federated_state` | `b6834e476f21f3d6b07f6489b5113576e566ab3987a2f4593405664476d07240` | `results/phase4/screen/f6_r2_*.json`, `f6_r3_*.json` (`d3c4eb3`); `results/phase3/ae_viability/*.json` (`fda5382`) | 100; `federated_pretrain`; schema `cg_fedllm.tgap_snapshot/v1` | R2, R3 |
| `phase4_gradient_forensics_seed1/gradient_forensics` | `560cb0f8153fc0a44e2910ae119ebdccdf228271f963969ff4352401833af9b2` | `results/phase4/screen/f6_r4_*.json` (`d3c4eb3`) | 10; `federated_pretrain`; same schema | R4 |

**R2/R3 payloads.** The 80 training records (time indices 0-15) each name an end-state file `file` and a
start-state file `start_file`, with SHA-256 `file_sha256` / `start_file_sha256` and tensor-content hashes
`end_adapter_hash` / `start_adapter_hash`. These are frozen by the hashed index: 80 end-state files under
`snapshots/` and 16 distinct start-state files under `starts/`, none shared with the validation records.

**R4 payloads (primary identity check, review decision Q2).** P5-A may read, under
`phase4_gradient_forensics_seed1/gradient_forensics/gradients/`, exactly these 14 files and no other file of that
directory. These are the files `load_client_round` would open for `meta.json` plus each pre-clip gradient
`stepNNN_grad.safetensors`, for the five time-index-0 records (step counts 2, 3, 1, 1, 2 from each `meta.json`).
P5-A does not open the `*_grad_clipped` or `*_delta` files. Hashes were recorded on 2026-10-04 by reading the
files only.

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

The start- and end-state files of the five R4 records are frozen by the hashed R4 index. P5-A may read them
through `load_states(verify=True)`, as the F6 construction path does, but the R4 AE input depends only on the
gradient files.

**Verification before any training (fail closed).**
1. **Primary, identity.** Both `index.jsonl` files match the hashes above. Every payload read matches its frozen
   SHA-256: R2/R3 through `load_states(verify=True)` (`tgap/snapshots.py`), which also checks tensor-content
   hashes; R4 against the table above. Any other file opened under the R4 `gradients/` directory is a protocol
   violation.
2. **Primary, record keys.** Within each index, `(time_index, client_id)` is unique; the protocol verifies this
   and aborts if it is violated. (Both frozen indexes satisfy it.)
3. **Secondary, semantic integrity.** The recomputed training-population statistics equal the committed F6 values
   `input_stats.train_rms`, `input_stats.train_max_abs` and `scale_rule.train_abs_p99_9.value` (`d3c4eb3`) to a
   relative tolerance of 1e-6. This check guards the construction code path, not file identity; it does not
   replace the hash checks.
4. Any mismatch stops P5-A before training. It is recorded as `INCOMPLETE (PROVENANCE)`, not as a result.

**Pre-registration verification (DERIVED, 2026-10-04, CPU only, no AE built or trained).** Every hash above
matched. Constructing R2, R3 and R4 from the hash-verified training populations with the code of section 4 and
applying the algorithm of section 5 reproduced the committed F6 p99.9 values bitwise; the recomputed
`train_rms` and `train_max_abs` differed from the committed values by relative 0.0.

**Code identity.** F6 ran at `e8100e9`. At the parent commit `afc9e33`, every function on the construction,
normalisation and training path has an AST identical to `e8100e9`: `forensics/representations.py`,
`forensics/gradients.py`, `compression/{gauge,autoencoder,normalization,codecs,metrics,layout}.py`,
`tgap/{train_ae,snapshots}.py`, `utils/seeding.py`. The only intervening change on these files is Ruff formatting
(`79fee1f`).

## 4. Representation construction

All three are INHERITED unchanged from Phase-4 F6. Definitions: `phase4_preregistration.md` section 4
(`5cc2324`). Implementation: `cmd_forensic_screen` in `cli.py`, `forensics/representations.py`,
`forensics/gradients.py` and `compression/gauge.py` (all `3d452cd`). Common to all three: Qwen1.5-1.8B LoRA r 8,
q/k/v/o, 24 layers, s = alpha/r = 16/8 = 2; Phi layout `layer_major_qkvo_AtB`; AE input `[1, 2048, 1536]` fp32;
latent `[64, 32, 24]` (1/64 of the elements).

| ID | Definition per snapshot record | Source | Construction (committed code) |
|---|---|---|---|
| R2 | balanced canonical factors of `M = s B_end A_end` per module: `B_c = Q_B U sqrt(S)`, `A_c = sqrt(S) Vᵀ Q_Aᵀ`; deterministic signs (largest-magnitude entry of each left vector positive); components with σ <= 1e-12 σ_max set to 0 | end state | `balanced_effective_state(end, s)` |
| R3 | balanced factors of the best rank-8 approximation of `dM = s (B_end A_end - B_start A_start)` (rank <= 16; Eckart-Young) | start and end state | `balanced_effective_delta(start, end, s, rank=8)` |
| R4 | mean over the client round's optimizer steps of the pre-clip factor gradients (float64 sum, divided by the step count, stored fp32) | the R4 files of section 3 | `mean_state(load_client_round(dir)["grads"])` |

Then `x = layout.forward(rep, geom)`. No other balancing, truncation or aggregation is applied. R3's rank-8
truncation keeps 98.7 % of the energy (median validation, Phase 4); that is a property of the representation.

**Training populations.** Exactly the F6 training split, from the inherited temporal split
(`split_indices(records, "temporal", 0.2, ...)`, `cbfedab`; `phase4_preregistration.md` section 1):
* R2, R3: time indices 0-15 of the Phase-3 federated set, 80 snapshots;
* R4: time index 0 of the F5 collection, 5 snapshots.

No validation-split payload is opened.

## 5. Normalisation

**Rule.** One frozen scalar `s_R` per representation R, fitted once on the entire training population of section
4 (R2/R3: 80 snapshots; R4: 5), shared by both controls. Encode `x / s_R`; decode `s_R * dec(z)`. The training loss
is the MSE in the normalised space. Every gate metric is computed after de-normalisation, in the original space.

The mode is the existing `global_maxabs_train` (`Normalizer`; `compression/normalization.py`, `3d452cd`). Despite
its historical name it is **not** an exact maximum. It computes:

```text
s_R = Q_0.999(|v|) / 0.95
```

`Q_0.999` is computed by the committed `train_abs_quantile` / `abs_quantiles` (`3d452cd`; `abs_quantiles` from
`cabc3d7`), with these frozen semantics:

| Item | Frozen value |
|---|---|
| Inputs | the Phi tensors `x_i` (fp32, `[1, 2048, 1536]`) of the training population, in the order of section 6 (ascending `(time_index, client_id)`, which is also the frozen index order) |
| Flattening | each `x_i` flattened row-major (`reshape(-1)`); flattened tensors concatenated in that order |
| Element count | `n = sum_i numel(x_i)` |
| Element limit | `M = 2**26 = 67,108,864` |
| Stride | `stride = max(1, ceil(n / M))`; from each flattened tensor keep elements 0, stride, 2 stride, ... (`[::stride]`, applied per tensor before concatenation) |
| Values | cast to float64, absolute values |
| Quantile | `numpy.quantile(v, 0.999)` (NumPy 2.4.6, the pinned environment) with the default method `"linear"`: for m sorted values `v_(0) <= ... <= v_(m-1)`, `h = 0.999 (m - 1)`, `Q = v_(floor h) + (h - floor h) (v_(floor h + 1) - v_(floor h))` |
| Target | 0.95 |
| Clipping | none; values with `|x / s_R| >= 1` are kept and reported (`range_report`, `cabc3d7`) |
| Degenerate scale | `s_R` must be finite and > 0 (existing `Normalizer` check); otherwise stop, `INCOMPLETE (PROVENANCE)` |
| Application | `x / s_R` and `y * s_R` in float32 with `s_R` cast to float32 (existing `_scale_like`, `cabc3d7`) |
| Uplink | 0 bytes (frozen scalar known to both ends) |

**Frozen values.** Reproduced bitwise on 2026-10-04 from the hash-verified inputs of section 3 with the algorithm
above; they equal the committed F6 `train_abs_p99_9` values (`d3c4eb3`) divided by 0.95.

| R | n | stride | `Q_0.999` (exact or strided) | `s_R` |
|---|---|---|---|---|
| R2 | 251,658,240 | 4 | 0.019335589041933415 (strided) | 0.020353251623087806 |
| R3 | 251,658,240 | 4 | 0.008558056039735668 (strided) | 0.009008480041827019 |
| R4 | 15,728,640 | 1 | 0.009295389744453092 (exact) | 0.009784620783634835 |

A run applies the frozen `s_R` above. It also recomputes `Q_0.999 / 0.95` with this algorithm and stops
(`INCOMPLETE (PROVENANCE)`) unless the recomputed value equals the frozen `s_R` to relative 1e-6; bitwise equality
is expected.

**Range.** Normalised-range statistics are reporting and integrity information only. They are **not** a pass
criterion.

**Deliberate Phase-5 choice.** P5-A applies p99.9 -> 0.95 scaling **unconditionally**, as motivated by Phase-4
section 17 item 2 and P4-D5. This is not the Phase-4 rule, which ran `global_maxabs_train` only if the training
p99.9 already exceeded 0.95 (`phase4_preregistration.md` section 4, `5cc2324`; deviation 38) and therefore never
ran it. The F6 verdicts stand; P5-A is a new experiment.

**Why p99.9 and not unit RMS or exact max** (DERIVED from committed F6 statistics, `d3c4eb3`). Unit RMS would
leave normalised maxima of 8.1 / 7.3 / 62.9 (R2 / R3 / R4), well outside Tanh's range; Phase 3 measured 42-59 % of
A values outside the range under RMS scaling (`phase3_findings.md` section 6, `8287811`). An exact maximum would
leave heavy-tailed R4 at RMS 0.015. Under p99.9 -> 0.95 the normalised RMS is 0.199 / 0.227 / 0.105 and the
normalised maxima are 1.61 / 1.66 / 6.57.

**C0, Tanh-range feasibility precondition** (input-only, before training, result-independent). For each control,
compute the inherited `tanh_range_ceiling` predictor `denorm(clamp(norm(x), -1, 1))` (`cabc3d7`) on the control's
selected snapshots and apply the control's criteria C2, C3 and (for `fixed_subset4`) C4 to it (section 10). If any
fails, no Tanh-headed decoder can pass that control under this scaling; the control is recorded
`NOT INFORMATIVE (TANH RANGE)` and is not trained. C0 is the only permitted pre-training skip besides provenance
failures.

## 6. Capacity controls

Two controls per representation. Both train and evaluate on the same snapshots by design: they test whether the
instrument can fit, not whether it generalises.

**Ordering.** `E` = the training records of section 4 sorted by `(time_index, client_id)` ascending. The key is
unique in both frozen indexes; the protocol verifies uniqueness (section 3) and aborts if it is violated, so no
tie-break is needed. `N = |E|`; positions are 0-based; `round_half_up(v) = floor(v + 0.5)`.

| Control | Snapshots k | Rule |
|---|---|---|
| `single_snapshot` | 1 | `E[floor((N - 1) / 2)]` (median position) |
| `fixed_subset4` | 4 | `E[round_half_up(j (N - 1) / 3)]` for j = 0, 1, 2, 3 (evenly spaced, both ends included) |

The rule uses record metadata only, never tensor values or AE outputs. Selected records (DERIVED from the frozen
indexes, before any P5-A result):

| R | N | `single_snapshot` (t, client) | `fixed_subset4` positions -> (t, client) |
|---|---|---|---|
| R2, R3 | 80 | position 39 -> (7, 91) | 0, 26, 53, 79 -> (0, 2), (5, 32), (10, 43), (15, 84) |
| R4 | 5 | position 2 -> (0, 55) | 0, 1, 3, 4 -> (0, 2), (0, 26), (0, 75), (0, 86) |

**Known properties of the selected records**, disclosed from committed findings (`phase4_findings.md` sections 6-7,
`203be49`):
* R2/R3 (0, 2) is a round-0 record: its start state has B = 0, so its R3 increment has rank <= 8.
* R4 (0, 55) and (0, 75) each took a single optimizer step at B = 0, so their R4 A part is exactly zero: half of the
  Phi columns of the R4 `single_snapshot` input are zero.

**Execution of controls** (review decision Q3). Both controls run for every representation unless C0 or a
provenance failure prevents that control. A `single_snapshot` outcome never causes `fixed_subset4` to be skipped or
changed, and no outcome prunes or reorders the six-training plan. Execution order: R2, R3, R4; within each,
`single_snapshot` then `fixed_subset4`.

**Batching** (INHERITED loop semantics, `train_autoencoder`, `cbfedab`; `phase3_preregistration.md` section 4).
Batch size `bs = min(4, k)`. Each iteration takes the first `bs` entries of a permutation of the k snapshots, drawn
from `numpy_rng(1, "ae_batches")`, and draws a new permutation when fewer than `bs` remain.
* `single_snapshot`: k = 1, bs = 1; every batch is the one snapshot.
* `fixed_subset4`: k = 4, bs = 4; every batch holds all four snapshots in permuted order.
* No snapshot is duplicated to fill a batch.

## 7. AE architecture and optimisation

INHERITED from `phase3_preregistration.md` section 4 (`cabc3d7`) and `phase4_preregistration.md` section 4
(`5cc2324`); implementation `compression/autoencoder.py` (`cbfedab`), `configs/phase3/tierb_seed1.yaml` (`cd4c5c4`).

| Item | Value |
|---|---|
| Architecture | reconstructed ResNet-3: 3x3 stem 1 -> 1 channel (no bias) + BN + ReLU; 6 stride-2 conv stages 1 -> 2 -> 4 -> 8 -> 16 -> 32 -> 64 (no bias) + BN + ReLU; 3 + 3 residual blocks (64); 6 ConvT stages; reflection pad + 7x7 conv (bias) + Tanh; 493,483 parameters |
| Input / latent | `[1, 2048, 1536]` fp32 -> `[64, 32, 24]`, 1/64 of the elements |
| Loss | MSE (`F.mse_loss`, mean over elements) in the normalised space |
| Optimiser | Adam, lr 2e-4, betas (0.9, 0.999), eps 1e-8, weight decay 0; no scheduler, no gradient clipping |
| Iterations | exactly 3,000 optimizer iterations; no early stopping, no extension |
| AE seed | {1} only |
| Seed derivation | init: `torch.manual_seed(derive_seed(1, "ae_init"))` inside `torch.random.fork_rng`; data order: `numpy_rng(1, "ae_batches")` (`utils/seeding.py`, `4cca6ae`) |
| Determinism | `configure_determinism(True)`: deterministic algorithms, cuDNN deterministic, benchmark off, TF32 off, `CUBLAS_WORKSPACE_CONFIG=:4096:8` (as F6, `run.deterministic: true`) |
| Device | the local CUDA GPU; allocator cap free VRAM - 256 MiB (deviation 32) |
| Non-finite loss | stop the training; outcome `INCOMPLETE (NON-FINITE LOSS)`; no rerun |

## 8. Checkpoint and evaluation semantics

**Final checkpoint only: an intentional Phase-5 departure.** Phase 3/4 gated the best-*validation* checkpoint
(`phase3_preregistration.md` section 4, `cabc3d7`; `phase4_preregistration.md` section 4, `5cc2324`). P5-A does not
inherit that rule. P5-A is a fixed-budget training-set capacity diagnostic, opens no validation split, and has a
budget of exactly 3,000 iterations; choosing an earlier checkpoint by observed loss would add an unnecessary
result-dependent choice. Only the iteration-3,000 checkpoint is saved, evaluated and gated.

**BatchNorm.** Gate metrics use eval mode with BatchNorm running statistics, as in Phase 3/4 (`_eval_set`,
`cbfedab`). The train-mode batch MSE of the training curve is reported, never gated.

**Procedure** (INHERITED code path `AutoEncoderCodec`, `fd71dcc`). After iteration 3,000: `ae.eval()`, no
gradients. For each unique selected snapshot `x` (fp32): `x_hat = s_R * dec(enc(x / s_R))` on the GPU in fp32; move
both to the CPU; compute section 9 in float64. Evaluated on the same snapshots and reported, never gated as AE
results: the zero predictor, the `tanh_range_ceiling` predictor (C0), and for `fixed_subset4` the subset-mean
predictor `m = (1/4) sum_j x_j` (computed in float64).

## 9. Metrics

For snapshot i with truth `x_i` and prediction `p_i` (any predictor), both in the original space, all n = 3,145,728
elements, float64:

| Quantity | Per snapshot i | Pooled over a set S of snapshots |
|---|---|---|
| sums | `sse_i = sum (p_i - x_i)²`, `sig_i = sum x_i²`, `hat_i = sum p_i²`, `dot_i = sum p_i x_i` | `SSE = sum_{i in S} sse_i`, `SIG`, `HAT`, `DOT` likewise |
| MSE | `sse_i / n` | `SSE / (|S| n)` |
| zero-predictor MSE | `sig_i / n` | `SIG / (|S| n)` |
| relative squared error (RSE) = MSE ratio to the zero predictor | `sse_i / sig_i` | `SSE / SIG` |
| cosine | `dot_i / sqrt(sig_i hat_i)` | `DOT / sqrt(SIG HAT)` |
| norm ratio | `sqrt(hat_i / sig_i)` | `sqrt(HAT / SIG)` |

**Undefined values.** RSE is undefined if `sig = 0`; cosine is undefined if `sig = 0` or `hat = 0`. An undefined or
non-finite gated value fails its criterion. Each selected snapshot must have `sig_i > 0`; otherwise the control is
`INCOMPLETE (PROVENANCE)`. These definitions match `phase3_preregistration.md` section 5 (`cabc3d7`) and
`reconstruction_metrics` (`fd71dcc`).

**Reported, never gated:** per-snapshot and pooled values for AE, zero, `tanh_range_ceiling` and subset mean;
normalised-range statistics (`range_report`); the training curve. Product-space, innovation, aggregate and
shift-pairing metrics (Phase-4 S4-S6) are not computed in P5-A.

## 10. Pass/fail decision rule

**Characterisation.** P5-A is a tiny-training-set capacity control, not the Phase-4 validation screen. To avoid
inventing new post-hoc numbers it reuses the numerical thresholds and the factor of Phase-4 S2 (cosine >= 0.90), S3
(RSE <= 0.50) and S7 (0.8 x an input-independent mean) (`phase4_preregistration.md` section 4, `5cc2324`;
`forensics/screen.py` `GATE`, `3d452cd`). It applies them deliberately to the preselected training snapshots of
each control. Differences from Phase 4:
* Phase-4 S2/S3/S7 were pooled over a validation split; P5-A C2/C3 apply to every selected training snapshot
  individually, which may be stricter;
* P5-A does not inherit or imply S4, S5 or S6;
* a P5-A pass does not establish representation compressibility (section 1).

For `single_snapshot`, C2 (RSE <= 0.50) implies a result below the zero predictor's RSE of 1.0, the condition
named in Phase-4 section 17 item 2. For `fixed_subset4`, C4 is the preregistered input-dependence check: it rules
out passing C2/C3 with a near-constant output on similar snapshots (in F6 the R4 train mean reached validation RSE
0.50 and cosine 0.82, `phase4_findings.md` section 8).

| # | Criterion | Calculation | Applies to |
|---|---|---|---|
| C0 | feasibility precondition | `tanh_range_ceiling` satisfies C2 and C3 for every selected snapshot, and C4 for `fixed_subset4` | both, before training |
| C1 | finite | every element of every AE reconstruction is finite, and every gated value of C2-C4 is defined and finite | both |
| C2 | RSE | `RSE_i <= 0.50` for **every** selected snapshot i (per snapshot, section 9) | both |
| C3 | cosine | `cos_i >= 0.90` for **every** selected snapshot i (per snapshot) | both |
| C4 | better than the subset mean | pooled AE RSE over the 4 snapshots `<= 0.8 x` pooled subset-mean RSE over the same 4; fails if the subset-mean RSE is 0 | `fixed_subset4` |

All boundaries are inclusive. Every required criterion must pass.

```text
control_outcome(R, c):
    if any provenance or input check fails:      return INCOMPLETE (PROVENANCE)        # not trained
    if not C0(R, c):                             return NOT INFORMATIVE (TANH RANGE)   # not trained
    train exactly 3,000 iterations (section 7)
        non-finite loss:                         return INCOMPLETE (NON-FINITE LOSS)
        resource stop (section 12):              return INCOMPLETE (RESOURCE STOP)
        crash or interruption:                   return INCOMPLETE (INTERRUPTED)
    evaluate the iteration-3,000 checkpoint in eval mode (section 8)
    required = C1, C2, C3 (+ C4 if c == fixed_subset4)
    return FIT PASS if all required hold else FIT FAIL
```

## 11. Aggregation and outcome wording

* **Snapshots:** AND over the selected snapshots for C2 and C3; no mean or median of per-snapshot values is gated.
* **Criteria:** AND over the required criteria.
* **Seeds:** one seed; no aggregation over seeds.
* **Controls -> representation**, for each representation separately:

| `single_snapshot` | `fixed_subset4` | Representation outcome |
|---|---|---|
| FIT PASS | FIT PASS | `CAPACITY FIT DEMONSTRATED` |
| FIT FAIL | any | `CAPACITY FIT NOT DEMONSTRATED` |
| any | FIT FAIL | `CAPACITY FIT NOT DEMONSTRATED` |
| every other combination (at least one `NOT INFORMATIVE` or `INCOMPLETE`, no FIT FAIL) | | `INCONCLUSIVE` |

* **Representations:** never aggregated; there is no overall P5-A pass.
* **Required qualifier** in every report of an outcome: "under the P5-A protocol (fixed ResNet-3,
  `global_maxabs_train` p99.9 -> 0.95, Adam 2e-4, 3,000 iterations, final checkpoint, AE seed 1, training snapshots
  only)".
* **Forbidden** in any P5-A report: "compressible", "incompressible", "viable", "validated", "reproduces", and
  `AE TRAINING-FIT FAILURE`.
* **What an outcome authorises.** `CAPACITY FIT DEMONSTRATED` authorises only drafting a separate scale-normalised
  re-screen preregistration for that representation. It does not authorise that screen, FAF or any downstream
  experiment. Other outcomes authorise nothing; the reviewer decides between a new, separately preregistered
  experiment and stopping representation work.

## 12. Resource ceilings and failure accounting

**Basis** (DERIVED from committed records). One F6 training of 3,000 iterations at batch 4 took 464-734 s of
training loop and 672-900 s of stage time, including evaluation of 100 or 10 snapshots (`results/phase4/screen/*.json`,
`d3c4eb3`; `phase4_findings.md` section 13). One fp32 AE state is 1,981,796 bytes; one Phi input is 12,582,912
bytes. The GPU runs under a software power cap and thermal slowdown (P3-D2), so timings vary.

| Ceiling | Value | Purpose |
|---|---|---|
| Trainings | 6 = 3 representations x 2 controls x 1 seed | forbids reruns, extra seeds and extra modes |
| Per-training GPU wall time | 1,800 s hard limit (2 x the longest F6 stage time, 899.6 s) | detects hangs, WDDM spill (P2-D5) or abnormal throttling |
| Aggregate GPU wall time | 10,800 s (6 x 1,800 s) | bounds the whole phase |
| Retained bytes | 20,000,000 = 6 x 3,000,000 per training + 2,000,000 run-level | admits checkpoints and records; excludes tensor dumps (one Phi tensor is 12.6 MB) |
| New client rounds | 0 | no data collection; any call into FL, TGAP collection or FAF is a protocol violation |

**Accounting units.**
* **Training:** one AE moved to the GPU for one P5-A control. Every started training counts, whatever its outcome
  (pass, fail, non-finite loss, resource stop, crash, interruption). Controls stopped by a provenance failure or by
  C0 are not trainings.
* **GPU wall time of a training:** the monotonic clock from `ae.to(device)` (after `torch.cuda.synchronize`) to the
  end of the section-8 evaluation (after `torch.cuda.synchronize`). Checked at every iteration and before
  evaluation. Failed and interrupted trainings count in full.
* **Setup time** (CPU: hash checks, representation construction, scale recomputation, C0): recorded as
  `setup_wall_s`, not ceilinged.
* **Retained bytes:** the sum of the exact sizes of all *finalized* files under the P5-A run root
  `<CGFED_RUNS>/phase5_p5a_seed1/`, counted from the first file P5-A writes.

**Artifact transactions.**
* A file becomes a finalized retained artifact only when its publication completes. Successfully finalized P5-A
  artifacts are never deleted, including to recover budget.
* Temporary files created during an atomic-write transaction may be removed after a failed or incomplete write. An
  incomplete temporary file is not a retained artifact and is not counted. A temporary file that survives the run
  is counted.
* A pre-existing file is never overwritten or deleted. An existing output path stops the training before it
  starts (`INCOMPLETE (PROVENANCE)`).
* Before publication the implementation determines the exact serialized byte count of each output where
  practicable and checks it against the remaining budget. Output that would exceed the ceiling is not published as
  a finalized artifact; the event is an engineering failure, P5-A stops, that control receives no outcome other than
  `INCOMPLETE (RESOURCE STOP)`, and a failure record with the byte counts is published within the 2,000,000-byte
  run-level reserve.

**Failed-training outputs.** A training that ends `INCOMPLETE` publishes its `p5a_record.json` (status, partial
training curve, resource ledger) and no checkpoint. Its bytes, time and training count stay charged.

**Hard stops.**
* A training starts only if `trainings < 6` and `aggregate GPU time + 1,800 <= 10,800`. Otherwise P5-A stops, and
  every remaining control is `INCOMPLETE (RESOURCE STOP)`.
* A training reaching 1,800 s is stopped at once: `INCOMPLETE (RESOURCE STOP)`; it counts.

**Retries.**
* Before a control's training starts (provenance, path, environment or launch error): the cause may be fixed and
  the step repeated, provided the fix changes no value of this document. Every attempt is recorded.
* After a training has started: **no retry**, whatever the outcome. A rerun would require a written reviewer
  decision recorded as a deviation and an explicit amendment of the training ceiling.

## 13. Implementation validation allowed before a real run

Synthetic data only:
* CPU unit tests with synthetic tensors and synthetic `index.jsonl` records (tiny geometries; few iterations through
  an injected budget in tests only);
* selection tests on synthetic metadata, including the exact positions of section 6 for N = 80 and N = 5, and the
  uniqueness abort;
* metric and gate tests on constructed tensors: exact RSE and cosine values, zero signal, zero reconstruction, NaN,
  inclusive boundaries, C4 with a zero subset-mean RSE, C0;
* normalisation tests of the section-5 algorithm (exact and strided cases, frozen-scale comparison, degenerate
  scale) on synthetic tensors;
* resource-ledger, timeout, retained-byte and artifact-transaction tests with an injected clock and synthetic sizes;
* one GPU smoke test of the training loop on random synthetic tensors of the real shape, at most 50 iterations,
  outputs in a temporary directory that is deleted, no result recorded or reported.

Reading the frozen `index.jsonl` files and hashing frozen payload files without constructing AE inputs for
training is allowed. Not allowed: any AE forward or backward pass on a real Phase-3/4 tensor, any timing run on
real inputs.

## 14. Forbidden without separate written authorisation

* any real P5-A training or evaluation (this document does not authorise one);
* changing any value of this document after any P5-A output (including a training curve) exists;
* changing the representation scope, training populations, selection rule, normalisation or thresholds;
* any hyperparameter, architecture, iteration, learning-rate, seed or normalisation-mode search;
* opening validation-split payloads, D2, held-out or benchmark data in P5-A;
* FAF, P5-B, P5-C, P5-D, any re-screen, any downstream evaluation;
* new client rounds or TGAP collection; Tier A, 7B or external GPUs;
* reporting any outcome with the forbidden words of section 11.

## 15. Result label, retained artifacts and provenance

**Label.** `PHASE5-DIAGNOSTIC` for every P5-A record, plus `DERIVED` for arithmetic on recorded results. It
follows the committed schema convention (`ResultLabel`, `config.py`; Phase-3 labels deviation 34, `cabc3d7`;
Phase-4 labels deviation 42, `203be49`). Adding it to the schema, with a deviation row, is part of the P5-A
implementation change.

**Per training** (`<CGFED_RUNS>/phase5_p5a_seed1/<representation>_<control>/`, outside Git):

| Artifact | Content |
|---|---|
| `config.resolved.yaml`, `config.sha256.json`, `run_metadata.json` | as every run (`provenance.md` section 7): commit, branch, dirty state and files, packages, CUDA/GPU facts including `nvidia-smi`, seeds |
| `p5a_record.json` | label; this preregistration's commit SHA (from the launch record); execution commit and tree state; representation; control; the input `index.jsonl` SHA-256; every selected record with its sorted position, `(time_index, client_id)`, `file_sha256`, `start_file_sha256`, adapter hashes, and for R4 the SHA-256 of every gradient file read; SHA-256 of every Phi input tensor; normalisation (`Normalizer.to_dict()`, recomputed vs frozen `s_R`); secondary statistics check; C0 values; seed and derived seed values; training curve (iterations 1, 50, ..., 3,000: batch MSE in the normalised space); final metrics per snapshot and pooled for the AE, zero, `tanh_range_ceiling` and subset-mean predictors; C1-C4 values and booleans; outcome; resource ledger (setup time, GPU time, trainings used, retained bytes); exact command |
| `autoencoder_p5a_final.safetensors` | the iteration-3,000 AE state with the record metadata (about 2 MB); trainings that end `INCOMPLETE` publish none |

Not retained: reconstructions, normalised tensors, latents, optimiser state, copies of inputs.

**Committed evidence after a run** (separate reviewed change): `results/phase5/p5a/` (the six `p5a_record.json`
files and one summary), a findings document and deviation rows. Negative, inconclusive and incomplete outcomes are
committed with the same weight as passes.

**Launch record** (`reproduction_protocol.md` section 6, `ca0a17f`), separate and reviewed before the run: this
preregistration's commit SHA, configuration path and overrides, full execution commit SHA, environment, exact
command.

## 16. Decision table

| ID | Decision | Approved value | Source / rationale | Status |
|---|---|---|---|---|
| P5-D00 | Go / no-go | P5-A capacity diagnostics authorised as a protocol (implementation only; no run) | `phase4_findings.md` §17 item 2 (`203be49`) | APPROVED |
| P5-D01 | Representations | exactly R2, R3, R4; R0/R1 excluded; R4 retained (Q1) | §17 item 2; P4-D5 (`203be49`); Phase 3 (`cabc3d7`, `8287811`) | APPROVED |
| P5-D02 | Data | frozen artifacts of §3; 0 new client rounds | index hashes in `d3c4eb3` | APPROVED |
| P5-D03 | Excluded work | no FAF, P5-B/C/D, re-screen, downstream evaluation, server semantics | smallest informative design | APPROVED |
| P5-D04 | Compute | local GPU only; no Tier A / 7B / external GPU | §17 item 6 (`203be49`) | APPROVED |
| P5-D05 | Training populations | R2/R3 t 0-15 (80); R4 t 0 (5); no validation payload opened | temporal split (`cabc3d7`, `5cc2324`; `split_indices` `cbfedab`) | APPROVED |
| P5-D06 | Input identity and integrity | primary: index hashes, R2/R3 payload hashes via `load_states(verify=True)`, the 14 R4 file hashes of §3, key uniqueness; secondary: F6 statistics to rel. 1e-6; any mismatch stops | review decision Q2 | APPROVED |
| P5-D07 | Normalisation statistic | `Q_0.999(|v|)` by the §5 algorithm (concatenation order, `2**26` limit, per-tensor stride, float64, NumPy `"linear"`) | `train_abs_quantile` (`3d452cd`), `abs_quantiles` (`cabc3d7`) | APPROVED |
| P5-D08 | Target | 0.95 | Phase-4 target (`5cc2324`) | APPROVED |
| P5-D09 | Mode | existing `global_maxabs_train`, applied unconditionally (a Phase-5 choice, §5) | `normalization.py` (`3d452cd`); §17 item 2 | APPROVED |
| P5-D10 | Fitting population | the whole training population; one scalar per representation, shared by both controls | §17 "frozen global scalar" | APPROVED |
| P5-D11 | Frozen scales | R2 0.020353251623087806; R3 0.009008480041827019; R4 0.009784620783634835; the frozen value is applied; recomputation must match to rel. 1e-6 | reproduced bitwise 2026-10-04 from hash-verified inputs; equal to `d3c4eb3` / 0.95 | APPROVED |
| P5-D12 | Clipping / inversion / range / degenerate scale | no clipping; metrics after de-normalisation; range reported only; `s_R` finite > 0 or stop | `phase3_preregistration.md` §4 (`cabc3d7`) | APPROVED |
| P5-D13 | C0 precondition | `tanh_range_ceiling` must meet the control's C2/C3 (and C4); else `NOT INFORMATIVE (TANH RANGE)`, not trained | predictor from `cabc3d7` | APPROVED |
| P5-D14 | `single_snapshot` selection | `E[floor((N-1)/2)]`, E sorted by unique `(time_index, client_id)`: R2/R3 (7, 91); R4 (0, 55) | metadata-only rule | APPROVED |
| P5-D15 | `fixed_subset4` selection | `E[round_half_up(j(N-1)/3)]`, j = 0..3: R2/R3 (0,2) (5,32) (10,43) (15,84); R4 (0,2) (0,26) (0,75) (0,86) | metadata-only rule | APPROVED |
| P5-D16 | Batching | `bs = min(4, k)`, permutations from `numpy_rng(1, "ae_batches")`; no duplication | `train_autoencoder` (`cbfedab`) | APPROVED |
| P5-D17 | Architecture | reconstructed ResNet-3, unchanged | `cbfedab`; `cabc3d7`, `5cc2324` | APPROVED |
| P5-D18 | Optimiser and loss | MSE in normalised space; Adam 2e-4, (0.9, 0.999), eps 1e-8, wd 0 | `cabc3d7`, `5cc2324` | APPROVED |
| P5-D19 | Iterations / stopping | exactly 3,000; no early stop; no extension | `cabc3d7`, `5cc2324` | APPROVED |
| P5-D20 | Seeds | AE seed {1} only | Phase 3/4 AE seed 1 (`cabc3d7`, `5cc2324`) | APPROVED |
| P5-D21 | Determinism and non-finite loss | `derive_seed(1, "ae_init")`, `numpy_rng(1, "ae_batches")`, `configure_determinism(True)`; non-finite loss -> `INCOMPLETE (NON-FINITE LOSS)` | `4cca6ae`; F6 config | APPROVED |
| P5-D22 | Checkpoint | iteration 3,000 only; intentional departure from Phase-3/4 best-validation (§8) | no validation data; fixed budget | APPROVED |
| P5-D23 | BatchNorm at evaluation | eval mode, running statistics | `cbfedab`; Phase 3 §4 | APPROVED |
| P5-D24 | Metrics | §9 per-snapshot and pooled formulas, float64, original space; undefined fails | `phase3_preregistration.md` §5 (`cabc3d7`); `fd71dcc` | APPROVED |
| P5-D25 | C2 | `RSE_i <= 0.50` for every selected snapshot | Phase-4 S3 number (`5cc2324`), applied per training snapshot | APPROVED |
| P5-D26 | C3 | `cos_i >= 0.90` for every selected snapshot | Phase-4 S2 number (`5cc2324`), applied per training snapshot | APPROVED |
| P5-D27 | C4 | `fixed_subset4`: pooled AE RSE <= 0.8 x pooled subset-mean RSE | Phase-4 S7 factor (`5cc2324`) | APPROVED |
| P5-D28 | C1 and aggregation | finite; AND over snapshots and criteria; inclusive bounds; §11 table; no cross-representation result | Phase-4 "ALL hold" | APPROVED |
| P5-D29 | Result label | `PHASE5-DIAGNOSTIC` (+ `DERIVED`) | label convention (deviations 34, 42) | APPROVED |
| P5-D30 | Outcome wording | `CAPACITY FIT DEMONSTRATED` / `CAPACITY FIT NOT DEMONSTRATED` / `INCONCLUSIVE` + qualifier; forbidden words | avoids overclaiming | APPROVED |
| P5-D31 | Training ceiling | 6; every started training counts | 3 x 2 x 1 | APPROVED |
| P5-D32 | Per-training GPU time | 1,800 s hard limit | 2 x 899.6 s (`d3c4eb3`) | APPROVED |
| P5-D33 | Aggregate GPU time | 10,800 s; a training starts only if 1,800 s still fit | 6 x 1,800 s | APPROVED |
| P5-D34 | Retained bytes | 20,000,000 finalized bytes under the run root | AE state 1,981,796 B; excludes tensor dumps | APPROVED |
| P5-D35 | Accounting and artifact transactions | §12 units; finalized artifacts never deleted; failed atomic-write temporaries may be removed; no overwrite; preflight exact byte counts | review correction 4 | APPROVED |
| P5-D36 | Retries and control execution | retry only before a control's training starts, without protocol change; none after; both controls always run unless C0 or provenance prevents (Q3) | review decision Q3 | APPROVED |
| P5-D37 | Authorisation by outcome | `CAPACITY FIT DEMONSTRATED`: may draft a separate re-screen preregistration only; otherwise nothing | §17 item 2 | APPROVED |

## 17. Approval record

| Item | Decision |
|---|---|
| Disposition | approved, conditional on six corrections, all incorporated in this document |
| Q1, R4 | retained; P5-A covers exactly R2, R3, R4 |
| Q2, R4 payload provenance | the 14 R4 file hashes (section 3) are the primary identity check; the statistics check is secondary |
| Q3, control execution | both controls always run unless C0 or provenance prevents one; no result-dependent pruning |
| Correction 1 | exact normalisation algorithm specified; frozen scales reproduced bitwise; unconditional scaling stated as a Phase-5 choice |
| Correction 2 | capacity gate characterised as a training-set control that reuses Phase-4 numbers, not the validation screen |
| Correction 3 | final checkpoint stated as an intentional departure from Phase 3/4 |
| Correction 4 | artifact-transaction and retention semantics specified (section 12) |
| Correction 5 | record ordering verified unique, abort on violation |
| Correction 6 | this document records its parent commit, not its own SHA |

**Before a real P5-A run, all of the following are required:**
1. the P5-A code reconciled with this document and reviewed, with tests on synthetic data (section 13);
2. a launch record (`reproduction_protocol.md` section 6) that cites this document's commit SHA;
3. a separate written run authorisation.

## 18. Intentional departures from earlier phases

| Topic | Phase 3/4 | P5-A |
|---|---|---|
| Input scaling | Phase 4: `global_maxabs_train` only if the training p99.9 exceeded 0.95 (never triggered) | unconditional p99.9 -> 0.95 (section 5) |
| Checkpoint | best-validation | iteration 3,000 only (section 8) |
| Gate | Phase-4 S1-S7, pooled over the validation split | C0-C4 on the selected training snapshots, C2/C3 per snapshot; reuses the S2/S3/S7 numbers only (section 10) |
| Data seen | train and validation splits | training populations only |
| Batch | 4 | `min(4, k)` (identical rule; k = 1 for `single_snapshot`) |

New deviation rows for these departures are added with the implementation change, before any run.
