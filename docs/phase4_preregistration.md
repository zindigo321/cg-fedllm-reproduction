# Phase 4 pre-registration: representation forensics (F3-F6) and the seed-1 baseline (F7-F8)

**Status.** This document was committed **before**:
* any F4 rank statistic, F5 gradient statistic or F6 AutoEncoder result was computed;
* the F7 baseline results were read.

The F0 evidence audit (`representation_forensics.md`) and the F1/F2 validations precede it.

**Labels:**

| Label | Applies to |
|---|---|
| PHASE4-FORENSIC | representation statistics and screens |
| PHASE4-BASELINE | the seed-1 baseline |
| PHASE4-DIAGNOSTIC | validation and timing |

**Never** are any of R2-R4 described as paper-specified. No MMLU, C-Eval or D2 downstream result is used to choose or rank a representation.

## 1. Data

* **R0-R3:** the frozen Phase-3 seed-1 TGAP snapshots of Qwen1.5-1.8B.
  * `federated_pretrain`: 100 client-rounds, 20 rounds, D1, micro-batch 1.
  * Temporal split: rounds 0-15 train (80 snapshots), rounds 16-19 validation (20).
* **R4 (F5):** a bounded new collection.
  * The **same** federated D1 schedule, seeds and semantics (namespace `tgap_fed`, seed 1, micro-batch 1), for **2 rounds** (10 client-rounds).
  * The optimisation is unchanged. As a check, the end states must equal the Phase-3 round-0/1 snapshots bitwise.
  * Temporal split: round 0 train (5), round 1 validation (5). This is a very small screen and is reported as such.

## 2. F4 structural prerequisite for `balanced_effective_delta_r8`

* **Increment:** for every snapshot and module, `dM = s (B_end A_end - B_start A_start)` with s = 16/8 = 2.
* **Per module:** exact rank (singular values > 1e-12 x max), the 16 singular values, and the rank-8 retained energy / truncation error / product cosine.
* **Per snapshot:** retained energy = (sum over modules of rank-8 energy) / (sum over modules of total energy), i.e. energy-weighted.
* **Gate:** if the **median over the 20 validation snapshots** of the per-snapshot retained energy is **< 95 %**, R3 is marked STRUCTURALLY LOSSY before compression. It is still screened, but never called an exact representation.

## 3. F5 gradient summaries (per client-round unless stated)

**Raw collection:** for every optimizer step, the accumulated gradient before clipping, the gradient after clipping (max norm 1.0), the parameter change of the step, and the client's start/end state.

| Summary | Definition |
|---|---|
| `last_step_gradient` | pre-clip gradient of the last step |
| `mean_step_gradient` | mean over steps of the pre-clip gradients **(R4)** |
| `optimizer_step_delta` | the per-step parameter changes; statistics are over steps |
| `local_epoch_delta` | end - start |

**Statistics for each summary:**
* per-element RMS;
* total squared norm;
* A/B energy ratio;
* inter-client cosine (same round, or same step index);
* inter-time cosine (round 0 vs round 1 means; consecutive steps);
* the first-order effective change `s (g_B A + B g_A)` at the start state. For `local_epoch_delta` it is compared with the exact `dM`.

**Paper comparison:** per-element RMS only, with geometry, element count, scaling and gauge stated.

## 4. F6 fixed compressibility screen

**Candidates:**

| ID | Representation | Source |
|---|---|---|
| R0 | `adapter_state` | Phase-3 result reused, frozen |
| R1 | `adapter_delta` | Phase-3 result reused, frozen |
| R2 | `balanced_effective_state` | balanced canonical factors of `s B A` (end state) |
| R3 | `balanced_effective_delta_r8` | balanced factors of the best rank-8 approximation of `dM` |
| R4 | `mean_step_gradient` | mean pre-clip gradient |

R2-R4 use the same Phi layout (`layer_major_qkvo_AtB`), so the AE input is `[1, 2048, 1536]` and the latent `[64, 32, 24]` (exactly 1/64).

**AE protocol:** identical to Phase 3. Fixed ResNet-3; MSE; Adam 2e-4 (0.9/0.999, eps 1e-8, no weight decay); batch 4 (R4: batch 4 over 5 training snapshots); 3000 iterations; evaluation every 50 iterations; best-validation checkpoint; AE init seed 1.

**Scale rule, fixed before training:** compute the training-split p99.9 of |x|.
* If it exceeds 0.95, the representation is also trained with `global_maxabs_train`: one frozen scalar `s_x = p99.9 / 0.95`, encode `x / s_x`, decode `s_x * dec(z)`, 0 uplink bytes. This is PHASE4 diagnostic preprocessing.
* Otherwise only `none` is run.

No other modes or hyper-parameters are tried.

**Metrics** (validation split, pooled, float64). Baselines: zero, train-mean (mean of the training inputs), identity.
* **Representation:** relative squared error, cosine and norm ratio for A, B and all.
* **Effective product.** The target is:
  * R2: `M = B_c A_c` (= `s B A`);
  * R3: the transmitted rank-8 product (also reported against the exact `dM`);
  * R4: the first-order `s (g_B A_s + B_s g_A)`.

  The reconstruction uses the same formula with the decoded factors. Reported: relative squared error, **relative Frobenius error** `||P_hat - P|| / ||P||`, cosine and norm ratio, all from low-rank Gram matrices.
* **Local innovation (R2 only):** `M_hat - M_start` vs `M - M_start`.
* **Aggregate fidelity, in product space only:**
  * sample-weighted sums per time index (R2: state and innovation; R3/R4: the update);
  * R4 also in gradient space (FedSGD-style averaging).

  None of these representations has operational server semantics defined in our pipeline, so these are reported, not gated.
* **Input dependence:** the update-relevant product quantity (R2: product innovation; R3/R4: product) paired with the next client of the same time index.

**Minimum screen gate.** The representation is "compressible enough for further study" only if ALL hold on validation, for the best-val AE of at least one of its run modes:

| # | Criterion |
|---|---|
| S1 | finite outputs |
| S2 | representation cosine (all) >= 0.90 |
| S3 | representation relative squared error (all) <= 0.50 |
| S4 | effective-product cosine >= 0.90 |
| S5 | effective-product relative Frobenius error <= 0.50 |
| S6 | input-dependent: update-relevant product relative squared error, matched pairing < shifted pairing |
| S7 | materially better than the train mean: representation RSE <= 0.8 x the train-mean's **and** update-relevant product RSE <= 0.8 x the train-mean's |

Passing authorises nothing operational: no FAF run, only a recommendation for another reviewer cycle.

## 5. F7 seed-1 baseline and F8 evaluation

**Semantics** (fixed by the F1 outcome; `configs/phase4/tierb_seed1.yaml`): physical micro-batch 1 x 32, with the deviation labelled and quantified.

**Arms:**
* Base (the initial adapter, i.e. the base model);
* LoRA-FT (20 rounds);
* FAF-Identity (20 rounds; must equal LoRA-FT bitwise in every round hash);
* `cent_resource_matched_seed1` (one pooled pass).

Held-out loss is evaluated after every round, one example at a time.

**F8 decision rule:**
* First, time a deterministic sample of the evaluation and project the full cost per model (MMLU test + C-Eval val).
* If <= 45 minutes per model: evaluate Base and LoRA-FT in full.
* Otherwise: evaluate a fixed stratified subset (pre-generated, seeded, recorded) and report the projected full cost.
* C-Eval test is evaluated only if full evaluation is unexpectedly cheap.
* Evaluation never influences any representation decision.
