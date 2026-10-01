# Phase 4 findings: representation forensics, micro-batch fidelity and the seed-1 Tier-B baseline

Qwen1.5-1.8B (bf16, gradient checkpointing), scientific seed 1, RTX 4060 Laptop 8 GB. The protocol for F3-F8 was
pre-registered in `phase4_preregistration.md` (`5cc2324`). That commit precedes every F4/F5/F6 statistic, and it
precedes the first completed baseline round (see section 15). The F0 audit is in `representation_forensics.md`
(`62dac71`), committed before any Phase-4 experiment.

**Labels:**

| Label | Applies to |
|---|---|
| PHASE4-FORENSIC | representation statistics and screens |
| PHASE4-BASELINE | the baseline |
| PHASE4-DIAGNOSTIC | validation and timing |
| DERIVED | arithmetic on recorded results |

No MMLU, C-Eval or D2 result was used to choose, rank or screen a representation. Phase-3 results are frozen and
reused (R0/R1), never rewritten.

## 1. Verdict

| # | Exit criterion | Status |
|---|---|---|
| 1 | paper forensics documented | ✓ `representation_forensics.md`, searches recorded |
| 2 | micro-batch semantics emulated or quantified as unresolved | ✓ emulated; GPU criterion not met, so quantified and the labelled fallback used |
| 3 | gauge diagnostics pass | ✓ |
| 4 | canonical effective state implemented and tested | ✓ |
| 5 | effective-delta rank / truncation measured | ✓ |
| 6 | bounded gradient forensics completed | ✓ |
| 7 | fixed screen completed | ✓ (every candidate FAILs) |
| 8 | exactly one seed-1 baseline completed | ✓ |
| 9 | Identity regression passes | ✓ bitwise, 13/13 |
| 10 | evaluation timing measured | ✓ (then full evaluation) |
| 11 | tests and CI pass | ✓ CPU 142 passed; GPU and model regression tests passed locally |
| 12 | branch pushed | ✓ |

* **Paper forensics (F0).**
  * No official code, supplement or author repository was found; the searches are recorded.
  * The PAPER-LITERAL encoded object is the factor pair `[A_i, B_i]`. Whether it is a state, an increment or a gradient remains **UNKNOWN**.
  * New paper inconsistencies: DR-22..28.
* **Micro-batch fidelity (F1).**
  * The virtual paper micro-batch (logical 16 x 2) is exact in code: chunk 16 is bitwise identical to physical 16, and CPU chunks 1/2/3/16 pass.
  * It misses the pre-registered GPU adapter criterion: 5.0e-5 > 1e-5, from Adam first-step sign flips at |g| <= 1.9e-7.
  * The baseline therefore uses micro-batch 1 x 32, a **PROMINENT deviation**: gradient cosine 0.866 with the paper's logical micro-batch on a fixed Qwen batch.
  * Clean-commit reruns reproduce both F1 records bitwise.
* **Gauge (F2-F4).**
  * Raw LoRA factor norms are gauge-dependent. On real data the raw factor energy goes from 263 to 310 at an unchanged spectrum.
  * The balanced canonical representation is implemented and tested (`‖A_c‖² + ‖B_c‖² = 2‖M‖_*`).
  * The rank-8 effective increment retains 98.7 % of its energy, so it is not structurally lossy.
* **Gradients (F5).**
  * The collection is bitwise unchanged (10/10 end states).
  * Raw factor gradients have 0.69-0.72x the paper's per-element RMS; 1-3-step increments 0.08-0.15x.
  * 90 % of steps were clipped.
* **Screen (F6).**
  * **No representation passes** (R0-R4).
  * For R2-R4 the fixed AE never reached the zero-output MSE on its own training data under the pre-registered `none` mode, so F6 does not measure their intrinsic 1/64 compressibility (P4-D5).
  * Nothing is forwarded; no operational AE/FAF run was started.
* **Baseline (F7).**
  * LoRA-FT held-out loss: 2.370 → 1.766 in 20 rounds.
  * FAF-Identity is bitwise identical (13/13 checks, 0 mismatches).
  * The resource-matched centralized reference reaches 1.756.
  * Logical communication: 2,516,582,400 B two-way over 20 rounds.
* **Evaluation (F8).**
  * Timed first: projections of 23.2 and 28.7 min per model, so the full evaluation ran.
  * MMLU test: Base 45.25 %, LoRA-FT 45.71 % (paired exact McNemar p = 0.010).
  * C-Eval val average: 59.08 % and 58.56 % (question-level accuracy identical, p = 1.0).
  * One seed; not used for any representation decision.

## 2. Paper representation forensics (F0)

Full audit: `docs/representation_forensics.md`.

**Search record.**
* Searched: GitHub code/repository search, web search, the arXiv API (all versions), Hugging Face, and the authors' pages.
* Result: no official code, supplementary material or author repository was found.
* One later paper by the same authors, DR-Encoder (arXiv 2412.17053, AAAI-25), cites CG-FedLLM as "FedCG".
  * It writes the encoded object as `G_i^t = A_i^t B_i^t`, with per-layer, per-epoch Gaussian statistics.
  * It does not say whether A and B are states or increments, and gives no code.

**Synthesis (unchanged by any Phase-4 result):**
* The PAPER-LITERAL object is the factor pair `[A_i, B_i]` that Algorithm 1 transmits.
* Its semantics (absolute state, increment or gradient) are **UNKNOWN**.
* Magnitude and distribution clues (`‖[A,B]‖² = 14.29`, i.e. a per-element RMS of 1.3e-3 over 8,388,608 LLaMA-7B factor elements; Gaussian ±0.01 histograms that narrow over time and are labelled "Gradient Value"; "converging toward 0") favour increment- or gradient-like objects.
* The literal transmission, the max-surface axis and the B·A visualisation scale are compatible with absolute states.
* No single reading fits every clue. The F0 audit added DR-22 to DR-27.
* DR-28 (added later, DERIVED): the paper's noise-table reconstruction MSE (4.59e-7..5.25e-7) is 27-31 % of the per-element signal power implied by its own `‖G‖² = 14.29`, a relative squared error of about 0.3 if both refer to the same object.

## 3. Micro-batch fidelity (F1, PHASE4-DIAGNOSTIC)

**Implementation:**
* `local_train.microbatch_mode = virtual_paper_microbatch` builds the logical micro-batch of 16 exactly as the physical one: same rows, same left padding to the longest row, same labels and masks.
* It streams `physical_chunk_size` rows at a time. Each chunk's summed token loss is divided by the logical micro-batch's label-token count times the accumulation group size.
* `physical_microbatch` is unchanged and remains the default.

**CPU (tiny LLaMA, tests):** chunks 1, 2, 3 and 16 meet every criterion against physical 16, including a full round with a partial accumulation group. With dropout on, runs are reproducible with no loss-scaling error.

**GPU small model** (`JackFram/llama-160m` @ `aca9b68`):
* Setup: fp32, eager attention, gradient checkpointing (recompute only), deterministic. Client 26, a real batch of 32 Dolly examples (cutoff 512), warm start from one local round. Reference: physical 16.
* File: `results/phase4/microbatch/f1_llama160m_validation.json`.

| Mode vs physical 16 | loss rel. | grad cosine | grad rel. L2 | adapter rel. L2 (after one Adam step) | verdict |
|---|---|---|---|---|---|
| virtual, chunk 16 | 9.5e-9 | 1 (bitwise) | 0 | 0 | PASS |
| virtual, chunk 2 | 6.9e-8 | 0.9999999971 | 7.73e-5 | **5.00e-5** | FAIL (adapter) |
| virtual, chunk 1 | 9.5e-8 | 0.9999999971 | 7.73e-5 | **5.00e-5** | FAIL (adapter) |
| thresholds | <= 1e-6 | >= 0.99999 | <= 1e-4 | <= 1e-5 | |

**Diagnosis of the adapter failure:**
* The virtual code path is exact. With chunk 16 it is bitwise identical to physical 16.
* The difference comes from fp32 reductions whose order depends on the kernel shape (1- or 2-row chunks versus 16-row micro-batches). Controls:
  * cuBLASLt instead of the default BLAS changes nothing;
  * chunk 1 vs chunk 2 differ by 3.7e-6 (gradient) and 2.1e-6 (adapter);
  * the same-shape noise floor (rows reversed within every micro-batch) is 3.8e-6 (gradient) and 1.5e-6 (adapter), which passes.
* The 5e-5 adapter difference is carried by **9 of 589,824 elements**. Their Adam first-step update flips sign:
  * Adam's first update is about `lr · g / (|g| + eps)`, so it is ±lr for any non-negligible g;
  * these elements have |g| <= 1.9e-7 (median 1.6e-8, against an overall median of 7.1e-5), where rounding decides the sign.
* With dropout on, every mode is bitwise reproducible. The loss ratio to physical 16 is 1.0000096 (chunk 1) and 1.0000000 (chunk 2), with a maximum per-batch deviation of 3.3e-5, so there is no systematic loss-scaling error.
* The pre-registered threshold was not changed. **F1 is not met on the GPU for chunks 1 and 2.**

**Qwen1.5-1.8B** (bf16, sdpa, gradient checkpointing):
* Setup: one real batch of 32 examples of client 26, at the Phase-3 round-10 global adapter.
* Physical 16 was not attempted on Qwen: Phase 3 measured that micro-batch 2 already came within 256 MiB of the allocator cap. Reference: virtual chunk 1.
* File: `results/phase4/microbatch/f1_qwen_validation.json`.

| Mode vs virtual chunk 1 | loss rel. | grad cosine (all / A / B) | grad rel. L2 | grad norm ratio | Adam first-step sign flips | padded tokens | s / step of 32 | peak alloc / reserved GiB |
|---|---|---|---|---|---|---|---|---|
| virtual chunk 1 (reference) | - | - | - | - | - | 16,384 | 14.0 | 4.46 / 5.07 |
| virtual chunk 2 | 1.7e-4 | 0.998 / 0.999 / 0.998 | 0.064 | 1.010 | 55,241 (1.8 %) | 16,384 | 9.6 | 5.38 / 6.23 |
| **physical micro-batch 1 (Phase 3 and this baseline)** | **2.8e-2** | **0.866 / 0.920 / 0.866** | **0.66** | **1.297** | **444,806 (14.1 %)** | 8,832 | 10.9 | 4.46 / 5.79 |

* The noise floor (rows reversed) is 1.3e-7. The chunk-2 difference (6.4 % rel. L2) is bf16 kernel-shape noise.
* The micro-batch-1 difference is ten times larger and is semantic. Micro-batch 1 averages per-example token means (every example weighs the same), whereas the paper's 16 x 2 averages tokens within each micro-batch of 16. The micro-batch-1 gradient is 30 % larger in norm on this batch.
* With dropout on, the micro-batch-1 loss differs from the reference by -1.6 % on average (max 4.2 %), whereas chunk 2 stays within 3.5e-4.

**Decision (F7 literal rule):** "If virtual passes F1, use logical mb16/acc 2; otherwise use mb1 and label the deviation prominently."
* Virtual did not pass the GPU adapter criterion, so the baseline uses **physical micro-batch 1 x 32**. This is a PROMINENT deviation.
* On Qwen, virtual chunk 2 would have been feasible: faster per step than micro-batch 1 (more padded tokens, fewer forward passes), with reserved memory 0.46 GiB below the cap.

**Provenance.** Both F1 records were produced from an uncommitted tree at `62dac71`. Clean reruns at `3c73e66` reproduce every deterministic field bitwise (`results/phase4/microbatch/*_rerun_clean.json`; section 15).

## 4. Gauge diagnostics (F2)

`compression/gauge.py` computes the spectrum of `M = s B A` from the r x r core `C = s R_B R_Aᵀ`, where `B = Q_B R_B` and `Aᵀ = Q_A R_A`. It never allocates a dense d x d matrix.

**Reported quantities:** ‖A‖², ‖B‖², ‖M‖_F², ‖M‖_*, singular values and stable rank, per module and in total.

**Tests** (`tests/unit/test_gauge.py`, CPU, all pass):
* factor norms change under `B' = B Q`, `A' = Q⁻¹ A` while the product and the spectrum do not;
* balanced factors: product preservation, gauge invariance, deterministic repeat;
* rank-deficient and near-zero singular values;
* Qwen geometry (2048 x 2048, r = 8) without a dense allocation;
* the effective delta against a dense reference;
* diagnostics on an adapter state.

**Real-data demonstration** (Phase-3 federated snapshot t16/c7):
* A random well-conditioned Q per module changes the raw factor energy from 263.13 to 309.94 (A² 261.53 → 307.01, B² 1.599 → 2.929).
* The largest relative singular-value change is 9.4e-16.
* Raw LoRA factor norms, including the paper's 14.29, are therefore gauge-dependent (R5).

## 5. Balanced effective state (F3, R2)

**Construction:**
* `balanced_effective_state`: `B_c = Q_B U √Σ`, `A_c = √Σ Vᵀ Q_Aᵀ`, from the SVD of the r x r core.
* Deterministic signs: the largest-magnitude entry of each left singular vector is positive.
* Numerically zero components (σ <= 1e-12 σ_max) are set to zero; ties are counted.

**Properties:**
* `B_c A_c = s B A`, independent of the gauge.
* `‖A_c‖² + ‖B_c‖² = 2 ‖M‖_*`, which is the minimum over all gauges.
* Data check (validation medians): balanced energy 70.076 = 2 x 35.038.

**Phase-3 federated snapshots** (`results/phase4/forensics/f4_federated_state_stats.json`, validation medians, 20 snapshots, rounds 16-19):

| Quantity | Value |
|---|---|
| raw factor energy ‖A‖² + ‖B‖² | 263.5 (A 261.9, B 1.64) |
| ‖M‖_F² | 2.34 |
| ‖M‖_* | 35.04 |
| balanced energy | 70.08 |
| stable rank per module | median 1.87 |
| R2 per-element RMS / max abs | 4.7e-3 / 0.033 |
| tied / zero components | 0 / 0 |

* The PEFT-default gauge carries 3.8 times the minimal factor energy, almost all of it in A.
* R2 is **not** claimed to be the paper's representation.

## 6. Effective-delta rank analysis (F4, R3)

**Definition:** `dM = s (B_e A_e - B_s A_s)`, with rank <= 2r = 16. Its exact spectrum comes from the 2r x 2r core, and the best rank-8 approximation follows Eckart-Young.

| Set (validation medians) | retained energy (rank 8) | truncation rel. Fro error | rank-8 product cosine | exact rank | dM energy |
|---|---|---|---|---|---|
| `federated_pretrain` (primary) | **0.9872** | 0.113 | 0.9936 | 16 in all 1,920 validation modules | 0.0715 |
| `local_pretrain` | 0.9820 | 0.134 | 0.9910 | | 0.0728 |

**Pre-registered prerequisite:** the median validation retained energy must be >= 0.95. It is met on both sets, so R3 is **not structurally lossy**. Round-0 increments (start = PEFT initial state, B = 0) have rank <= 8 and retain 100 %.

**Spectrum:** the mean normalised validation spectrum is
`1.000, 0.624, 0.481, 0.419, 0.379, 0.350, 0.321, 0.288 | 0.148, 0.061, 0.033, 0.022, 0.016, 0.012, 0.009, 0.007`.

**Gauge artefact in raw terms:**
* Raw factor increments ‖ΔA‖² + ‖ΔB‖² are 3.6e-4 of the raw state energy.
* In effective terms the per-round change ‖dM‖² is about 3 % of ‖M‖².
* The Phase-3 observation that "the innovation is tiny compared with the state" is therefore largely a property of the PEFT gauge (A ~ Kaiming-uniform, B ~ 0), not of the update.

## 7. Bounded actual-gradient forensics (F5, PHASE4-FORENSIC)

**Collection** (`results/phase4/forensics/f5_gradient_forensics.json`; commit `e8100e9`, tree clean apart from untracked `results/phase4/eval/`).

* **Schedule.** The Phase-3 federated D1 schedule, seeds and training semantics:
  * namespace `tgap_fed`, seed 1;
  * physical micro-batch 1 x 32;
  * AdamW, lr 1.5e-4 with linear decay over the local epoch, gradient clipping at max norm 1.0, optimiser reset every round.
* **Scope.** 2 rounds x 5 clients = 10 client-rounds and 20 optimizer steps (1-3 per client: D1 holds 27-80 examples per client).
* **Recorded per step:** the accumulated gradient before clipping, the gradient after clipping, the parameter change, and the start/end states.
* **Storage.** 722 MB, outside Git (`CGFED_RUNS/phase4_gradient_forensics_seed1/gradient_forensics`).

**Unchanged optimisation:** all **10/10** end states are bitwise identical to the Phase-3 round-0/1 snapshots, and the summed recorded step deltas reproduce every end state. The hooks are read-only on the GPU too.

| Family (per client-round unless stated) | n | per-element RMS | mean squared norm (min-max) | A/B energy | inter-client cosine | inter-time cosine (round-0 vs round-1 means) | max abs | RMS / max abs | RMS vs paper 1.305e-3 |
|---|---|---|---|---|---|---|---|---|---|
| `pre_clip_step_gradient` (per step) | 20 | 9.40e-4 | 2.78 (0.79-6.76) | 0.046 | 0.73 | 0.87 | 0.090 | 0.010 | 0.72 |
| `last_step_gradient` | 10 | 9.29e-4 | 2.71 (1.31-4.74) | 0.041 | 0.69 | 0.64 | 0.090 | 0.010 | 0.71 |
| **`mean_step_gradient` (R4)** | 10 | 9.06e-4 | 2.58 (0.75-5.30) | 0.038 | 0.88 | 0.88 | 0.064 | 0.014 | 0.69 |
| `optimizer_step_delta` (per step) | 20 | 1.03e-4 | 0.033 (0.005-0.071) | 0.58 | 0.72 | 0.71 | 1.5e-4 (= lr) | 0.69 | 0.079 |
| `local_epoch_delta` | 10 | 1.92e-4 | 0.115 (0.035-0.223) | 0.63 | 0.78 | 0.71 | 3.0e-4 | 0.64 | 0.147 |

Geometry: Qwen1.5-1.8B, r 8, q/k/v/o, 3,145,728 factor elements, s = alpha/r = 2. **No row is gauge-invariant.** Under `B' = B Q`, `A' = Q⁻¹ A`, the factor gradients transform as `g_B Q⁻ᵀ` and `Qᵀ g_A`.

**Further statistics:**
* **Clipping.** Pre-clip total gradient norm: mean 1.59, range 0.89-2.60. **18 of 20 steps (90 %) were clipped.**
* **Consecutive step deltas** within a client-round have mean cosine 0.84.
* **First-order effective change.** `s (ΔB A_s + B_s ΔA)` matches the exact `ΔM = s (B_e A_e - B_s A_s)` of the local epoch with mean cosine 0.9998 and mean relative Frobenius error 1.6 % (max 3.3 %). The local update is in the linear regime.
* **Round-0 structure.** At round 0, B = 0, so the first step's A-gradient is **exactly zero** for every client.
  * Clients 55 and 75 took a single step, so their `mean_step_gradient` has no A part at all.
  * A share of the R4 energy, round 0 (the R4 training split): 0.011, 0.032, 0, 0, 0.014.
  * Round 1 (the R4 validation split): 0.070-0.102.
  * **The pre-registered temporal split of R4 is therefore a systematic train/validation shift**, and only 5 + 5 samples are available.
* **Shape.** Gradients are heavy-tailed (RMS / max ≈ 0.01). Optimizer steps are near-saturated Adam steps (RMS / max ≈ 0.7, max = lr).

**Paper comparison** (DERIVED, per-element RMS only, R5):
* The paper's `‖G‖² = 14.29` over 8,388,608 LLaMA-7B factor elements is an RMS of 1.305e-3, with alpha, base precision and gauge unknown.
* Our raw gradients have 0.69-0.72 times that RMS; our 1-3-step local deltas have 0.08-0.15 times it.
* By magnitude alone, the paper's number is closer to a raw factor gradient than to our increments. This does **not** establish what the paper encoded, for three reasons:
  * the loss scale, model, data, LoRA alpha and step count all differ;
  * the norm is gauge-dependent;
  * an increment over many more local steps (the paper's local epochs are unknown) would grow towards the same magnitude.
* **Distribution shape** (`results/phase4/forensics/f5_distribution_shape.json`, commit `0d436c3`; gradients exclude exact zeros):
  * **R4 / gradients.** 99.94 % of non-zero values lie within ±0.01 (|x| p50 1.2e-4, p99 4.1e-3, p99.9 8.7e-3, max 0.064). This is compatible with the paper's ±0.01 histogram range. But the distribution is strongly leptokurtic (excess kurtosis 72; pre-clip per-step gradients 86, against 0 for a Gaussian). A histogram cut at ±0.01 would show it as a sharp peak at zero.
  * **1-3-step Adam deltas** (zeros included). They are confined to ±1.5e-4 (step) or ±3.0e-4 (local epoch) and platykurtic (excess kurtosis -1.20 and -1.39, saturated at ±lr). They are 30-70 times narrower than the paper's range. The exact zeros are the A factors of first steps at B = 0.
  * By range, the paper's description is compatible with raw gradients at our scale, and with increments only if those accumulate far more steps than ours. "Gaussian-like" fits neither family literally.

## 8. Fixed compressibility screen (F6, PHASE4-FORENSIC)

**Protocol** (pre-registered in `phase4_preregistration.md` section 4; commit `e8100e9`, tree clean apart from untracked result files):
* Fixed ResNet-3 AE, Phi `layer_major_qkvo_AtB` [1, 2048, 1536], latent [64, 32, 24] = 1/64.
* MSE, Adam 2e-4, batch 4, 3,000 iterations, evaluation every 50, best-D1-validation checkpoint, AE init seed 1.
* Scale rule: `global_maxabs_train` only if the training p99.9 of |x| exceeds 0.95. It never did (0.019 / 0.0086 / 0.0093), so **only `none` ran**.
* R0 and R1 are the frozen Phase-3 results, read against the same criteria (`screen-reuse`, DERIVED, not retrained).

| ID | Representation | data (train / val) | input RMS | modes | S1 | S2 rep cos >= 0.90 | S3 rep RSE <= 0.50 | S4 prod cos >= 0.90 | S5 prod rel Fro <= 0.50 | S6 input-dependent (margin) | S7 vs train mean | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R0 | `adapter_state` (Phase 3, best of `none` / `global_rms` / `factor_rms`) | 80 / 20 | A 0.013 | 3 | ✓ | ✗ (≤ 0.19) | ✗ (≥ 0.965) | ✗ (≤ 0.034) | ✗ (≥ 0.999) | ✗ / trivial ✓ (1e-7) | ✗ | **FAIL** |
| R1 | `adapter_delta` (Phase 3 A8, `factor_rms`) | 80 / 20 | - | 1 | ✓ | ✗ (1.4e-4) | ✗ (1.00) | ✗ (-5e-5) | ✗ (1.00) | trivial ✓ (3.6e-8) | ✗ | **FAIL** |
| R2 | `balanced_effective_state` | 80 / 20 | 0.0041 | `none` | ✓ | ✗ (-0.0006) | ✗ (7.09) | ✗ (0.0000) | ✗ (57.3) | ✗ (-2.4e-8) | ✗ | **FAIL** |
| R3 | `balanced_effective_delta_r8` | 80 / 20 | 0.0020 | `none` | ✓ | ✗ (0.0001) | ✗ (1.02) | ✗ (0.0000) | ✗ (1.01) | trivial ✓ (5e-6) | ✗ | **FAIL** |
| R4 | `mean_step_gradient` | 5 / 5 | 0.0010 | `none` | ✓ | ✗ (0.0005) | ✗ (318) | ✗ (0.0007) | ✗ (12.6) | trivial ✓ (6.5e-5) | ✗ | **FAIL** |

Values are validation, best-val AE (R0/R1: the Phase-3 best-val AE). "Trivial ✓" means S6 passed with a relative margin below 1e-4, which carries no information. The pre-registered S6 has no margin, so the margin is reported with it.

**No candidate passes. Nothing is forwarded, and no operational AE/FAF run was started.**

**Baselines (validation):**

| ID | train-mean rep RSE / cos | train-mean prod cos | zero rep RSE | best-val AE rep norm ratio |
|---|---|---|---|---|
| R2 | 0.83 / 0.41 | 0.67 | 1.00 | 2.47 |
| R3 | 1.01 / 0.002 | 0.016 | 1.00 | 0.15 |
| R4 | 0.50 / 0.82 | 0.84 | 1.00 | 17.8 |

* **R2.** The AE's innovation RSE is 113,892, against 26.9 for the train mean. Aggregate innovation rel Fro: 337.
* **R3.** Against the exact dM the AE gives cosine 0.0000 and rel Fro 1.012.
* **R4.** The aggregate gradient-space cosine is 0.0005 for the AE, against 0.86 for the train mean.

**Why the AE fails: a scale floor, not an information limit** (training-space MSE from the recorded curves):

| ID | input mean square (zero-predictor MSE) | AE MSE at iteration 1 | last training-batch MSE at 3,000 | ratio to zero predictor | best-val iteration |
|---|---|---|---|---|---|
| R2 | 1.64e-5 | 0.194 | 7.0e-5 | 4.3x | 1550 |
| R3 | 4.17e-6 | 0.194 | 4.6e-5 | 11x | 2850 |
| R4 | 1.05e-6 | 0.181 | 9.7e-5 | 93x | 2650 |

* The randomly initialised AE outputs about 0.44 RMS (Tanh), about 100-440 times the inputs.
* Within the paper-silent budget of 3,000 Adam steps at 2e-4, it never reaches even the zero-output MSE **on its own training data**. On the training split, R2/R3/R4 give rep RSE 3.3 / 10.1 / 84 at cosine ≈ 0.
* So F6 does not measure whether R2-R4 are compressible at 1/64. It shows that the fixed AE protocol under `none` cannot represent inputs at these scales.
* This is a design limitation of our pre-registered scale rule, which only guarded against Tanh overflow, not against inputs far below the AE's output scale. The pre-registration forbids adding modes, so none were added.
* Phase-3 R0 under `none` failed the same way. Its normalised variants failed differently: they fitted the shared state but carried no client innovation.
* R4 has two additional limits, both pre-registered and reported:
  * the screen is tiny (5 + 5 snapshots);
  * training round 0 has a near-zero A part (section 7), so the split is shifted.

## 9. Representation decision matrix

| ID | Representation | Source evidence | Exact semantics | Gauge-invariant | Exact / lossy before the AE | Compression metrics (validation, best-val AE) | F6 | Paper-supported |
|---|---|---|---|---|---|---|---|---|
| R0 | `adapter_state` | Alg. 1 transmits `[A_i, B_i]` after local training; Table 1 shape = factor count; Shepherd uploads states | raw PEFT factors of the client's end state (A ≈ Kaiming-uniform init + drift, B learned from 0) | no | exact | rep cos ≤ 0.19, prod cos ≤ 0.034, innovation cos ≈ -0.04 in the normalised modes (Phase 3) | **FAIL** | **PAPER-LITERAL object; semantics UNKNOWN** |
| R1 | `adapter_delta` | magnitude/histogram clues; "local updates", "difference between two sequential model parameters" | raw factor change end - start of one local round (1-3 steps at our scale) | no | exact | rep cos 1.4e-4, innovation cos -5e-5 (Phase 3 A8) | **FAIL** | consistent with several clues; not literal |
| R2 | `balanced_effective_state` | none (PHASE4-FORENSIC construct) | canonical balanced factors of `s B A`: `B_c = Q_B U √Σ`, `A_c = √Σ Vᵀ Q_Aᵀ`, deterministic signs | **yes** | exact (product preserved; fp32 storage) | rep cos -0.0006, RSE 7.09; prod cos 0.0000 | **FAIL** (scale floor) | no |
| R3 | `balanced_effective_delta_r8` | none (PHASE4-FORENSIC); in the spirit of "ΔW = BA" plus the increment reading | balanced factors of the best rank-8 approximation of `dM = s (B_e A_e - B_s A_s)` | **yes** | **lossy**: rank-8 truncation keeps 98.7 % of the energy (median val), rel Fro error 0.113, product cos 0.994 | rep cos 0.0001, RSE 1.02; prod cos 0.0000 | **FAIL** (scale floor) | no |
| R4 | `mean_step_gradient` | "compress gradients", "low-rank decomposition of input gradients", histogram axis "Gradient Value"; per-element RMS 0.69x the paper's (DERIVED) | mean over local steps of the pre-clip factor gradients | no | exact as a statistic. It is **not** the update: AdamW, clipping (90 % of steps) and the LR schedule intervene | rep cos 0.0005, RSE 318; prod cos 0.0007 | **FAIL** (scale floor; 5 + 5 shifted split) | wording only; no operational server semantics |

* **Communication.** No communication claim is made for R2-R4: none has defined operational transmission semantics in our pipeline. The 1/64 latent is an element ratio of the AE input, not a system communication reduction.
* **Paper support.** No row is upgraded to PAPER-LITERAL by any Phase-4 result.

## 10. Seed-1 Tier-B baseline (F7, PHASE4-BASELINE)

**Setup.** RESOURCE-FEASIBLE, not paper-faithful:
* Qwen1.5-1.8B (`7846de7e`), bf16, sdpa, gradient checkpointing.
* LoRA r 8, alpha 16 (s = 2), dropout 0.05 on q/k/v/o.
* Local training: **physical micro-batch 1 x 32 accumulation** (the PROMINENT deviation, section 3), one local epoch on the client's D2.
* Federation: 100 Dirichlet(0.5) clients; Shepherd sampler `RandomState(round)` with K = 5; sample-weighted FedAvg of adapter states; 20 rounds.
* Held-out: Dolly loss on 80 examples (17,488 label tokens) after every round, one example at a time.
* Scientific seed 1 (`run.seed`, `lora.init_seed`).

**Arms and provenance.** Every run records its commit, tree state and resolved-config SHA-256 in `run_metadata.json` / `config.sha256.json`.

| Arm | Stage | Commit | Tree at start | Resolved-config SHA-256 | Result |
|---|---|---|---|---|---|
| Base | initial adapter (B = 0, output identical to the base model) | - | - | - | held-out 2.36993 |
| LoRA-FT | `lora_ft` | `ad6b3dd` | clean | `972a4129…` | 20 rounds, held-out 1.76646 |
| FAF-Identity | `faf_identity` | `ff24f5e` | clean except the untracked result directory `results/phase4/screen/` | `4c3ee8fb…` | 20 rounds, bitwise identical to LoRA-FT |
| Cent | `cent_resource_matched_seed1` | `ff24f5e` | clean except the untracked result directory `results/phase4/screen/` | `574993e2…` | 1 pooled pass (10,445 examples, 327 optimizer steps), held-out 1.75555 |

**Training-path code between the two FL commits.** The only change on the training path from `ad6b3dd` to `ff24f5e` is in `federated/simulator.py`: an optional `observer_factory` that is `None` in every baseline run, so `trainer.train` is called exactly as before. The other changes do not touch the training path:
* `cmd_run_fl` has byte-identical source at both commits;
* `config.py` only adds the evaluation-subset fields of `BenchmarkSpec`;
* `compression/normalization.py` is imported only when an AE codec is built;
* `forensics/*` and `evaluation/*` are not imported by `run-fl`.

**Per-round trajectory, LoRA-FT** (`results/phase4/baseline/seed1_baseline.json`). FAF-Identity is bitwise identical in every compared quantity (section 11). Only wall time, throughput and memory differ, because they are measurements of the machine.

**lora_ft**: status complete, 20 rounds, held-out 2.36993 -> 1.76646, round wall 3512 s, commit `ad6b3dd`, dirty []

| round | clients | samples | opt. steps | train loss (sample-weighted) | held-out loss | ‖A‖² | ‖B‖² | ‖M‖_F² | ‖M‖_* | ‖ΔA‖ | ‖ΔB‖ | ‖ΔM‖_F | label tok/s | peak reserved GiB | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 2 26 55 75 86 | 589 | 21 | 2.3732 | 2.14629 | 256.31 | 0.174 | 0.233 | 11.86 | 0.2450 | 0.4173 | 0.4828 | 662 | 5.91 | 216 |
| 1 | 33 80 81 84 93 | 605 | 21 | 2.1412 | 1.97540 | 257.12 | 0.493 | 0.665 | 19.58 | 0.4231 | 0.3824 | 0.4458 | 618 | 5.91 | 226 |
| 2 | 16 24 30 56 83 | 468 | 17 | 1.9350 | 1.88643 | 257.90 | 0.762 | 1.037 | 23.95 | 0.3647 | 0.3183 | 0.3744 | 517 | 5.91 | 158 |
| 3 | 6 64 67 93 96 | 529 | 18 | 1.8523 | 1.83481 | 258.49 | 0.939 | 1.285 | 26.42 | 0.2881 | 0.2499 | 0.2953 | 572 | 5.91 | 186 |
| 4 | 10 16 20 63 96 | 469 | 17 | 1.7822 | 1.81479 | 259.20 | 1.114 | 1.537 | 28.63 | 0.2966 | 0.2410 | 0.2873 | 578 | 5.91 | 166 |
| 5 | 28 32 46 66 74 | 440 | 16 | 1.6913 | 1.78764 | 259.93 | 1.289 | 1.796 | 30.66 | 0.2848 | 0.2383 | 0.2861 | 622 | 5.91 | 165 |
| 6 | 0 17 34 45 60 | 481 | 18 | 1.7563 | 1.78787 | 260.43 | 1.411 | 1.979 | 32.01 | 0.2321 | 0.2105 | 0.2532 | 597 | 5.91 | 167 |
| 7 | 26 37 49 78 91 | 611 | 22 | 1.6935 | 1.77889 | 260.94 | 1.511 | 2.132 | 33.22 | 0.2250 | 0.1955 | 0.2352 | 624 | 5.91 | 217 |
| 8 | 1 17 23 44 55 | 589 | 21 | 1.6208 | 1.77877 | 261.32 | 1.588 | 2.251 | 34.17 | 0.1846 | 0.1692 | 0.2032 | 573 | 5.91 | 203 |
| 9 | 3 42 46 68 75 | 391 | 15 | 1.6477 | 1.77306 | 261.59 | 1.643 | 2.336 | 34.82 | 0.1540 | 0.1451 | 0.1747 | 604 | 5.91 | 146 |
| 10 | 14 19 37 43 66 | 419 | 17 | 1.6466 | 1.77527 | 262.02 | 1.727 | 2.467 | 35.79 | 0.1890 | 0.1776 | 0.2145 | 526 | 5.91 | 141 |
| 11 | 22 41 46 49 58 | 459 | 18 | 1.6184 | 1.77006 | 262.32 | 1.796 | 2.574 | 36.58 | 0.1754 | 0.1678 | 0.2030 | 600 | 5.91 | 165 |
| 12 | 14 17 41 68 92 | 485 | 19 | 1.7623 | 1.76998 | 262.68 | 1.869 | 2.688 | 37.43 | 0.1697 | 0.1617 | 0.1958 | 553 | 5.91 | 170 |
| 13 | 14 37 43 62 83 | 529 | 20 | 1.6332 | 1.77077 | 263.15 | 1.950 | 2.820 | 38.35 | 0.1756 | 0.1520 | 0.1850 | 490 | 5.91 | 165 |
| 14 | 24 35 39 44 55 | 612 | 21 | 1.6120 | 1.76628 | 263.53 | 2.027 | 2.944 | 39.17 | 0.1752 | 0.1600 | 0.1953 | 578 | 5.91 | 215 |
| 15 | 36 46 51 57 84 | 396 | 14 | 1.6386 | 1.76509 | 263.94 | 2.098 | 3.061 | 39.96 | 0.1686 | 0.1521 | 0.1850 | 608 | 5.91 | 149 |
| 16 | 7 25 42 47 71 | 497 | 18 | 1.5405 | 1.76731 | 264.16 | 2.153 | 3.150 | 40.52 | 0.1512 | 0.1447 | 0.1766 | 526 | 5.91 | 164 |
| 17 | 4 9 28 71 73 | 457 | 16 | 1.6294 | 1.76640 | 264.48 | 2.210 | 3.246 | 41.12 | 0.1506 | 0.1429 | 0.1745 | 539 | 5.91 | 152 |
| 18 | 33 43 47 53 54 | 576 | 20 | 1.6067 | 1.76711 | 264.89 | 2.292 | 3.381 | 41.98 | 0.1831 | 0.1708 | 0.2092 | 521 | 5.91 | 186 |
| 19 | 2 16 18 72 85 | 474 | 18 | 1.6912 | 1.76646 | 265.06 | 2.331 | 3.444 | 42.39 | 0.1403 | 0.1343 | 0.1638 | 505 | 5.91 | 156 |

**Centralized reference.** `cent_resource_matched_seed1` is NOT the paper's Cent.

**cent**: status complete, 1 round, held-out 2.36993 -> 1.75555, round wall 3457 s, commit `ff24f5e`, dirty []

| round | clients | samples | opt. steps | train loss (sample-weighted) | held-out loss | ‖A‖² | ‖B‖² | ‖M‖_F² | ‖M‖_* | ‖ΔA‖ | ‖ΔB‖ | ‖ΔM‖_F | label tok/s | peak reserved GiB | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 10445 | 327 | 1.6598 | 1.75555 | 272.69 | 8.018 | 12.746 | 81.28 | 3.6198 | 2.8316 | 3.5702 | 553 | 5.91 | 3457 |

**Compute and effective update (DERIVED from the same file):**

| Arm | samples seen | optimizer steps | label tokens | final held-out | ‖M‖_F | ‖M‖_* | stable rank (module median) |
|---|---|---|---|---|---|---|---|
| Base | 0 | 0 | 0 | 2.36993 | 0 | 0 | - |
| LoRA-FT (= FAF-Identity) | 10,076 | 367 | 1,917,348 | 1.76646 | 1.856 | 42.39 | 1.89 |
| Cent (resource-matched) | 10,445 | 327 | 1,908,029 | 1.75555 | 3.570 | 81.28 | 1.95 |

**Logical communication** (uncompressed fp32 adapter state, 3,145,728 elements; no compression anywhere in F7):

| arm | uplink / client / round (B) | downlink / client / round (B) | two-way per round (B) | uplink total | downlink total | two-way total |
|---|---|---|---|---|---|---|
| lora_ft | 12,582,912 | 12,582,912 | 125,829,120-125,829,120 | 1,258,291,200 | 1,258,291,200 | 2,516,582,400 |
| faf_identity | 12,582,912 | 12,582,912 | 125,829,120-125,829,120 | 1,258,291,200 | 1,258,291,200 | 2,516,582,400 |
| cent | n/a | n/a | n/a | n/a | n/a | n/a (centralized reference (one pooled client): no client-server communication; the simulator's logical bytes are not a communication cost) |

The FAF-Identity payload serialises to 12,582,992 B (an 80-B header). The logical bytes equal the uncompressed state, because Identity compresses nothing.

**Observations (DERIVED; one seed, no error bars):**
* **Held-out loss.** It falls steeply over rounds 0-5 (2.370 → 1.788) and then flattens: 1.765-1.775 from round 9 on, best 1.76509 at round 15, final 1.76646.
* **Centralized reference.** At matched compute (10,445 vs 10,076 examples; 1.91 M vs 1.92 M label tokens) it reaches 1.75555, 0.011 lower. Its effective update is 1.9 times larger in Frobenius norm, consistent with FedAvg partially cancelling heterogeneous client updates.
* **Effective state.** It grows monotonically: ‖M‖_* from 11.9 to 42.4, ‖M‖_F² from 0.23 to 3.44. The per-round effective update ‖ΔM‖_F shrinks from 0.48 to 0.16-0.21.
* **Raw factor norms.** ‖A‖² moves only from 256.3 to 265.1 (+3.4 %), because it is dominated by the Kaiming-uniform initialisation, while ‖B‖² grows from 0.17 to 2.33. Raw factor norms are therefore a poor measure of training progress (R5).
* **Resources.** Peak reserved memory is 5.91 GiB in every round (allocator cap 6.69 GiB). Throughput is 490-662 label tokens/s, with the GPU under software power capping and software thermal slowdown (section 13).

## 11. Identity regression (F7)

`baseline-summary --identity-pair lora_ft,faf_identity` runs `compare_runs` (`src/cg_fedllm/federated/regression.py`) on the two complete run directories. Record: `results/phase4/baseline/seed1_baseline.json`.

| Check | Rounds | Result |
|---|---|---|
| selected clients and sample counts | 20/20 | equal |
| held-out loss (with example and token counts) | 20/20 | bitwise equal |
| global adapter hash | 20/20 | equal |
| raw ‖A‖², ‖B‖² | 20/20 | equal |
| effective BA norms (‖M‖_F², ‖M‖_*, stable-rank summary) | 20/20 | equal |
| update norms (‖ΔA‖, ‖ΔB‖, ‖ΔM‖_F) | 20/20 | equal |
| per-client records: start/end hashes, steps, micro-batches, label/padded tokens, first/last/mean loss | 100 client-rounds | equal |
| Identity codec round trip: recovered state equals the local state bitwise | 100 client-rounds | all |
| final adapter | - | bitwise equal; tensor hash `f22a96c7…a92e`; file SHA-256 `52eb1bd5…12e6` for both |
| initial adapter | - | tensors bitwise equal (the file bytes differ only in safetensors metadata order, P4-D4) |
| resolved configs | - | differ only in `codec.type` (`none` → `identity`); config SHA-256 `972a4129…` / `4c3ee8fb…` |
| provenance | - | `ad6b3dd` (clean) / `ff24f5e` (clean except an untracked result directory); training path unchanged (section 10) |

**Verdict: VALIDATED.** 13/13 checks pass, with 0 mismatches.
* Both arms use the physical micro-batch path, so the numerical caveat of the virtual path (P4-D2) does not apply.
* Bitwise equality was required and obtained.
* The comparison has negative controls in the test-suite: a tampered held-out loss (detected at round 1) and an unexpected config key both fail validation.

## 12. Benchmark timing and results (F8)

**Timing first** (PHASE4-DIAGNOSTIC; `results/phase4/eval/`).
* **Sample.** The first 10 questions of every subject: MMLU test 570, C-Eval val 520; 5-shot; `reference_eval_v1`; Qwen1.5-1.8B bf16.
* **Plan.** `eval-cost` replays the scorer's batching on CPU:
  * full plan: MMLU test 14,042 questions / 9,778,411 padded tokens; C-Eval val 1,346 / 713,823;
  * sample plan: 570 / 347,161 and 520 / 279,169.

| Model | sample padded tok/s (MMLU / C-Eval) | projected full MMLU test + C-Eval val | measured |
|---|---|---|---|
| Base | 7,670 / 7,596 | 23.2 min | 24.1 min wall (score 1,243 s + 97 s) |
| LoRA-FT (unmerged LoRA) | 6,198 / 6,161 | 28.7 min | 29.5 min wall (score 1,506 s + 111 s) |

* **Decision rule** (pre-registered): full evaluation if every model projects to <= 45 min. Result: **full** (max 28.7 min; `f8_timing_projection.json`).
* **C-Eval test** was not run. It was not unexpectedly cheap: about 14-18 min per model, from Phase 2's 6.53 M tokens at the measured rates.
* The first two timing attempts ran out of memory (P4-D3) before any result existed. They are preserved.

**Results** (PHASE4-BASELINE, 5-shot, `reference_eval_v1`, bf16, single run, seed-1 adapter; `results/phase4/eval/f8_eval_{base,lora_ft}_full.json`):

| Benchmark | Base | LoRA-FT (20 rounds) | Δ |
|---|---|---|---|
| MMLU test overall (question-weighted, 14,042) | 45.25 % | 45.71 % | +0.46 pp |
| MMLU subject macro | 46.45 % | 47.12 % | +0.67 pp |
| MMLU STEM / humanities / other / social sciences | 38.54 / 41.13 / 52.78 / 50.21 | 38.97 / 41.57 / 53.05 / 50.89 | +0.43 / +0.45 / +0.28 / +0.68 |
| C-Eval val average (subject macro, 1,346) | 59.08 % | 58.56 % | -0.52 pp |
| C-Eval val STEM / Social Science / Humanities / Other | 51.97 / 75.59 / 60.56 / 55.51 | 51.28 / 74.69 / 59.78 / 55.89 | -0.68 / -0.90 / -0.78 / +0.38 |
| C-Eval val Hard | 37.97 % | 37.83 % | -0.14 pp |

**Paired comparison** (DERIVED, `results/phase4/eval/f8_paired_comparison.json`: same questions, per-question correctness):

| Benchmark | both correct | Base only | LoRA-FT only | neither | question-weighted accuracy (Base / LoRA-FT) | exact McNemar p (two-sided) |
|---|---|---|---|---|---|---|
| MMLU test | 6,086 | 268 | 332 | 7,356 | 45.25 % / 45.71 % | 0.010 |
| C-Eval val | 756 | 29 | 29 | 532 | 58.32 % / 58.32 % | 1.0 |

The test compares these two fixed models on a fixed question set.
* It does **not** estimate training-seed variance: one seed was run.
* The +0.46 pp on MMLU is a small but detectable difference of this particular adapter.
* On C-Eval val, the -0.52 pp in the subject-macro average comes only from how the 58 discordant questions fall across subjects.

* **Sanity check.** Qwen-team numbers for Qwen1.5-1.8B are MMLU 46.8, C-Eval 59.7 (protocol not fully specified; **not** targets). Base is within 1.6 pp and 0.6 pp.
* **Scope of use.** These benchmarks played no part in any representation decision, and no compressed arm was evaluated: no codec passed F6.

## 13. Resource accounting (seed 1; RTX 4060 Laptop 8 GB, WDDM)

**Platform.** The software power cap and software thermal slowdown were active throughout: about 17 W against a 57.5 W limit (`nvidia-smi` throttle reasons). Every GPU process ran under an allocator cap of free VRAM - 256 MiB = 6.69 GiB. No system setting was changed. Throttling varied: during the F8 evaluation the GPU was seen unthrottled at 79.5 W.

| Stage | Runs | Wall time | Peak memory (reserved) | Notes |
|---|---|---|---|---|
| F1 GPU validation | llama-160m (5 modes), Qwen (3 modes) | 131 s + 242 s | 4.53 GiB / 6.23 GiB | plus clean reruns: 120 s + 187 s (plus one 16 s config-error attempt) |
| F4 statistics | 2 snapshot sets | 32 s + 33 s (CPU) | - | float64, no dense d x d |
| F7 LoRA-FT | 20 rounds | 58.7 min (rounds 3,512 s) | 5.91 GiB | 490-662 label tok/s |
| F7 FAF-Identity | 20 rounds | 60.0 min (rounds 3,568 s) | 5.91 GiB | |
| F7 Cent reference | 1 pooled pass | 58.5 min (3,457 s) | 5.91 GiB | 553 label tok/s |
| F8 timing | Base, LoRA-FT samples | 98 s + 119 s | - | plus 2 failed OOM attempts (33 s, 53 s) and 2 memory probes (about 3 min) |
| F5 gradients | 2 rounds x 5 clients | 212 s | - | 722 MB of dumps |
| F6 screens | R2, R3, R4 | 743 s + 713 s + 909 s | about 1.2 GB used | `none` only |
| F8 full evaluation | Base, LoRA-FT | 24.1 min + 29.5 min | <= 4.43 GiB (probe) | MMLU test + C-Eval val |

* **GPU time.** About 4.9 GPU-hours in total for Phase 4, all on the local GPU (no external GPU, no 7B).
* **Disk outside Git.** About 1.5 GB under `CGFED_RUNS`: gradient forensics 915 MB, baseline 567 MB, screens 48 MB, evaluation 7 MB.
* **Logical communication of the F7 baseline.** Per client per round, 12,582,912 B uplink + 12,582,912 B downlink; over 20 rounds, 2,516,582,400 B two-way. Identity compresses nothing.

## 14. Negative results (all reported, none averaged away)

1. **F1 adapter criterion.** It is not met on the GPU by the virtual paper micro-batch (chunks 1 and 2): adapter rel. L2 5.0e-5 > 1e-5. The baseline therefore uses micro-batch 1, which differs semantically from the paper (gradient cosine 0.866 on a fixed batch).
2. **R2 `balanced_effective_state`** fails the F6 screen (S2-S7).
3. **R3 `balanced_effective_delta_r8`** fails the F6 screen (S2-S5, S7). F4 shows rank 8 retains 98.7 % of the effective increment, so the failure is in the AE stage.
4. **R4 `mean_step_gradient`** fails the F6 screen (S2-S5, S7).
5. **R0/R1** remain failures under the Phase-4 criteria (S2-S5, S7).
6. **The F6 screen is uninformative for small-scale inputs** (P4-D5). In R2-R4 the AE never reached the zero-output MSE on its own training data.
7. **S6 passes trivially** for R0 (`global_rms`), R1, R3 and R4. Relative margins are 1e-7 to 6.5e-5, so the pre-registered criterion without a margin is not informative.
8. **F8 timing failed twice** (OOM, P4-D3) before the scorer fix.
9. **Federated vs centralized reference.** At equal compute (1.92 M vs 1.91 M label tokens), the federated baseline is 0.011 held-out loss behind the centralized reference (1.766 vs 1.756). One seed, no error bar.
10. **F1 provenance gap** (P4-D6): the original F1 results came from an uncommitted tree.

## 15. Deviations and provenance notes

**Deviations** (full rows in `deviations.md`, Phase 4 rows 35-43):

| Row | Deviation | Label |
|---|---|---|
| 35 | **Baseline micro-batch 1 x 32**, not the paper's 16 x 2; the virtual emulation failed the GPU adapter criterion | ENVIRONMENT + pre-registered rule, **PROMINENT** |
| 36 | held-out loss one example at a time | OURS |
| 37 | read-only gradient hooks, a verified no-op | OURS |
| 38 | screen input scaling `none` (+ `global_maxabs_train` only above Tanh range) | PHASE4 diagnostic, pre-registered |
| 39 | `cent_resource_matched_seed1`, not the paper's Cent | OURS |
| 40 | benchmarks in bf16 | ENVIRONMENT |
| 41 | R0/R1 reused and read against S1-S7 | OURS (R2) |
| 42 | three Phase-4 result labels | OURS |
| 43 | evaluation memory: no KV cache, halved batch budget | ENVIRONMENT |

**Provenance notes:**
* **Pre-registration timing.**
  * `phase4_preregistration.md` (`5cc2324`, 05:38:38) precedes every F4, F5 and F6 statistic and every F8 result.
  * It was committed 2.7 min after the LoRA-FT run started (05:35:59; config `ad6b3dd` committed before the start) and before that run's first round completed (05:39:42).
  * The initial (base) held-out evaluation (05:36:06) predates it; that number enters no pre-registered rule.
* **F1 originals** were produced from an uncommitted tree at `62dac71`. Both were rerun from a clean commit (`3c73e66`) with the same commands. The Qwen rerun used the hash-verified config `tierb_seed1_virtual_chunk2.yaml`, whose resolved SHA-256 `33a71083…` is identical to the original run's. Every deterministic field (217 + 117 numbers) is reproduced bitwise (`*_rerun_clean.json`). A first Qwen rerun attempt pointed at the wrong config file and stopped at config validation before any computation; its directory is preserved.
* **Baseline commits.** LoRA-FT ran at `ad6b3dd`; FAF-Identity and Cent at `ff24f5e`. The training path is unchanged between them (section 10). The chained script that ran them outlived a stopped shell task (P4-D7) but executed exactly the planned commands.
* **Code added after pre-registration.** Every change below added reporting, comparison, or evaluation infrastructure, made before the respective results existed; no pre-registered criterion or threshold changed:
  * the S6 margin field;
  * `screen-reuse`;
  * the extended `compare_runs`;
  * `eval-cost` / `eval-projection`;
  * the evaluation subset option (unused);
  * `gradient-shape`;
  * the scorer KV-cache fix.
* **R0/R1 mapping.** It was regenerated at a clean HEAD (`7699221`) before commit. The first copy had been produced from an uncommitted tree.
* **Preserved failures.** Both failed F8 timing runs are preserved (`eval_timing_base_failed_oom_*`). Each run directory records its commit, tree state and resolved-config SHA-256.

## 16. Remaining ambiguities

* **What `G` is.** The semantics of the encoded `[A_i, B_i]` (state, increment or gradient) remain UNKNOWN. No source found in F0 resolves them. By per-element RMS the paper's `‖G‖²` is nearer raw gradients than our 1-3-step increments, but it is gauge-, model-, loss-scale- and step-count-dependent (R5).
* **Paper-literal micro-batch semantics.** Whether to accept the GPU emulation (exact in code; numerical differences at Adam's first step) as paper-literal is a reviewer decision. On Qwen it would be feasible: faster than micro-batch 1, 0.46 GiB below the cap.
* **The AE screen.** Whether any of R2-R4 is compressible at 1/64 by the fixed ResNet-3 is open, because F6 under `none` cannot answer it (P4-D5). Whether the paper normalised its AE inputs is unknown (paper silent). DR-23 suggests an input-insensitive decoder; DR-28 shows its reported error is about 30 % of its own signal power.
* **Server semantics.** For R2-R4 these are undefined: how a server would aggregate balanced factors, rank-8 increments or mean gradients and update the global adapter. No communication claim is made for them.
* **The paper's Cent** configuration is unknown. Ours is a resource-matched centralized reference.
* **Variance.** One scientific seed, so no variance estimate for the baseline or the benchmarks.
* **Evaluation precision.** bf16, against unknown paper settings. The evaluator's correctness evidence remains the Phase-2 fp32 lm-eval cross-check on Qwen1.5-0.5B.

## 17. Reviewer decisions required for Phase 5

1. **Micro-batch semantics.** Either:
   * accept `virtual_paper_microbatch` (chunk 2) as the paper-literal emulation despite the 5e-5 adapter difference, and rerun the seed-1 baseline with it; or
   * keep micro-batch 1 as the labelled resource mode.
2. **Representation screen.** Whether to authorise a narrowly scoped, pre-registered **scale-normalised re-screen** of R2-R4 under the fixed ResNet-3, e.g. a frozen global scalar to unit RMS or to p99.9 = 0.95. It should come with an AE capacity control: fit a single snapshot or a training subset to below the zero-predictor MSE. That would separate optimisation failure from incompressibility. Or decide to stop representation work.
3. **The S6 criterion.** Give it a minimum relative margin, or an explicit shuffled-input control, in any future screen.
4. **Operational semantics.** Whether any forensic representation should receive defined server semantics before further study. Until then, no FAF run is possible for R2-R4.
5. **Seeds.** Whether to add seeds to the baseline (the LoRA-FT vs Cent gap of 0.011 has no error bar), and whether to evaluate Cent and C-Eval test (about 14-18 min per model).
6. **Scope.** No Tier A, no 7B, no external GPU (unchanged unless the reviewer decides otherwise).

## 18. Commands (seed 1)

Environment for every command: `CGFED_RUNS=D:\cgfed-runs`, `HF_HUB_OFFLINE=1`, conda env `cgfedllm`. Every run directory records its commit, tree state and resolved-config SHA-256.

```text
# F1 micro-batch fidelity (PHASE4-DIAGNOSTIC). The originals ran from an uncommitted tree at 62dac71; the *_rerun_clean files rerun them from a clean commit
cgfed validate-microbatch --config configs/phase4/validate_llama160m.yaml --stage llama160m_physical16_vs_virtual --modes physical:16,physical:16@cublaslt,virtual:16,virtual:1,virtual:2 --out results/phase4/microbatch/f1_llama160m_validation.json
# Qwen: the original run used the then-uncommitted content of tierb_seed1.yaml, identical to the committed tierb_seed1_virtual_chunk2.yaml (resolved SHA-256 33a71083...)
cgfed validate-microbatch --config configs/phase4/tierb_seed1_virtual_chunk2.yaml --set run.result_label=PHASE4-DIAGNOSTIC --stage qwen_virtual_vs_physical1 --modes virtual:1,virtual:2,physical:1 --adapter <phase3>/tgap_federated_state/fl/rounds/r0009/global_adapter.safetensors --out results/phase4/microbatch/f1_qwen_validation.json

# F2-F4 (PHASE4-FORENSIC; CPU, frozen Phase-3 snapshots; commit 3d452cd)
cgfed forensic-stats --config configs/phase4/f6_screen.yaml --snapshots <phase3>/tgap_federated_state --out results/phase4/forensics/f4_federated_state_stats.json
cgfed forensic-stats --config configs/phase4/f6_screen.yaml --set run.result_label=PHASE4-FORENSIC --snapshots <phase3>/tgap_local_state --out results/phase4/forensics/f4_local_state_stats.json

# F7 baseline (PHASE4-BASELINE)
cgfed run-fl --config configs/phase4/tierb_seed1.yaml --stage lora_ft
cgfed run-fl --config configs/phase4/tierb_seed1.yaml --set codec.type=identity --stage faf_identity
cgfed run-fl --config configs/phase4/cent_resource_matched_seed1.yaml --stage cent_resource_matched_seed1
cgfed baseline-summary --run lora_ft=<runs>/lora_ft --run faf_identity=<runs>/faf_identity --run cent=<runs>/cent_resource_matched_seed1 --identity-pair lora_ft,faf_identity --label PHASE4-BASELINE --out results/phase4/baseline/seed1_baseline.json

# F8 timing (PHASE4-DIAGNOSTIC) and decision (DERIVED)
cgfed eval-cost --config configs/phase4/eval_qwen15_1p8b_bf16.yaml --set run.result_label=PHASE4-DIAGNOSTIC --out results/phase4/eval/eval_cost_full.json
cgfed eval-cost --config configs/phase4/eval_timing_qwen15_1p8b_bf16.yaml --out results/phase4/eval/eval_cost_timing_sample.json
cgfed evaluate --config configs/phase4/eval_timing_qwen15_1p8b_bf16.yaml --stage eval_timing_base
cgfed evaluate --config configs/phase4/eval_timing_qwen15_1p8b_bf16.yaml --adapter <runs>/lora_ft/final_adapter.safetensors --stage eval_timing_lora_ft
cgfed eval-projection --timing-run base=<eval runs>/eval_timing_base --timing-run lora_ft=<eval runs>/eval_timing_lora_ft --cost-full results/phase4/eval/eval_cost_full.json --cost-sample results/phase4/eval/eval_cost_timing_sample.json --out results/phase4/eval/f8_timing_projection.json

# F5 (PHASE4-FORENSIC)
cgfed gradient-forensics --config configs/phase4/gradient_forensics_seed1.yaml --stage gradient_forensics --reference-snapshots <phase3>/tgap_federated_state --out results/phase4/forensics/f5_gradient_forensics.json

# F6 (PHASE4-FORENSIC; R0/R1 DERIVED reuse)
cgfed screen-reuse --phase3 R0_adapter_state_none=results/phase3/ae_viability/a5_federated_state_none.json ... --out results/phase4/screen/f6_r0_r1_phase3_reuse.json
cgfed forensic-screen --config configs/phase4/f6_screen.yaml --stage f6 --candidate balanced_effective_state --snapshots <phase3>/tgap_federated_state --out results/phase4/screen/f6_r2_balanced_effective_state.json
cgfed forensic-screen --config configs/phase4/f6_screen.yaml --stage f6 --candidate balanced_effective_delta_r8 --snapshots <phase3>/tgap_federated_state --out results/phase4/screen/f6_r3_balanced_effective_delta_r8.json
cgfed forensic-screen --config configs/phase4/f6_screen.yaml --stage f6 --candidate mean_step_gradient --snapshots <runs>/phase4_gradient_forensics_seed1/gradient_forensics --gradients <same>/gradients --out results/phase4/screen/f6_r4_mean_step_gradient.json
```
