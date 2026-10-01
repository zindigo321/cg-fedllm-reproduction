# Phase 3A findings: Tier-B AutoEncoder diagnosis (Qwen1.5-1.8B, scientific seed 1)

Protocol, metrics and gates: `phase3_preregistration.md`, committed before any Phase-3 GPU data existed
(`cabc3d7`). Evidence: `results/phase3/`. Labels: PHASE3-DIAGNOSTIC, PHASE3-SENSITIVITY, DERIVED. No C-Eval,
MMLU or D2 result was used for any decision. Benchmark evaluation was not run, because Phase 3B was not entered.

## 1. Verdict

* **A6: NO PRIMARY CODEC IS VIABLE.** None of the three pre-registered candidates (`none`, `global_rms`,
  `factor_rms`) passes the reconstruction gate on `federated_pretrain + adapter_state`.
* All three fail three criteria in the same way:
  * innovation cosine of about -0.02;
  * aggregate-update cosine of about -0.04;
  * no input dependence.

  Each candidate fails some of the factor-level criteria too (section 8).
* **A7** (one-round D2 FAF probe) was **not run**, because A6 selected no candidate.
* **Phase 3B was not entered.** There was no 20-round FAF, no replicate seeds and no benchmark evaluation.
* **A8 sensitivities:** neither sensitivity passes A6. `local_pretrain + adapter_state` (normalisation `none`, rule-determined because A6 selected nothing) fails 6 of 7 criteria, like the primary. `federated_pretrain + adapter_delta` with `factor_rms` collapses to a near-zero output (B norm ratio 4.1e-4, innovation cosine 1.4e-4) and fails G3-G6. The 'only the delta representation passes' branch therefore does not apply.
* The paper-literal reconstruction, absolute LoRA state at 1/64 with the reconstructed ResNet-3 AE, was not
  viable under our recovered implementation. Neither documented normalisation variant was viable either.

## 2. Configuration actually run

This is a resource-feasible configuration, NOT paper-faithful:
* Qwen1.5-1.8B @ `7846de7e`, bf16, no quantization (the paper appears to use an 8-bit base), SDPA;
* gradient checkpointing, deterministic kernels (verified bitwise-repeatable);
* allocator capped at 6.689 GiB;
* LoRA r 8 / alpha 16 / dropout 0.05 on q/k/v/o (alpha and dropout inferred from Shepherd);
* 100 Dirichlet(0.5) clients; per-client D1/D2 30/70;
* Shepherd sampler, K = 5 per round; sample-weighted FedAvg (the Shepherd baseline);
* lr 1.5e-4, 1 epoch, effective batch 32 as **micro-batch 1 x 32 accumulation** (section 3);
* scientific seed 1: `run.seed = lora.init_seed = autoencoder.init_seed = 1`.

## 3. A3: realistic-sequence calibration (PHASE3-DIAGNOSTIC)

| micro-batch | peak allocated (GiB) | peak reserved (GiB) | headroom to cap (GiB) | measurements within 256 MiB of cap | real tok/s | padded tok/s | s / optimizer step | worst-case micro-batch allocated (GiB) | rule outcome |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 5.405 | 6.596 | 0.094 | 3 of 6 | 1,249 | 1,653 | 5.23 | 5.369 | switch to micro-batch 1 |
| 1 | 4.487 | 6.463 | 0.227 | 1 of 6 | 990 | 1,003 | 6.61 | 4.450 | keep |

Bounded run: FL round 0 on D2 (the 5 sampler-selected clients, real trainer, seeds and initial adapter) plus one worst-case micro-batch of the longest examples (512 tokens). Allocator cap 6.689 GiB (free at start 6.939 GiB, margin 256 MiB); bf16 weights 3.469 GiB; model load 7.4 s.

| token lengths after cutoff 512 | n | mean | p50 | p90 | p95 | p99 | max | at cutoff |
|---|---|---|---|---|---|---|---|---|
| all D1 examples | 4,490 | 180.3 | 128 | 422 | 512 | 512 | 512 | 6.8 % |
| all D2 examples | 10,445 | 182.9 | 131 | 430 | 512 | 512 | 512 | 7.4 % |
| schedule `fl` (D2, 100 client-rounds) | 10,076 | 190.5 | 138 | 456 | 512 | 512 | 512 | – |
| schedule `tgap_fed` (D1, 100 client-rounds) | 4,333 | 189.5 | 137 | 450 | 512 | 512 | 512 | – |
| schedule `tgap_local` (D1, 100 client-rounds) | 5,060 | 235.8 | 190 | 512 | 512 | 512 | 512 | – |

| schedule | real tokens | padded at micro-batch 1 | padded at micro-batch 2 | optimizer steps |
|---|---|---|---|---|
| `fl` | 1,919,297 | 1,951,912 (x1.017) | 2,640,776 (x1.376) | 367 |
| `tgap_fed` | 821,101 | 835,176 (x1.017) | 1,128,200 (x1.374) | 178 |
| `tgap_local` | 1,192,940 | 1,208,480 (x1.013) | 1,595,776 (x1.338) | 180 |

**Decision.** The pre-registered rule triggered at micro-batch 2. Peak allocated memory stayed at 5.405 GiB,
but variable-length sequences fragment the caching allocator: peak *reserved* memory came within 256 MiB of
the cap in 3 of 6 measurements. The primary configuration therefore became micro-batch 1 x 32 accumulation.
At micro-batch 1, one measurement came near the cap, so the rule was not triggered again.

**Sustained throughput.** Throughput over the long A4 runs was lower than in the short calibration (DERIVED
from per-client timings). The federated collection averaged 700 label tokens/s, falling from about 1,000 to
about 500 after roughly 10 minutes. The local collection ran at 450-800 tokens/s. The cause is platform
throttling: `nvidia-smi` reported the software power cap and software thermal slowdown active at 51 °C on AC
power. Any time projection should use the sustained figure.

## 4. A4: TGAP datasets (DERIVED statistics; snapshots outside Git)

| | `federated_pretrain` (PRIMARY) | `local_pretrain` (SENSITIVITY) |
|---|---|---|
| snapshots / time indices | 100 / 20 | 100 / 20 |
| unique clients | 67 | 5 |
| participation histogram (participations: clients) | 1: 42, 2: 18, 3: 6, 4: 1 | 20: 5 |
| distinct start states | 20 | 96 |
| collection wall time | 1184 s | 1814 s |
| D1 samples per snapshot (min / p50 / max) | 17 / 43 / 85 | 31 / 47 / 80 |
| ‖A‖ per snapshot: mean [min, max] | 16.1122 [15.9988, 16.1975] | 16.1728 [15.9988, 16.6562] |
| ‖B‖ per snapshot: mean [min, max] | 1.0129 [0.1880, 1.3151] | 1.3052 [0.1880, 2.6710] |
| ‖innovation‖: mean [min, max] | 0.3178 [0.1880, 0.4717] | 0.3144 [0.1880, 0.4612] |
| ‖innovation A‖: mean [min, max] | 0.2200 [0, 0.3453] | 0.2198 [0, 0.3383] |
| ‖innovation B‖: mean [min, max] | 0.2264 [0.1879, 0.3473] | 0.2217 [0.1879, 0.3461] |
| ‖innovation‖ / ‖state‖: mean [min, max] | 0.0197 [0.0118, 0.0294] | 0.0194 [0.0118, 0.0287] |
| pooled energy A / B, state | 25,961 / 111.1 | 26,158 / 203.3 |
| pooled energy A / B, innovation | 5.10 / 5.23 | 5.13 / 5.03 |
| \|state A\| values p50 / p99 / max | 0.0111 / 0.0228 / 0.0259 | 0.0111 / 0.0231 / 0.0278 |
| \|state B\| values p50 / p99 / max | 5.53e-04 / 0.0022 / 0.0039 | 6.69e-04 / 0.0033 / 0.0059 |
| \|innovation A\| values p50 / p99 / max | 1.55e-04 / 2.95e-04 / 3.00e-04 | 1.50e-04 / 2.98e-04 / 3.00e-04 |
| \|innovation B\| values p50 / p99 / max | 1.58e-04 / 2.95e-04 / 3.00e-04 | 1.50e-04 / 2.97e-04 / 3.00e-04 |
| cosine of innovation: same time, different clients / same client, different times / different times | 0.407 / 0.286 / 0.099 | 0.294 / 0.252 / 0.163 |
| cosine of innovation of B: same time, different clients / same client, different times / different times | 0.356 / 0.277 / 0.048 | 0.208 / 0.191 / 0.085 |
| cosine of state: same time, different clients / same client, different times / different times | 1.000 / 0.998 / 0.997 | 0.992 / 0.995 / 0.991 |
| cosine of the sample-weighted mean innovation: adjacent time indices / all pairs | 0.255 / 0.172 | 0.596 / 0.393 |
| temporal split: train / validation | t 0-15 (80) / t 16-19 (20) | t 0-15 (80) / t 16-19 (20) |

**Schedules.**
* *Federated:* the per-round client schedule is exactly the FL baseline's (`RandomState(round)`).
* *Local:* each of the 5 FL round-0 clients runs 20 sequential local rounds. Time index 0 coincides with
  federated round 0, which has the same clients, start and data; only the seed namespace differs.
* *Temporal split* (both sets): time indices 0-15 train (80 snapshots), 16-19 validation (20 snapshots).

## 5. Representation statistics

The absolute state is almost entirely the shared random A initialisation.
* **Pooled energy:** A 25,961 vs B 111. The per-snapshot A RMS is 0.01285, close to the RMS of PEFT's
  Kaiming-uniform initialisation, 1/sqrt(3·2048) = 0.01276.
* **State similarity:** the mean pairwise state cosine is 0.997-1.000 (federated).
* **Innovation:** a client round moves the state by 2.0 % of its norm (federated; local 1.9 %). A and B move
  by equal energy (5.10 vs 5.23 pooled), elementwise about 1.5e-4 per Adam step, with a maximum of
  3.0e-4 = lr·(1 + 2/3 + 1/3) for a 3-step round.
* **Innovation energy fraction:** on the validation split, ρ = sum‖u‖² / sum‖x‖² = **3.6e-4**.
* **Error budget (DERIVED):** an error orthogonal to the innovation must satisfy state rel. sq. error
  <= (1/0.9² - 1)·ρ ≈ **8.5e-5** for the innovation cosine to reach 0.90. This is the precision the
  paper-literal (state) codec needs. The paper reports ‖G‖² = 14.29; at LLaMA-7B scale the raw PEFT A
  initialisation alone has about 341 (DR-10).
* **Sharing:** innovations are moderately shared within a round (cosine 0.41) and weakly across rounds (0.10
  federated).

## 6. Normalisation study (A2/A5; diagnostic variants, not paper behaviour)

| run | representation | mode | fitted scale(s) (D1 train only) | normalised \|v\| >= 1 on validation: A / B | best-val iteration | val MSE (training space) at it. 1 -> best | AE training time | decoder outputs with \|y\| > 0.99 |
|---|---|---|---|---|---|---|---|---|
| A5 `none` | adapter_state | none | – | 0 / 0 | 3000 | 0.016 -> 1.86e-04 | 478 s | 9.70e-05 |
| A5 `global_rms` | adapter_state | global_rms | s = 0.00909 | 0.590 / 0 | 3000 | 1.029 -> 0.979 | 455 s | 1.38e-04 |
| A5 `factor_rms` | adapter_state | factor_rms | s_A = 0.01283, s_B = 7.87e-04 | 0.422 / 0.463 | 3000 | 1.369 -> 1.316 | 456 s | 1.68e-04 |
| A8 fed. delta `factor_rms` | adapter_delta | factor_rms | s_A = 1.82e-04, s_B = 1.84e-04 | 0.385 / 0.373 | 2400 | 0.918 -> 0.902 | 411 s | 0 |
| A8 local state `none` | adapter_state | none | – | 0 / 0 | 3000 | 0.016 -> 1.85e-04 | 480 s | 9.16e-05 |

**Tanh range.** Under `global_rms`, 59 % of normalised A values lie outside the decoder's Tanh range. Under
`factor_rms`, 42 % of A values and 46 % of B values do. This caps achievable reconstruction even for a perfect
decoder: the `tanh_range_ceiling` gives A rel. sq. error 0.21 for `global_rms`, and A 0.080 / B 0.23 for
`factor_rms`, on validation. No clipping was applied; the overflow is measured.

**Training curves.** Every A5 candidate's best-D1-validation checkpoint is its final one (iteration 3000), but
the curves had flattened. Over the last 1,000 iterations validation MSE fell by:
* 1.0 % for `none`: 1.93e-4 at 1,500, 1.86e-4 at 3,000;
* 0.26 % for `global_rms`;
* 0.54 % for `factor_rms`.

Checkpoint selection on D1 validation therefore made no difference. The training budget is not the binding
constraint at the precision the gate requires.

## 7. AutoEncoder reconstruction study (validation, best-val AE, PHASE3-DIAGNOSTIC)

| run | predictor | A rel. sq. err. | A cos | B rel. sq. err. | B cos | B norm ratio | all rel. sq. err. | A / B p95 abs. err. | innovation cos | innovation rel. sq. err. | innovation norm ratio | agg.-update cos | agg.-update rel. L2 | agg.-update A / B norm ratio | B·A innovation cos | shift control matched / shifted |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A5 `none` | autoencoder_best_val | 1.650 | 8.02e-05 | 91.996 | -3.31e-04 | 9.539 | 2.217 | 0.021 / 0.002 | -0.013 | 6146.306 | 78.379 | -0.024 | 136.796 | 162.702 / 101.008 | -6.40e-04 | 6146.306 / 6146.306 |
| A5 `none` | zero | 1.000 | nan | 1.000 | nan | 0 | 1.000 | 0.021 / 0.002 | -0.019 | 2772.028 | 52.621 | -0.036 | 91.887 | 126.661 / 10.315 | -0.045 | 2772.028 / 2772.028 |
| A5 `none` | train_mean | 0.003 | 0.999 | 0.195 | 0.929 | 0.689 | 0.004 | 0.001 / 8.88e-04 | -0.127 | 11.514 | 3.118 | -0.242 | 5.766 | 6.377 / 4.187 | -0.069 | 11.514 / 11.514 |
| A5 `none` | tanh_range_ceiling | 0 | 1.000 | 0 | 1.000 | 1.000 | 0 | 0 / 0 | 1.000 | 0 | 1.000 | 1.000 | 0 | 1.000 / 1.000 | 1.000 | 0 / 1.845 |
| A5 `global_rms` | autoencoder_best_val | 0.952 | 0.234 | 3.004 | 0.013 | 1.429 | 0.965 | 0.021 / 0.004 | -0.019 | 2675.876 | 51.700 | -0.036 | 90.266 | 123.603 / 18.065 | -0.044 | 2675.876 / 2675.877 |
| A5 `global_rms` | zero | 1.000 | nan | 1.000 | nan | 0 | 1.000 | 0.021 / 0.002 | -0.019 | 2772.028 | 52.621 | -0.036 | 91.887 | 126.661 / 10.315 | -0.045 | 2772.028 / 2772.028 |
| A5 `global_rms` | train_mean | 0.003 | 0.999 | 0.195 | 0.929 | 0.689 | 0.004 | 0.001 / 8.88e-04 | -0.127 | 11.514 | 3.118 | -0.242 | 5.766 | 6.377 / 4.187 | -0.069 | 11.514 / 11.514 |
| A5 `global_rms` | tanh_range_ceiling | 0.211 | 0.957 | 6.92e-16 | 1.000 | 1.000 | 0.210 | 0.012 / 5.82e-11 | 0.009 | 581.811 | 24.109 | -0.021 | 42.096 | 58.175 / 1.000 | 0.166 | 581.811 / 583.120 |
| A5 `factor_rms` | autoencoder_best_val | 0.971 | 0.170 | 0.972 | 0.168 | 0.142 | 0.971 | 0.022 / 0.002 | -0.019 | 2692.195 | 51.858 | -0.036 | 90.524 | 124.781 / 10.165 | -0.045 | 2692.195 / 2692.195 |
| A5 `factor_rms` | zero | 1.000 | nan | 1.000 | nan | 0 | 1.000 | 0.021 / 0.002 | -0.019 | 2772.028 | 52.621 | -0.036 | 91.887 | 126.661 / 10.315 | -0.045 | 2772.028 / 2772.028 |
| A5 `factor_rms` | train_mean | 0.003 | 0.999 | 0.195 | 0.929 | 0.689 | 0.004 | 0.001 / 8.88e-04 | -0.127 | 11.514 | 3.118 | -0.242 | 5.766 | 6.377 / 4.187 | -0.069 | 11.514 / 11.514 |
| A5 `factor_rms` | tanh_range_ceiling | 0.080 | 0.980 | 0.232 | 0.935 | 0.610 | 0.081 | 0.009 / 0.001 | 0.012 | 224.327 | 14.956 | -0.027 | 26.129 | 35.771 / 4.920 | 0.073 | 224.327 / 225.358 |

Identity (the ceiling) gives 0 error, cosine 1 and norm ratio 1 in every group. Training-split numbers are in the JSON files: the AE is equally poor there, and it is not overfitting.

What the numbers say:

* **`none` (paper-literal) is worse than predicting zero.** A error is 1.65, B error 92, and the B norm is
  9.5x too large. The output barely depends on the input: the shift control gives 6146.306 for both pairings.
  Its squared distance from the train mean is 530x that of the inputs, so it is a near-constant wrong
  output.
* **Normalisation brings the A error (and, under `factor_rms`, the B error) just below 1, and nothing else.**
  `global_rms` reaches A 0.952 (B stays at 3.0); `factor_rms` reaches A 0.971 and B 0.972. The B norm ratio is 1.43 for `global_rms`, while under
  `factor_rms` B collapses (0.142). Both outputs carry no information about the client update:
  * innovation cosine -0.019;
  * aggregate-update cosine -0.036;
  * B·A innovation cosine about -0.04;
  * shift control matched ≈ shifted.
* **`train_mean`, an input-independent decoder, shows the metric blind spot.** State error is 0.0042 overall
  (A 0.003), yet the innovation cosine is -0.13 and the aggregate-update cosine -0.24.

## 8. A6 gate (pre-registered minimum viability; validation, best-val AE)

| run | G1 finite | G2 A < 1 | G3 B < 1 | G4 innov. cos >= 0.90 | G5 B ratio in [0.5, 2] | G6 agg.-update cos >= 0.90 | G7 input-dependent | **overall** |
|---|---|---|---|---|---|---|---|---|
| A5 `none` | PASS | FAIL (1.650) | FAIL (91.996) | FAIL (-0.013) | FAIL (9.539) | FAIL (-0.024) | FAIL (innov. 6146.306 vs train-mean 11.514; shift 6146.306 vs 6146.306) | **FAIL** |
| A5 `global_rms` | PASS | PASS (0.952) | FAIL (3.004) | FAIL (-0.019) | PASS (1.429) | FAIL (-0.036) | FAIL (innov. 2675.876 vs train-mean 11.514; shift 2675.876 vs 2675.877) | **FAIL** |
| A5 `factor_rms` | PASS | PASS (0.971) | PASS (0.972) | FAIL (-0.019) | FAIL (0.142) | FAIL (-0.036) | FAIL (innov. 2692.195 vs train-mean 11.514; shift 2692.195 vs 2692.195) | **FAIL** |

**Priority rule:** none -> global_rms -> factor_rms gives **no selection**. The verdict is NO PRIMARY CODEC IS
VIABLE (`results/phase3/ae_viability/a6_selection.json`). The paper-style SNR is reported, never gated. It is
1.4e6 to 3.3e6, against about 1e12 implied by the paper's tables. It scales with the element count, so it is
not comparable across geometries.

## 9. A7: one-round D2 FAF probe

Not run, by the pre-registered rule: it runs only if A6 selects a candidate.

## 10. A8: sensitivity diagnostics (PHASE3-SENSITIVITY)

| run | predictor | A rel. sq. err. | A cos | B rel. sq. err. | B cos | B norm ratio | all rel. sq. err. | A / B p95 abs. err. | innovation cos | innovation rel. sq. err. | innovation norm ratio | agg.-update cos | agg.-update rel. L2 | agg.-update A / B norm ratio | B·A innovation cos | shift control matched / shifted |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A8 fed. delta `factor_rms` | autoencoder_best_val | 1.000 | 3.17e-04 | 1.000 | -3.21e-05 | 4.11e-04 | 1.000 | 2.25e-04 / 2.25e-04 | 1.42e-04 | 1.000 | 4.09e-04 | 3.23e-04 | 1.000 | 6.76e-04 / 7.16e-04 | -5.30e-05 | 1.000 / 1.000 |
| A8 fed. delta `factor_rms` | zero | 1.000 | nan | 1.000 | nan | 0 | 1.000 | 2.25e-04 / 2.25e-04 | nan | 1.000 | 0 | nan | 1.000 | 0 / 0 | nan | 1.000 / 1.000 |
| A8 fed. delta `factor_rms` | train_mean | 1.109 | 0.145 | 1.088 | 0.039 | 0.339 | 1.099 | 3.12e-04 / 2.86e-04 | 0.101 | 1.099 | 0.431 | 0.189 | 1.132 | 0.864 / 0.605 | 0.048 | 1.099 / 1.099 |
| A8 fed. delta `factor_rms` | tanh_range_ceiling | 0.020 | 0.994 | 0.017 | 0.995 | 0.913 | 0.018 | 4.30e-05 / 4.04e-05 | 0.995 | 0.018 | 0.909 | 0.994 | 0.153 | 0.878 / 0.892 | 0.995 | 0.018 / 1.682 |
| A8 local state `none` | autoencoder_best_val | 1.598 | 7.80e-04 | 37.402 | -4.73e-04 | 6.033 | 2.141 | 0.022 / 0.003 | -0.082 | 6027.586 | 77.549 | -0.105 | 125.162 | 134.769 / 106.005 | -0.009 | 6027.586 / 6022.144 |
| A8 local state `none` | zero | 1.000 | nan | 1.000 | nan | 0 | 1.000 | 0.022 / 0.003 | -0.121 | 2815.028 | 52.926 | -0.154 | 85.667 | 106.849 / 12.408 | -0.380 | 2815.028 / 2809.578 |
| A8 local state `none` | train_mean | 0.012 | 0.994 | 0.646 | 0.627 | 0.429 | 0.021 | 0.003 / 0.003 | -0.486 | 60.413 | 7.237 | -0.635 | 8.190 | 8.085 / 6.414 | -0.430 | 60.413 / 54.964 |
| A8 local state `none` | tanh_range_ceiling | 0 | 1.000 | 0 | 1.000 | 1.000 | 0 | 0 / 0 | 1.000 | 0 | 1.000 | 1.000 | 0 | 1.000 / 1.000 | 1.000 | 0 / 1.700 |

| run | G1 finite | G2 A < 1 | G3 B < 1 | G4 innov. cos >= 0.90 | G5 B ratio in [0.5, 2] | G6 agg.-update cos >= 0.90 | G7 input-dependent | **overall** |
|---|---|---|---|---|---|---|---|---|
| A8 fed. delta `factor_rms` | PASS | PASS (1.000) | FAIL (1.000) | FAIL (1.42e-04) | FAIL (4.11e-04) | FAIL (3.23e-04) | PASS (innov. 1.000 vs train-mean 1.099; shift 1.000 vs 1.000) | **FAIL** |
| A8 local state `none` | PASS | FAIL (1.598) | FAIL (37.402) | FAIL (-0.082) | FAIL (6.033) | FAIL (-0.105) | FAIL (innov. 6027.586 vs train-mean 60.413; shift 6027.586 vs 6022.144) | **FAIL** |

* **`local_pretrain + adapter_state` (`none`) fails the same way as the primary.**
  * A 1.60, B 37.4, B norm ratio 6.0;
  * innovation cosine -0.082, aggregate-update cosine -0.105;
  * input-independent output: the shifted pairing even scores slightly better (6022.1 vs 6027.6).

  The TGAP-source interpretation does not change the conclusion for the state representation.
* **`federated_pretrain + adapter_delta` (`factor_rms`): the output range is not the obstacle.** The scales are s_A = 1.82e-4 and s_B = 1.84e-4, the per-element Adam step size. 38 % / 37 % of normalised values lie outside the Tanh range, yet the Tanh-range ceiling still reaches innovation cosine 0.995.
* **The delta AE learned to output almost nothing.** Its best-D1-validation checkpoint is iteration 2400 (validation MSE 0.918 -> 0.902 in the normalised space). On validation:
  * the A and B relative squared errors equal the zero predictor's within 2e-7 (G2 passes by 9e-8, G3 fails by 2e-7);
  * B norm ratio 4.1e-4, innovation cosine 1.4e-4, aggregate-update cosine 3.2e-4.
* **G7 passes formally under the strict pre-registered inequality**, but the margin is 4e-8 in relative squared error. That is no practically meaningful input dependence. The input-independent train-mean predictor does better on deltas (innovation cosine 0.10) than the AE.

## 11. Reference codes at the AE's element ratio (DERIVED interpretation aid, validation, never a gate)

| data (validation) | code | A rel. sq. err. | B rel. sq. err. | B norm ratio | all rel. sq. err. | innovation cos | innovation rel. sq. err. | agg.-update cos | B·A innovation cos |
|---|---|---|---|---|---|---|---|---|---|
| federated delta | lowpass_dct | 0.983 | 0.984 | 0.128 | 0.984 | 0.128 | 0.984 | 0.130 | 0.130 |
| federated delta | topk_dct | 0.879 | 0.881 | 0.347 | 0.880 | 0.346 | 0.880 | 0.394 | 0.351 |
| federated delta | topk_raw | 0.991 | 0.963 | 0.192 | 0.977 | 0.151 | 0.977 | 0.188 | 0.196 |
| federated delta | train_pca | 0.871 | 0.901 | 0.394 | 0.886 | 0.343 | 0.886 | 0.476 | 0.332 |
| federated state | lowpass_dct | 0.980 | 1.700 | 0.857 | 0.984 | -0.019 | 2728.711 | -0.035 | -0.044 |
| federated state | topk_dct | 0.832 | 8.616 | 2.802 | 0.881 | -0.015 | 2441.397 | -0.032 | -0.034 |
| federated state | topk_raw | 0.900 | 1.000 | 0 | 0.901 | -0.007 | 2497.531 | -0.021 | -0.045 |
| federated state | train_pca | 2.38e-04 | 0.038 | 0.990 | 4.72e-04 | 0.150 | 1.310 | 0.150 | 0.156 |
| local state | lowpass_dct | 0.980 | 1.276 | 0.554 | 0.984 | -0.120 | 2771.019 | -0.152 | -0.377 |
| local state | topk_dct | 0.832 | 4.027 | 1.807 | 0.881 | -0.111 | 2479.143 | -0.144 | -0.339 |
| local state | topk_raw | 0.897 | 1.000 | 0 | 0.899 | -0.095 | 2529.687 | -0.123 | -0.380 |
| local state | train_pca | 3.36e-04 | 0.023 | 0.995 | 6.83e-04 | 0.203 | 1.924 | 0.348 | 0.194 |

* **State representations (federated and local).** Every generic code at 1/64 (low-pass DCT, best-k DCT, best-k raw) loses 83-98 % of the A energy, and its innovation cosine is at or below zero (-0.12 to -0.007). The training-span decoder reconstructs the state to 4.7e-4 (federated) / 6.8e-4 (local), but the validation-round innovation only to cosine 0.15 / 0.20.
* **Delta representation.** The update itself is close to incompressible at 1/64 for generic codes:
  * a fixed low-pass DCT keeps 1.6 % of its energy (innovation cosine 0.13);
  * the optimistic best-k DCT reaches cosine 0.35;
  * the training-span decoder reaches 0.34 (aggregate-update cosine 0.48).

  None approaches 0.90.
* **What this does and does not show.** These are reference codes, neither the paper's method nor optimal codes. They bound how much simple transform or linear structure exists at a 64x element reduction of our Tier-B updates. The AE results above are consistent with those bounds.

## 12. Micro-batch loss-normalisation diagnostic (PHASE3-DIAGNOSTIC)

| micro-batch | cos vs micro-batch 1 (ours) | rel. L2 diff. vs micro-batch 1 | cos vs full-batch token mean | rel. L2 diff. vs token mean | per-example weight max/min |
|---|---|---|---|---|---|
| 1 (ours) | 1.0000 | 0.000 | 0.8444 | 0.713 | 1.00 |
| 2 | 0.9432 | 0.346 | 0.9408 | 0.380 | 3.81 |
| 4 | 0.8930 | 0.462 | 0.9742 | 0.234 | 5.62 |
| 8 | 0.8788 | 0.488 | 0.9792 | 0.207 | 7.25 |
| 16 (paper) | 0.8761 | 0.493 | 0.9839 | 0.182 | 8.79 |
| 32 | 0.8444 | 0.542 | 1.0000 | 0.000 | 6.81 |

Fixed batch: the first optimizer step of FL round 0 for client 26 (32 examples, 75-511 label tokens each). Adapter: federated-TGAP global at the start of round 10. Real trainer path (micro-batch 1) vs unpadded emulation: cosine 0.99945, relative L2 0.0355, 25/32 examples left-padded. Gradients are taken before clipping and AdamW.

The trainer's normalisation is token-mean per micro-batch, then the mean over the accumulation group. It is
unchanged and applies to all arms. Under the forced micro-batch 1, every example's mean token loss gets equal
weight.

On one fixed real batch, the accumulated gradient at micro-batch 1 has cosine **0.876** (relative difference
0.49) with the paper's micro-batch 16, and 0.844 with the full-batch token mean. At micro-batch 16, per-example
weights span 8.8x; at micro-batch 1 they are uniform. This is a material optimisation difference from the
paper's setting, reported rather than corrected.

Separately, left padding makes the last pad position of a padded example predict its first real token. This is
inherited HF/Shepherd label-shift behaviour. It affected 25 of 32 examples and accounts for most of the 3.5 %
difference between the real trainer path and the unpadded emulation.

## 13. Interpretation

**Supported by the evidence:**
* The Phase-2 smoke failure was not a small-scale artefact. On real Tier-B snapshots, the reconstructed
  ResNet-3 AE at 1/64 does not transmit the client's update, with or without either normalisation.
* Scale imbalance is real: A carries 234x the energy of B. Normalisation therefore changes which
  factor-level criteria pass, but it does not explain the failure.
* The binding constraint is the transmitted representation:
  * the absolute state is about 99.96 % shared initialisation plus history;
  * the update FAF needs is 3.6e-4 of its energy;
  * generic 1/64 codes keep at most 17 % of the A energy;
  * even a decoder that memorises the training span reconstructs the state to 4.7e-4 yet recovers only 0.15
    of the validation-round innovation direction.
* The Tanh output head is incompatible with RMS-normalised inputs: 42-59 % of A values are out of range.

**Not reproduced:**
* any viable CG-FedLLM compressor on our Tier-B configuration;
* the paper's near-perfect reconstruction (paper-style SNR about 1e12, ‖G‖² = 14.29).

The claims of equal or better C-Eval/MMLU than LoRA-FT were not tested, because Phase 3B was not entered.

**Still ambiguous:**
* state vs delta: the paper's ‖G‖² of 14.29 is far below a raw PEFT A initialisation (DR-10);
* local vs federated TGAP;
* the AE details (stem, normalisation, budget);
* whether the paper used a different LoRA initialisation or base precision;
* the evaluation protocol;
* Cent;
* LoRA alpha/dropout;
* the aggregation notation;
* the checkpoint policy;
* the micro-batch normalisation;
* DR-01 to DR-21.

## 14. Resource accounting (seed 1)

| Stage (seed 1) | Wall time | GPU memory (measured) | Storage | Outcome |
|---|---|---|---|---|
| Phase-2 GPU smoke regression (llama-160m) | 62 s | - | 118 MB (outside Git) | all 6 reviewed adapter hashes identical |
| Determinism check (Qwen1.5-1.8B, 2 x 1 step) | < 1 min | 4.64 GiB peak allocated | - | bitwise identical |
| A3 calibration, micro-batch 2 | 128 s | 5.405 / 6.596 GiB allocated / reserved | 7 KB (committed) | rule triggered |
| A3 calibration, micro-batch 1 | 156 s | 4.487 / 6.463 GiB | 7 KB (committed) | kept |
| A4 federated TGAP (D1, 20 rounds x 5 clients) | 1,197 s | trained as A3 micro-batch 1 (not separately instrumented) | 1.7 GB (outside Git) | 100 snapshots |
| A4 local TGAP (5 clients x 20 steps) | 1,827 s | as above | 2.4 GB (outside Git) | 100 snapshots |
| Micro-batch diagnostic | 45 s | - | 8 KB | - |
| A5 AE training (`none` / `global_rms` / `factor_rms`) | 529 / 502 / 510 s (training loop 478 / 455 / 456 s) | about 1.2 GiB (nvidia-smi, incl. context) | 16 MB each (outside Git) | all finite |
| A5 viability evaluation (3) | 186 / 166 / 180 s | - | 140-144 KB each | gate FAIL x3 |
| A8 AE training (local state / federated delta) | 529 / 460 s | about 1.2 GiB | 16 MB each | all finite |
| A8 viability evaluation (2) | 166 / 163 s | - | 140-148 KB each | gate FAIL x2 |
| TGAP statistics (2) / reference codes (3) | about 20 s each / 71, 43, 42 s (CPU) | - | 265 KB each / 30 KB each | - |
| **Total** | **about 1.9 h of stage wall time** | peak allocated 4.49 GiB (training), cap 6.689 GiB | 4.1 GB outside Git | no run failed; one transient TLS push error (retried); one CI failure (`eb771bb`, fixed) |

## 15. Deviations, negative results and provenance notes

**Deviations and negative results:**
* *A3 rule triggered:* micro-batch 2 became micro-batch 1 x 32 accumulation (pre-registered contingency,
  `cd4c5c4`).
* *Negative results:* every A5 candidate failed A6, and A7 was not run. A8 results are in section 10.
* *CI failure on `eb771bb`:* the CPU CI lacked `scipy` (collection error in `test_compressibility.py`). Fixed in
  `1429a70` by declaring the version already in the verified lock.
* *Platform throttling:* the software power cap and thermal slowdown reduced sustained throughput (section 3).
* *Read-side label migration* for one annotated Phase-2 label; no reviewed evidence was rewritten.

**Provenance notes:**
* The micro-batch-1 calibration, both TGAP collections, the micro-batch diagnostic and the `none` AE ran while
  `configs/phase3/tierb_seed1.yaml` held the uncommitted micro-batch switch.
* Some of these runs also had an uncommitted docs edit and the then-uncommitted reference-code module, which is
  not on their code path.
* Each run's resolved config hash equals the committed config's hash:
  * `44c98e715a00`: base, also the A5 `none` AE;
  * `81a131495901`: local, also the A8 local-state AE;
  * `e430a297ea4b`: `global_rms`;
  * `20db958d84e2`: `factor_rms`;
  * `226be05cda8a`: delta.
* The later AE runs record clean commits (`cd4c5c4`, `eb771bb`, `1429a70`), with at most uncommitted docs. The
  AE training code has not changed since `cabc3d7`.
* The statistics and reference-code files were regenerated from the clean commit `cd4c5c4` and are identical.

## 16. Commands (seed 1; `CGFED_RUNS=D:\cgfed-runs`, `HF_HUB_OFFLINE=1`)

```
cgfed calibrate-train --config configs/phase3/tierb_seed1.yaml --stage a3_calibration       # micro-batch 2 (then switched)
cgfed calibrate-train --config configs/phase3/tierb_seed1.yaml --stage a3_calibration_mb1
cgfed collect-tgap    --config configs/phase3/tierb_seed1.yaml   --stage tgap_federated_state
cgfed collect-tgap    --config configs/phase3/a4_tgap_local.yaml --stage tgap_local_state
cgfed microbatch-diag --config configs/phase3/tierb_seed1.yaml --stage microbatch_diag \
      --adapter $CGFED_RUNS/phase3_tierb_qwen15_1p8b_seed1/tgap_federated_state/fl/rounds/r0009/global_adapter.safetensors
cgfed tgap-stats      --config <set config> --snapshots <set dir> --out results/phase3/tgap/<set>_seed1_stats.json
for m in none global_rms factor_rms:
  cgfed train-ae      --config configs/phase3/a5_ae_$m.yaml --snapshots <federated dir> --stage ae_fed_state_$m
  cgfed ae-viability  --config configs/phase3/a5_ae_$m.yaml --snapshots <federated dir> --ae-dir <run>/ae_fed_state_$m \
                      --out results/phase3/ae_viability/a5_federated_state_$m.json
cgfed ae-select --gate none=... --gate global_rms=... --gate factor_rms=... --label PHASE3-DIAGNOSTIC --out results/phase3/ae_viability/a6_selection.json
cgfed train-ae / ae-viability with configs/phase3/a8_ae_local_state.yaml and configs/phase3/a8_ae_federated_delta_factor_rms.yaml
cgfed reference-codes --config <config> --snapshots <dir> --out results/phase3/reference_codes/<name>.json
```
