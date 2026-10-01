# Phase 3A pre-registration: Tier-B AutoEncoder diagnosis and viability gate

This document fixes every protocol, metric, threshold and decision rule of Phase 3A. It was committed **before**
any Phase-3 GPU data (calibration, TGAP snapshots, AE training) existed. Thresholds are the reviewer's
**minimum viability gates**, not paper reproduction targets. No C-Eval, MMLU or D2 result is used to choose any
compressor setting; the only model selection is the AE's best-D1-validation checkpoint (D9).

Result labels: `PHASE3-DIAGNOSTIC` (primary-hypothesis diagnostics), `PHASE3-SENSITIVITY` (local TGAP, delta
representation), `DERIVED` (statistics computed from recorded data), `PHASE3-TIERB-CORE` (Phase 3B only).

## 1. Fixed configuration (`configs/phase3/tierb_seed1.yaml`)

| Item | Value | Status |
|---|---|---|
| Model | `Qwen/Qwen1.5-1.8B@7846de7e`, bf16, no quantization, SDPA, gradient checkpointing | RESOURCE-FEASIBLE; the paper appears to use an 8-bit base |
| Local training | 1 epoch, lr 1.5e-4, effective batch 32 = micro-batch 2 x 16 accumulation, AdamW fresh per invocation, linear decay, clip 1.0, cutoff 512 | paper values except micro-batch (paper: 16) |
| LoRA | r 8, alpha 16, dropout 0.05, q/k/v/o | alpha/dropout INFERRED from Shepherd |
| FL | 100 Dirichlet(0.5) clients, fraction 0.05 (K = 5), 20 rounds, sample-weighted FedAvg, sequential clients | Shepherd baseline semantics |
| Seed | scientific seed 1: `run.seed = lora.init_seed = autoencoder.init_seed = 1` | partition/D1-D2 split (42) and client sampling (`RandomState(round)`) are seed-independent |
| Determinism | `torch.use_deterministic_algorithms(True)` | verified bitwise-repeatable for bf16 + SDPA + checkpointing on this GPU |
| Memory guard | allocator capped at free VRAM - 256 MiB before loading | WDDM spill guard (P2-D5) |

## 2. A3: realistic-sequence calibration (`cgfed calibrate-train`)

Token lengths after the cutoff for all D1 and D2 examples, and the exact real/padded token totals of the FL
(D2), federated-TGAP (D1) and local-TGAP (D1) schedules in the trainer's data order at micro-batch 1 and 2.
Bounded real training is one FL round 0 on D2: the 5 clients selected by the sampler, the real trainer and
seeds, and the initial adapter. It is followed by one forced worst-case micro-batch made of the longest training
examples.

**Decision rule (fixed):** switch the primary configuration to bf16 + GC + **micro-batch 1** if the peak
allocated memory of any measurement exceeds **6.3 GiB**, or if the peak reserved memory comes within **256 MiB**
of the allocator cap in **two or more** measurements. Otherwise keep micro-batch 2. Quantization is not
considered unless bf16 is unstable.

## 3. A4: TGAP collection (seed 1, D1 only, snapshot payloads outside Git)

* **PRIMARY:** `federated_pretrain + adapter_state`. Uncompressed FL on D1 for 20 rounds with the FL sampler's
  schedule (K = 5 per round) and sample-weighted FedAvg, giving 100 client-round snapshots. Each snapshot stores
  the client's post-training state and its round-start global state.
* **SENSITIVITY:** `local_pretrain + adapter_state`. Each of the 5 clients selected in FL round 0 runs 20
  sequential local rounds on its own D1. It starts from the same initial adapter and continues from its own
  previous state, with a fresh optimizer each round and no aggregation, giving 100 snapshots. Time index 0
  matches the federated schedule; later time indices cannot follow the per-round FL schedule without changing
  the implemented local semantics.
* **Statistics (`cgfed tgap-stats`, DERIVED, committed):**
  * counts: snapshots, unique clients, participation histogram;
  * per-snapshot A/B state norms, RMS and max;
  * pooled |value| quantiles;
  * innovation norms and innovation relative to state;
  * cosine similarity: inter-client within a time index, same client across time, and between the
    sample-weighted mean innovations of time indices;
  * the train/validation membership and the snapshot manifest with hashes.

## 4. A5: AutoEncoder protocol (identical for every candidate)

* **Fixed:**
  * the reconstructed ResNet-3 architecture;
  * latent `[64, d/64, W/64]`, compression ratio exactly 1/64. For Qwen1.5-1.8B the Phi input is
    `[1, 2048, 1536]` and the latent is `[64, 32, 24]`.
* **Optimisation:** MSE; Adam (lr 2e-4, betas 0.9/0.999, eps 1e-8, no weight decay); batch 4; 3000 iterations
  with no early stopping; AE init seed 1; epoch-wise batch permutations from seed 1.
* **Split:** temporal. The first 80 % of time indices train (rounds 0-15, 80 snapshots); the last 20 %
  validate (rounds 16-19, 20 snapshots).
* **Checkpoints:** evaluated every 50 iterations on the validation split in eval mode, using BatchNorm running
  statistics. Two checkpoints are saved:
  * `autoencoder.safetensors`, the final iteration;
  * `autoencoder_best_val.safetensors`, the lowest validation MSE in the AE's training space among
    iterations 50, 100, ..., 3000, with ties going to the earliest.

  The best-validation checkpoint is the one gated and probed.
* **Ladder (fixed order, no per-candidate tuning)** on the PRIMARY data:
  1. `none`: paper-literal, no arithmetic.
  2. `global_rms`: one scalar s = RMS of every element of the training split; encode `x/s`, decode `s * dec(z)`.
  3. `factor_rms`: s_A and s_B = RMS of all A (resp. B) elements of the training split, applied per Phi column
     block.

  Scales are fitted on the training split only, frozen, and stored in the checkpoint metadata. They add 0
  uplink bytes. There is no clipping: the fraction of normalised values with |v| >= 1 (outside the decoder's
  Tanh range) is reported per factor.
* **Reference predictors** on both splits:
  * `zero`;
  * `train_mean`: the mean of the training split, i.e. an input-independent decoder;
  * `identity`: the ceiling;
  * `tanh_range_ceiling`: `denorm(clamp(norm(x), -1, 1))`, the best any Tanh-headed decoder can do under the
    fitted normalisation (DERIVED).

## 5. Metrics (A1; `cg_fedllm/compression/diagnostics.py`)

All sums are pooled over the snapshots of a split, in float64. For vectors v (truth) and v_hat:

| Quantity | Definition |
|---|---|
| MSE | sum ‖v_hat - v‖² / n |
| pooled relative squared error | sum ‖v_hat - v‖² / sum ‖v‖² |
| cosine | sum ⟨v_hat, v⟩ / sqrt(sum ‖v_hat‖² · sum ‖v‖²); undefined (fails any gate) if either is 0 |
| norm ratio | sqrt(sum ‖v_hat‖² / sum ‖v‖²) |
| max abs error | over all elements |
| p50 / p95 abs error | exact quantiles; a strided subsample above 2^25 elements, flagged |

**Groups**, each reported for A, B and A+B:
* **transmitted:** the representation the codec sees, i.e. the state or the delta;
* **state:** the recovered post-training adapter (`G + delta_hat` for delta);
* **innovation:** recovered state minus the client's round-start state. In `federated_pretrain` that is the
  round-start global G_t; in `local_pretrain` it is the client's own previous state.

**Product:** the effective change B·A of every q/k/v/o module, for the state and for the innovation
(`B A - B_s A_s`). It is computed exactly from r x r Gram matrices.

**Offline FedAvg replay:**
* For each time index t, `U_t = sum_i w_i (e_i - s_i)` and `U_hat_t = sum_i w_i (e_hat_i - s_i)`, with Shepherd's
  L1-normalised sample weights.
* Pooled over the time indices of the split, the replay reports:
  * relative L2 error `sqrt(sum ‖U_hat - U‖² / sum ‖U‖²)`;
  * cosine;
  * norm ratios for A and B.

  The same metrics are reported for the aggregated state.

**Shift control:** the innovation error when each reconstruction is paired with the true innovation of the next
client of the same time index, cyclically by client ID. An input-independent decoder scores identically under
both pairings.

**Paper-style SNR** (reported, never gated): mean per-snapshot signal energy / pooled per-element MSE.

## 6. A6: reconstruction gate and selection rule

A candidate passes only if **all** of the following hold on the D1 temporal **validation** split for its
best-validation AE:

| # | Criterion | Definition |
|---|---|---|
| G1 | finite outputs | every reconstruction finite and every gated metric finite |
| G2 | A pooled relative squared error < 1.0 | transmitted representation, A factors |
| G3 | B pooled relative squared error < 1.0 | transmitted representation, B factors |
| G4 | innovation cosine >= 0.90 | pooled, A+B |
| G5 | reconstructed B norm ratio in [0.5, 2.0] | transmitted representation, B factors |
| G6 | aggregate-update cosine >= 0.90 | offline FedAvg replay, pooled, A+B |
| G7 | input-dependent | pooled innovation relative squared error **below** the `train_mean` predictor's **and below** the shift-control pairing's |

"Transmitted" means the state for `adapter_state` and the delta for `adapter_delta`. For delta, a state-level
A/B error would be trivially small, because the exact G is added back; it is reported but not gated.

**Selection (priority rule):** select `none` if it passes; otherwise `global_rms` if it passes; otherwise
`factor_rms` if it passes; otherwise **NO PRIMARY CODEC IS VIABLE**. Downstream loss is never consulted.

## 7. A7: one-round D2 FAF probe (only if A6 selects a candidate)

The probe is one real FL round 0 on D2 from the initial adapter, with the same selected clients, data and seeds,
for each of three codecs:
* `IdentityCodec`;
* the selected AE (best-validation checkpoint and its normaliser);
* `ConstantMeanCodec`, which returns the D1-train mean.

The AE passes the operational gate only if:
* all local and held-out losses are finite;
* the aggregated server update (new global minus old global) has cosine >= 0.90 against Identity's;
* that update's relative L2 error against Identity's is <= 0.50;
* the new global B norm ratio against Identity is in [0.5, 2.0];
* the held-out Dolly loss is <= Identity's held-out loss + 0.50;
* the AE beats ConstantMean on both the aggregated-update relative L2 error and the pooled per-client
  innovation relative squared error, each lower by at least 1 % relative.

If the probe fails, no 20-round FAF is run: A8 follows, then Phase 3 stops before 3B. The gates are never
weakened.

## 8. A8: sensitivity diagnostics (PHASE3-SENSITIVITY, run regardless of A6)

* `local_pretrain + adapter_state` with the mode selected in A6. If A6 selects no mode, the paper-literal `none`
  is used. One run only; no further modes are searched.
* `federated_pretrain + adapter_delta` with `factor_rms`.

Both use the A5 protocol and are evaluated against the A6 criteria. They never redefine the primary
interpretation.

**Stop rules:**
* If no primary state candidate passes A6/A7, Phase 3 stops before 3B.
* If only the delta representation passes, that is reported exactly as such.

## 9. Micro-batch loss-normalisation diagnostic (bounded)

The trainer's normalisation is fixed for every arm and is not changed: token-mean cross-entropy per
micro-batch, averaged over the micro-batches of an accumulation group.

* **Batch:** one fixed batch of 32 examples, namely the first optimizer step of FL round 0 for the round-0
  client with the most D2 examples, in the trainer's data order.
* **Adapter:** the federated-TGAP global at the start of round 10, so A and B both have gradients.
* **Measurement:** LoRA dropout is disabled for the measurement only. Per-example token-sum gradients of the
  unpadded examples emulate micro-batch 1, 2, 4, 8, 16 and 32 exactly. The diagnostic reports:
  * cosine and relative L2 against micro-batch 2 (ours), micro-batch 16 (the paper's) and the full-batch token
    mean;
  * the per-example weight spread;
  * the real micro-batch-2 accumulation for comparison.

  With left padding the last pad position predicts the first real token: this is inherited HF/Shepherd
  label-shift behaviour, and it adds one label token per padded example in the real path.

## 10. Phase 3B entry

Phase 3B starts only if `federated_pretrain + adapter_state` passes both A6 and A7. Before any 20-round run,
the following are committed: the protocol, the selected normalisation, the seeds, the experiment matrix and
the evaluation schedule.

## 11. Mandatory interpretation rules

**Never describe:**
* Qwen1.5-1.8B as paper-faithful;
* bf16 as equivalent to the paper's 8-bit base;
* `global_rms` or `factor_rms` as paper-specified;
* sample-weighted FedAvg as specified by CG-FedLLM;
* `federated_pretrain` as conclusively the paper's TGAP;
* `cent_resource_matched` as the paper's Cent.

If only a normalised variant passes, the conclusion reads: *"The paper-literal reconstruction was not viable
under our recovered implementation; a documented normalisation variant was viable."*
