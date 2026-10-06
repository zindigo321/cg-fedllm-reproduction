# Deviations and inferred choices

Every departure from the paper or from Shepherd's behaviour, and every choice the paper does not specify.
Labels: **PAPER-SILENT** (our choice where the paper says nothing), **SHEPHERD-DEVIATION** (differs from the
reconstructed baseline), **ENVIRONMENT** (forced by hardware/software), **SCOPE** (deferred by the reviewer).

## Reproduction semantics

| # | Topic | Our choice | Paper / Shepherd | Label | Configurable |
|---|---|---|---|---|---|
| 1 | LoRA A initialisation | seeded (`derive_seed(lora.init_seed, "lora_init")`), identical across TGAP/FAF/LoRA-FT | Shepherd leaves it unseeded | SHEPHERD-DEVIATION | `lora.init_seed` |
| 2 | Data order / dropout RNG | derived from `(run.seed, namespace, round, client)` | Shepherd: global RNG state (datasets `shuffle()` + Trainer seed 42) | SHEPHERD-DEVIATION (makes resume bit-identical) | `run.seed` |
| 3 | Gradient accumulation | gradient = mean over the micro-batches of each group; a partial final group is flushed at the epoch end | 2023 HF Trainer divided by the nominal count and did not flush | SHEPHERD-DEVIATION | `local_train.accumulation_normalization`, `partial_accumulation` |
| 4 | Loss reduction | token-mean cross-entropy per micro-batch | Trainer default (same in 2023) | — | `local_train.loss_reduction` |
| 5 | Client processing / summation order | ascending client ID | Shepherd iterates a Python `set` | SHEPHERD-DEVIATION (float summation order only) | — |
| 6 | Position IDs with left padding | explicit `cumsum(mask) - 1` | Shepherd relied on model defaults (shift-invariant under RoPE) | SHEPHERD-DEVIATION (numerically negligible) | — |
| 7 | Base precision in the smoke | fp32 llama-160m | paper/Shepherd: 8-bit 7B | ENVIRONMENT / SCOPE | `model.dtype`, `model.quantization` |
| 8 | D1/D2 granularity | per client after partitioning, 30/70, `floor(0.3 n + 0.5)` | paper: 3:7, granularity not stated | PAPER-SILENT (R7) | `data.d1_fraction`, `data.d1d2_split_seed` |
| 9 | Transmitted representation | both `adapter_state` and `adapter_delta`; baselines use `adapter_state` | paper ambiguous | PAPER-SILENT (R2) | `federated.representation` |
| 10 | Phi block ordering | `layer_major_qkvo_AtB` | shape stated, ordering not | PAPER-SILENT (R4) | `codec.layout`, `tgap.layout` |
| 11 | TGAP source | both modes; smoke default `local_pretrain` | ECAI/v3 vs v1 wording | PAPER-SILENT (R6) | `tgap.source` |
| 12 | AE details | 3x3 stem, bias-free convs before BN, six (not seven) ConvT, reflection padding in residual blocks | table/diagram partly inconsistent | PAPER-SILENT (parameter/MAC counts match Table 3) | `autoencoder.*` |
| 13 | AE training | batch 4 (smoke 2), 600 iterations, Adam betas (0.9, 0.999), temporal split 80/20, no normalisation, final-iteration checkpoint | only MSE/Adam/2e-4 stated | PAPER-SILENT | `autoencoder.*` |
| 14 | Cent | `cent_smoke_sample_matched` (one pooled client, same local recipe, K*T/N = 1 pass) — **not** the paper's Cent | unknown | PAPER-SILENT (R13) | `federated.pooled` |
| 15 | Checkpoint policy | final round only | paper reports selected epochs | PAPER-SILENT (R14) | — |
| 16 | Diverged runs | recorded as `diverged_in_round_t` (no final adapter) | n/a | — | — |

## Evaluation

| # | Topic | Our choice | Label |
|---|---|---|---|
| 17 | Protocol | `reference_eval_v1` (see `reproduction_protocol.md`) — our protocol, not the paper's (which is unstated and inconsistent with official numbers) | PAPER-SILENT (R11) |
| 18 | C-Eval subject name in the header | Chinese name from the official mapping (the official LLaMA script passes the English handle) | PAPER-SILENT |
| 19 | MMLU template | lm-eval 0.4.x (no double space after "about") | PAPER-SILENT |
| 20 | Batching | `max_batch_attention` = 4,000,000 (B * L_max^2) to avoid WDDM shared-memory spill | ENVIRONMENT (numerics unchanged within 2e-5 log-prob) |
| 21 | Special tokens | tokenizer default (`add_special_tokens=True`; adds BOS for LLaMA) | PAPER-SILENT |
| 22 | lm-eval cross-check data | lm-eval's `mmlu` task loads `cais/mmlu` at the Hub's current `main`; that SHA is recorded at run time and compared with our pinned revision | — |

## Measurement

| # | Topic | Our choice | Label |
|---|---|---|---|
| 23 | GPU micro-benchmark memory | PyTorch allocator capped at the dedicated VRAM free at start minus 256 MiB, so a configuration that does not fit raises a recorded OOM instead of silently spilling into shared memory (WDDM, P2-D5) | ENVIRONMENT |
| 24 | GPU micro-benchmark numerics | synthetic random tokens at a fixed length of 512 (the `cutoff_len` upper bound), non-deterministic kernels allowed; loss values are meaningless, only memory and time are reported | OURS |

## Phase 3 (Tier B; pre-registered in `phase3_preregistration.md`)

| # | Topic | Our choice | Paper / Shepherd | Label | Configurable |
|---|---|---|---|---|---|
| 25 | Tier-B base precision | Qwen1.5-1.8B bf16, no quantization | the paper appears to use an 8-bit base | ENVIRONMENT (resource-feasible, reviewer D2) | `model.dtype`, `model.quantization` |
| 26 | Local micro-batch | **1 x 32 accumulation** = effective batch 32. The planned 2 x 16 was replaced by the pre-registered A3 rule (at micro-batch 2 the peak reserved memory came within 256 MiB of the allocator cap in 3 of 6 real-data measurements; `results/phase3/calibration/`) | paper: 16 x 2 | ENVIRONMENT (8 GB); changes example weighting (diagnostic) | `local_train.micro_batch_size` |
| 27 | AE input normalisation | `none` (paper-literal) plus `global_rms` / `factor_rms` as DIAGNOSTIC variants, fitted on the D1 training split only, frozen, 0 uplink bytes, no clipping | paper silent | PAPER-SILENT (reviewer A2) | `autoencoder.normalization` |
| 28 | AE checkpoint | final + best-D1-validation; the best-validation AE is gated and probed | paper silent | PAPER-SILENT (reviewer D9) | `autoencoder.checkpoint_policy` |
| 29 | AE budget | 3000 iterations, eval every 50, batch 4, Adam betas 0.9/0.999, no weight decay | paper: MSE/Adam/2e-4 only | PAPER-SILENT (reviewer D9) | `autoencoder.*` |
| 30 | Local-TGAP clients | the 5 clients the FL sampler selects in round 0 | v1 wording only | OURS (aligns time index 0 with the federated schedule) | `tgap.local_client_selection` |
| 31 | Scientific seeds | seed s sets `run.seed`, `lora.init_seed`, `autoencoder.init_seed`; partition, D1/D2 split and sampler are seed-independent | paper silent | OURS | `run.seed`, ... |
| 32 | Training memory guard | allocator capped at free VRAM - 256 MiB in every Phase-3 GPU command | n/a | ENVIRONMENT | `run.allocator_cap_margin_mb` |
| 33 | Left-padding label shift | the last pad position of a left-padded example predicts its first real token (one extra label token per padded example; the count depends on micro-batch composition) | identical in Shepherd/HF Trainer with left padding | — (inherited, documented, unchanged) | — |
| 34 | Result-label schema | three Phase-3 labels added; one Phase-2 record's annotated label (`"DERIVED (...)"`) is accepted by a read-side migration rule instead of rewriting reviewed evidence | n/a | OURS | `canonical_result_label` |

## Phase 4 (pre-registered in `phase4_preregistration.md`)

| # | Topic | Our choice | Paper / Shepherd | Label | Configurable |
|---|---|---|---|---|---|
| 35 | Baseline local micro-batch | **physical 1 x 32 accumulation** (the pre-registered F7 fallback). `virtual_paper_microbatch` (logical 16 x 2, streamed in chunks) is implemented. It is exact on CPU and meets the GPU loss/gradient criteria, but fails the adapter criterion (5.0e-5 > 1e-5) through 9 Adam first-step sign flips at \|g\| <= 1.9e-7. On Qwen, the micro-batch-1 gradient has cosine 0.866 with the logical-16 gradient on a fixed batch (`results/phase4/microbatch/`) | paper 16 x 2 | ENVIRONMENT + pre-registered rule; **PROMINENT** | `local_train.microbatch_mode`, `local_train.physical_chunk_size` |
| 36 | Held-out loss batching | one example at a time for every baseline arm | n/a | OURS (removes the padding dependence of the held-out loss) | `local_train.eval_micro_batch_size` |
| 37 | Gradient instrumentation | read-only hooks (pre-clip gradient, clipped gradient, step delta). A no-op for the optimisation: bitwise on CPU (test) and against the Phase-3 GPU snapshots (F5) | n/a | OURS | `observer_factory` (simulator) |
| 38 | Screen input scaling | `none`; `global_maxabs_train` (frozen scalar p99.9(\|x\|)/0.95, 0 uplink bytes) only if the training p99.9 exceeds 0.95 | paper silent | PHASE4 diagnostic (pre-registered) | `autoencoder.normalization` |
| 39 | Centralized reference | `cent_resource_matched_seed1`: one pooled client with every client's D2, one pass, same local recipe (10,445 examples vs 10,076 seen by the FL run) | paper's Cent configuration unknown | OURS (reference, **not** the paper's Cent) | `federated.pooled` |
| 40 | Benchmark precision | Qwen1.5-1.8B bf16 (the training precision) with `reference_eval_v1`; the evaluator's correctness evidence remains the Phase-2 fp32 lm-eval cross-check | paper unknown | ENVIRONMENT | `model.dtype` |
| 41 | R0/R1 screen | reused from Phase 3 (not retrained) and read against S1-S7 (`screen-reuse`, DERIVED). S6 there pairs factor-space innovations | n/a | OURS (reviewer R2) | — |
| 42 | Result-label schema | PHASE4-FORENSIC, PHASE4-BASELINE, PHASE4-DIAGNOSTIC added; Phase-2/3 labels unchanged and still accepted | n/a | OURS | `ResultLabel` |
| 43 | Evaluation memory (Qwen1.5-1.8B bf16) | The scorer calls the model with `use_cache=False`, and the batch budget is `max_batch_tokens` 8,192, B·L² <= 2e6, batch size 16. Two timing attempts ran out of memory under the 6.69 GiB cap before any result existed; both run directories are preserved. Root cause: the default KV cache (192 KiB/token) of the previous batch stayed alive during the next forward. The first fix (halving the Phase-2 budget of 16,384 / 4e6) was based on an incomplete diagnosis and is kept as margin. With both changes, the real scorer loop over the worst-case region of the full plan peaks at 3.76 GiB allocated / 4.43 GiB reserved. The Phase-2 recorded-evaluation regression (model test) still passes | n/a | ENVIRONMENT (logits are bitwise unchanged without the cache, as tested; batching only groups requests; Base and LoRA-FT use identical settings) | `eval.max_batch_tokens`, `eval.max_batch_attention` |

## Phase 5 (P5-A v2 protocol `phase5_preregistration_v2.md`, `dd4bf3f`, with the v2.1 addendum `phase5_preregistration_v2_1.md`, `33e51fd`; v1 `52a0dd4` historical; real invocation 1 completed and closed, 2026-10-06)

| # | Topic | Our choice | Paper / Shepherd | Label | Configurable |
|---|---|---|---|---|---|
| 44 | Result-label schema | `PHASE5-DIAGNOSTIC` added for every P5-A record (`DERIVED` for arithmetic on recorded results); Phase-2/3/4 labels unchanged and still accepted | n/a | OURS | `ResultLabel` |
| 45 | AE input scaling (P5-A v2) | new mode `global_exact_maxabs_train`: `s_r = max(\|x\|) / 0.95`, the exact maximum over the whole training population (80 / 80 / 5), applied **unconditionally**, no clipping, 0 uplink bytes, one frozen scalar per representation shared by both controls (R2 0.034553585083861103, R3 0.015751632224572334, R4 0.06768756791165001, DERIVED from committed F6 `train_max_abs`); the run recomputes the maximum and stops unless it agrees to rel. 1e-6, never replacing the frozen value. `global_maxabs_train` keeps its Phase-4 p99.9 meaning (row 38, never triggered); the F6 verdicts stand | paper silent | PHASE5 diagnostic (pre-registered) | `configs/phase5/p5a_v2_seed1.yaml` (locked) |
| 46 | AE checkpoint (P5-A) | iteration-3,000 checkpoint only, evaluated in eval mode with BatchNorm running statistics; Phase 3/4 gated the best-D1-validation checkpoint (row 28). P5-A opens no validation split | paper silent | PHASE5 diagnostic (pre-registered) | locked |
| 47 | Fit gates (P5-A v2) | C0 Tanh-range feasibility (an integrity check under exact max-abs), then C1 finite and per *training* snapshot: `single_snapshot` memorisation standard RSE <= 0.01 and cosine >= 0.99 (new v2 numbers; the earlier 0.01 / 0.99 scaffolding was UNSOURCED, never approved); `fixed_subset4` subset fit standard RSE <= 0.50 and cosine >= 0.90 for **every** snapshot plus C4 pooled RSE <= 0.8 x subset-mean RSE, evaluated as written also at a zero subset-mean error. Reuses the Phase-4 S2/S3/S7 numbers for the subset standard only, not the validation screen. Batch `min(4, k)` without duplication. R2/R3 primary, R4 prespecified secondary (always scheduled) | n/a | PHASE5 diagnostic (pre-registered) | locked |
| 48 | Run accounting (P5-A v2) | one run-wide 20,000,000-byte ceiling on all files under `phase5_p5a_v2_seed1/` with per-artifact caps and evidence reservation (estimated worst case 18,990,000); separate CPU-only input preflight before training; at most 3 training invocations with recovery and closure; measured vs charged GPU time | n/a | PHASE5 diagnostic (pre-registered) | locked |
| 49 | Recovery and publication states (P5-A v2.1 addendum, before any output) | invocation 1 may re-enter after an abandoned first publication (empty `ledger/`, recognized temporaries kept and counted; any started, foreign or unknown state refused); a started control is never `RESOURCE STOP` because of bytes: completed trainings whose files cannot be finalized are `INCOMPLETE (PUBLICATION)`, training-stop statuses are kept, and the byte cause is recorded as `resource_stop_cause`; CPU preprocessing environment fingerprint must match the preflight's | n/a | PHASE5 diagnostic (pre-registered addendum) | locked |
| 50 | CPU test execution scope (historical) | every full CPU suite run before 2026-10-05 (including the 308-pass runs) created a CUDA context through an inherited `peft` import; no GPU computation and no data access. Later CPU gates run with `CUDA_VISIBLE_DEVICES=-1` and a read-only probe | n/a | ENVIRONMENT (execution-scope deviation, disclosed; implementation review F5) | launch/test environment |

### Invocation-1 execution and review scope (2026-10-06)

| # | Topic | Recorded disposition | Label |
|---|---|---|---|
| 51 | P5-A v2 + v2.1 real execution | Preflight 1 passed; only authorised GPU invocation 1 ran, at execution HEAD `6603bd8d9156831562dd76c0bf9dba5a071e867f`, with the frozen configuration and six-control order. All six controls completed 3,000 iterations and are `FIT FAIL`; each representation is `CAPACITY FIT NOT DEMONSTRATED` under the fixed protocol. Run closed normally, charged total 2,279.906 s, archived run-root footprint 12,361,855 bytes. No new protocol deviation was identified in the supplied evidence; original records and prior disclosures are preserved. See `phase5_p5a_v2_findings.md` | PHASE5-DIAGNOSTIC (production evidence); DERIVED (audit) |
| 52 | Invocation-1 artifact audit limits | Checkpoint buffers/content, identities, recorded metric arithmetic, gate decisions and publication accounting were checked from the complete archive. No original-payload replay, new model evaluation, GPU peak-memory measurement or repeatability experiment was performed. C1 reconstruction finiteness and raw reduction sums remain producer observations. Negative fit does not establish incompressibility, insufficient general AE capacity or a cause of F6 failures, and authorises nothing downstream | DERIVED / review scope |

## Scope (reviewer decisions)

U-Former and 1-D CNN AEs, DP, all 7B experiments, Qwen-7B rows, formal replicate policy and mechanistic
control experiments are deferred (R5, R9, R10, R15, R16). Controls are implemented and unit-tested.
