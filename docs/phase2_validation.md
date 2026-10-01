# Phase-2 validation record

Every number below carries one label: **PAPER-REPORTED**, **PHASE2-SMOKE**, **LOCAL-MICROBENCH**, **DERIVED**
or **UNKNOWN**. Nothing here is a reproduction of a CG-FedLLM paper result. Smoke-scale numbers validate the
machinery only.

Machine: Windows 11 (native), RTX 4060 Laptop GPU 8 GB (WDDM), i9-14900HX, conda env `cgfedllm`
(Python 3.11.16, torch 2.14.0+cu130, transformers 5.17.0, peft 0.21.1, lm-eval 0.4.13). Exact freeze:
`requirements/lock-win-py311-cu130.txt`.

## 1. Exit gates

| # | Gate | Outcome | Evidence |
|---|---|---|---|
| 1 | dedicated environment works | PASS | conda env `cgfedllm`; CPU and GPU suites; all runs below |
| 2 | environment frozen/recorded | PASS | `requirements/*.txt`; every run writes `run_metadata.json` (packages, CUDA, Git) |
| 3 | repository scaffold complete | PASS | `docs/architecture.md` |
| 4 | CPU unit/integration suite passes | PASS | §2 |
| 5 | partition deterministic and Shepherd-compatible | PASS | §3 |
| 6 | evaluator tests pass | PASS | `tests/unit/test_evaluation.py` |
| 7 | MMLU scorer matches lm-eval within 0.5 pp | PASS | delta 0.00 pp, 14,042/14,042 questions agree (§4.2) |
| 8 | local LoRA smoke succeeds | PASS | §5 (`lora_ft`, N = 1 check) |
| 9 | federated baseline smoke succeeds | PASS | §5 |
| 10 | Phi round trip exact | PASS | `tests/unit/test_representation_layout.py` (bitwise; 5 geometries: tiny, llama-160m, LLaMA-7B r 8 and r 16, Qwen1.5-1.8B) |
| 11 | ResNet AE shape/compression tests pass | PASS | `tests/unit/test_autoencoder.py` |
| 12 | TGAP snapshot collection works | PASS | both source modes, §5 |
| 13 | AE trains on smoke snapshots / control without NaNs | PASS | §6 |
| 14 | FAF runs end to end | PASS | §5 |
| 15 | CPU IdentityCodec equivalence exact | PASS | bitwise (`test_identity_codec_equals_uncompressed_baseline`, CPU smoke) |
| 16 | GPU equivalence within documented tolerance | PASS | §5 (bitwise; delta path 1.9e-7 <= 1e-6) |
| 17 | resume test passes | PASS | CPU test and GPU smoke: bitwise |
| 18 | logical byte accounting verified | PASS | `test_codecs_metrics.py`; smoke: AE uplink x 64 = identity uplink |
| 19 | llama-160m Tier-C smoke completes | PASS | §5 |
| 20 | Qwen0.5B feasibility recorded | PASS | §7 (bf16 with/without checkpointing; plus the evaluator run, §4) |
| 21 | Qwen1.8B feasibility recorded (downloaded) | PASS | §7 (bf16, nf4, int8) |
| 22 | no prohibited large/generated assets in Git | PASS | `.gitignore`; largest tracked files are the ID-only manifests/fixtures |
| 23 | commits logically structured | PASS | one commit per milestone (see `git log`) |
| 24 | Phase-2 branch pushed | PASS | `origin/phase2-foundation-smoke` |

## 2. Tests

Testing-standard coverage (item -> test):

| Item | Test |
|---|---|
| 1 config validation | `tests/unit/test_config.py` |
| 2 sampler regression | `test_sampling_aggregation.py::test_sampler_matches_audited_shepherd_algorithm`, `test_selection_count_rule` |
| 3 sample-weighted FedAvg regression | `test_sample_weighted_mean_matches_shepherd_formula_bitwise` |
| 4 all aggregation modes | `test_weights_and_all_modes`, `test_factors_are_aggregated_independently_not_in_product_space`, `test_literal_sum_is_diagnostic_and_scales_factors` |
| 5 state/delta conversion | `test_representation_layout.py::test_state_and_delta_representations` |
| 6 Phi round trip | `test_phi_roundtrip_is_bitwise_exact` |
| 7 layout key ordering | `test_default_layout_block_order_and_independence_from_dict_order` |
| 8 incompatible layouts rejected | `test_incompatible_geometries_fail_loudly` |
| 9-12 AE dimensions, CR, parameters, forward/backward | `tests/unit/test_autoencoder.py` |
| 13 codec determinism | `test_gaussian_noise_codec_is_seeded_per_round_and_client`, `test_autoencoder_codec_determinism_and_latent_bytes` |
| 14 payload bytes | `test_identity_codec_is_lossless_and_accounts_raw_bytes`, `test_constant_mean_codec_is_input_independent_and_sends_nothing` |
| 15 TGAP snapshot schema | `tests/unit/test_snapshots.py`, `tests/integration/test_tgap_collect.py` |
| 16 D1/D2 disjointness | `test_d1_d2_split_is_disjoint_complete_and_deterministic` |
| 17 manifest stability | `test_committed_manifest_is_reproduced_byte_for_byte`, `test_canonical_json_is_stable` |
| 18 LoRA-only updates | `test_local_training_updates_only_lora_and_is_deterministic` |
| 19 one FL round, 20 N = 1 equivalence | `test_one_round_and_n1_equivalence` |
| 21 IdentityCodec equivalence | `test_identity_codec_equals_uncompressed_baseline` |
| 22 resume determinism | `test_resume_reproduces_uninterrupted_run` |
| 23 evaluation prompt/scoring fixtures | `tests/unit/test_evaluation.py` |
| 24 MMLU cross-check vs lm-eval | `scripts/crosscheck_mmlu_lmeval.py` -> `results/phase2/mmlu_lmeval_crosscheck.json`, asserted by `tests/unit/test_recorded_results.py` |
| 25 end-to-end Tier-C smoke | `tests/integration/test_smoke_cpu.py` (CPU), §5 (GPU) |

Additional: `tests/integration/test_bench_gpu.py` (gpu marker) checks the benchmark harness and that an
allocation beyond the allocator cap is recorded as OOM; `tests/model/test_recorded_eval_regression.py`
(model + gpu markers, local cache only) re-scores 36 recorded evaluator questions with Qwen1.5-0.5B.

Commands: `pytest -q -m "not gpu and not model"` (CI and local CPU), `pytest -q -m "gpu and not model"` (local GPU),
`CGFED_RUN_MODEL_TESTS=1 HF_HUB_OFFLINE=1 pytest -q -m model` (local cache).

## 3. Data and partition (PHASE2-SMOKE / DERIVED)

* Source: Shepherd's Dolly file (first Dolly release, 15,015 records), SHA-256 `52e0c44e2155bfc9...`
  (`configs/base/shepherd_dolly.yaml`).
* Oracle: Shepherd `client_data_allocation.py` @ `bcffa00` executed under pandas 2.3.3 / numpy 2.4.6
  (`scripts/make_shepherd_oracle.py`, fixtures regenerated byte-identically). Our partition equals the oracle
  exactly for the Dirichlet (100 clients) and shard (10 clients) modes: held-out IDs, remaining order,
  client order and sizes.
* Manifests (rebuilt byte for byte in CI): `dolly_shepherd_dirichlet100_seed42.json` (100 clients, sizes
  47-303, 80 held out; 13 Dirichlet attempts) and `dolly_smoke_subset12_4clients.json` (4 clients, 10-25,
  16 held out). Per-client D1/D2 = 30/70 (`floor(0.3 n + 0.5)`), disjoint and complete (INFERRED, R7).
* Finding P2-D1: Shepherd's partition depends on the pandas version (see `discrepancies.md`).

## 4. Evaluator validation (PHASE2-SMOKE: base Qwen1.5-0.5B, fp32, no adapter)

### 4.1 reference_eval_v1 results

Source: `results/phase2/eval_validation_qwen15_0p5b.json` (run `eval_full`; evaluation code = commit
`0d42b68`, see the provenance notes in that file).

| Benchmark (5-shot) | Questions | Headline | Categories | Hard | Score time |
|---|---|---|---|---|---|
| MMLU test | 14,042 | overall **38.76 %** (question-weighted); subject macro 39.92 % | STEM 33.60, humanities 36.77, social sciences 43.00, other 42.41 | - | 1,230 s |
| C-Eval val | 1,346 | average **50.00 %** (subject macro) | STEM 46.83, Social Science 60.63, Humanities 48.46, Other 47.63 | 39.08 % | 92 s |
| C-Eval test | 12,342 | average **48.78 %** (subject macro) | STEM 45.43, Social Science 62.46, Humanities 48.21, Other 43.01 | 35.87 % | 798 s |

All requests were single-token, none truncated, none shot-reduced (longest 5-shot context: MMLU 3,096 tokens,
C-Eval 1,378; median 524 / 485). Scoring throughput (LOCAL-MICROBENCH, fp32, sdpa, `max_batch_attention` 4e6):
MMLU 9.78 M tokens in 1,230 s (7.9 k tok/s), C-Eval test 6.53 M tokens in 798 s (8.2 k tok/s).
Sanity references (published by the Qwen team for Qwen1.5-0.5B, protocol/split not fully specified; **not**
targets): MMLU 39.2, C-Eval 50.5. Our values are within 0.5 pp (MMLU) and 0.5-1.7 pp (C-Eval val/test).

### 4.2 lm-eval cross-check (hard gate)

`results/phase2/mmlu_lmeval_crosscheck.json` (PHASE2-SMOKE; DERIVED comparison). lm-eval 0.4.13, unmodified, tasks
`mmlu_<subject>`, 5-shot, same Qwen1.5-0.5B weights in float32, batch size 1; `cais/mmlu` main resolved to
`c30699e8`, identical to our pinned revision.

| | ours (`reference_eval_v1`) | lm-eval | difference |
|---|---|---|---|
| MMLU test overall (question-weighted, 14,042 questions) | 38.7552 % | 38.7552 % | **0.00 pp** (gate <= 0.5 pp: **PASS**) |
| per-subject accuracy (57 subjects) | | | max abs 0.00 pp |
| per-question correctness | | | **14,042 / 14,042 agree (100 %)** |

Device split (the unmodified lm-eval fp32 reference needs ~7.5 GB at the longest contexts; see
`discrepancies.md` P2-D5): 55 subjects / 13,673 questions on the GPU (allocator capped at 7.18 GB, 2,271 s) and
`high_school_european_history` + `high_school_us_history` / 369 questions (contexts up to 3,096 tokens) on the CPU
(2,122 s). Both parts ran from commit `62721f7` (only `docs/phase2_validation.md` modified). An earlier
single-device attempt spilled into WDDM shared memory and was stopped (2,873 of 56,168 requests after 9 min,
GPU at 17 W); a capped single-device attempt raised OOM at the longest context.

### 4.3 C-Eval structural validation

Official answer-only prompt with the Chinese subject name, first-5 dev shots, single-token `A`-`D`
continuations, subject-macro categories/average, Hard = 8 subjects; verified by prompt fixtures,
mapping/aggregation tests and batching-invariance tests. No C-Eval number is compared with a target.

## 5. Tier-C end-to-end smoke (PHASE2-SMOKE)

Run `smoke_final` from the clean commit `25a3f7f` (`cgfed smoke --config configs/smoke/llama160m_smoke.yaml`);
summary: `results/phase2/smoke/smoke_summary.{json,md}`. Setting: JackFram/llama-160m @ `aca9b687` (fp32, eager
attention), deterministic Dolly subset (12 per category; 4 Dirichlet(0.5) clients with 10-25 examples, 16 held
out), per-client D1/D2 30/70, cutoff 256, LoRA r 8 / alpha 16 / dropout 0.05 on q,k,v,o, local batch 8 (micro 4),
lr 1.5e-4, 1 epoch, 2 rounds at fraction 0.5 (2 of 4 clients), `sample_weighted_mean`, `adapter_state` unless
stated, layout `layer_major_qkvo_AtB` (Phi input [1, 768, 768]).

**Runtime:** 88 s wall clock (sum of stages 74.8 s; target <= 20 min). Stages (s): lora_ft 6.2, tgap_local 4.0,
tgap_federated 4.0, AE training 25.8, faf_identity 6.2, faf_identity_delta 6.3, faf_autoencoder 6.3, cent 4.2,
resume 6.3, evaluation 5.6.

**Equivalence gates (GPU):**

| Check | Result | Gate |
|---|---|---|
| FAF + IdentityCodec (`adapter_state`) vs uncompressed LoRA-FT | bitwise equal, every round hash equal | bitwise or rel. L2 <= 1e-6 |
| FAF + IdentityCodec (`adapter_delta`) vs LoRA-FT | rel. L2 1.86e-7, max abs 1.36e-6 (fp32 rounding of G + (end - G)) | rel. L2 <= 1e-6 |
| resume after an interruption vs uninterrupted | bitwise equal | bitwise |
| N = 1 federated round vs plain local training | bitwise equal | bitwise or rel. L2 <= 1e-6 |
| run-to-run (this run vs an earlier separate run, `smoke_v2`) | all final adapter hashes identical (LoRA-FT `28c3e9a0`, delta `ff2a45ed`, FAF-AE `d1a04b2c`, Cent `179e1f52`, TGAP-federated `45792c2a`) and identical AE metrics | (extra evidence) |

**Held-out loss** (token-weighted CE on 16 held-out Dolly examples; PHASE2-SMOKE):

| Run | initial | round 0 | round 1 | uplink bytes (logical, whole run) |
|---|---|---|---|---|
| LoRA-FT (uncompressed) | 3.58824 | 3.53689 | 3.47855 | 9,437,184 |
| FAF + IdentityCodec (state) | 3.58824 | 3.53689 | 3.47855 | 9,437,184 |
| FAF + IdentityCodec (delta) | 3.58824 | 3.53689 | 3.47855 | 9,437,184 |
| FAF + AutoEncoder (state, AE from local TGAP) | 3.58824 | 3.58952 | **8.02149** | 147,456 (= 1/64) |
| `cent_smoke_sample_matched` (1 pooled pass; not the paper's Cent) | 3.58824 | 3.46666 | - | 2,359,296 |

Per client upload: 589,824 LoRA parameters = 2,359,296 bytes (fp32) uncompressed vs a 9,216-element latent =
36,864 bytes with the AE. The FAF-AE run completes but the global adapter is corrupted by the AE's
reconstruction error (section 6): squared global norms after round 0 are A 132.3 / B 77.6 (uncompressed
127.8 / 0.013) and after round 1 A 4,055 / B 1,808 (uncompressed 128.1 / 0.052). With `adapter_state` the Phi
tensor's energy is dominated by the near-constant random A initialisation, while the useful signal lives in the
small B factors, so an error that is moderate relative to the whole tensor is enormous relative to B.

The first GPU smoke attempt (60 AE iterations) hit a non-finite local training loss in FAF-AE round 1 and aborted (round-0 global norm^2 of A
4,999.8 vs 127.8 uncompressed, of B 3,426 vs 0.013; held-out loss 17.58). Since then the simulator records such runs as
`diverged_in_round_t` instead of raising (tested with a NaN codec), and the smoke AE budget is 600 iterations.

**TGAP collection** (both source modes share the snapshot schema; DERIVED statistics in
`results/phase2/smoke/tgap_snapshot_stats.json`):

| | `local_pretrain` (smoke default) | `federated_pretrain` |
|---|---|---|
| snapshots | 8 (4 clients x 2 sequential local steps on D1) | 8 (2 rounds x 4 clients, fraction 1.0) |
| distinct start states | 5 | 2 |
| mean norm A_end / B_end | 11.307 / 0.118 | 11.307 / 0.108 |
| relative A update (median) | 0.72 % (0 at the first step: B = 0 gives zero A-gradient) | 0.72 % |
| mean norm of (end - start) | 0.0982 | 0.0982 |
| mean pairwise cosine of updates | 0.448 | 0.463 |

At smoke scale the two modes are statistically almost indistinguishable; neither is claimed to be the paper's.

**Evaluation hook:** the reference evaluator runs on the base model, LoRA-FT and FAF-AE adapters (10 MMLU +
10 C-Eval questions each) to exercise the adapter -> evaluator path only; these accuracies carry no information.

## 6. AutoEncoder

DERIVED (exact, from the implementation): encoder 246,785 / decoder 246,698 parameters (0.9414 / 0.9411
MiB fp32; paper Table 3: 0.94 MB each). MACs at the paper's LLaMA-7B input [1, 4096, 2048]: encoder 0.755 G
(paper 0.81 G); decoder 1.091 G counted on the input grid of each transposed convolution, 1.770 G on its
output grid (paper 1.80 G). Latent [64, H/64, W/64]: compression ratio exactly 1/64 = 1.5625 % for every
tested geometry.

**Smoke training behaviour (PHASE2-SMOKE, `smoke_final`).** 8 `local_pretrain` snapshots in `adapter_state`
form, Phi input [1, 768, 768] -> latent [64, 12, 12]; temporal split (time step 0 = 4 training snapshots, step 1 =
4 validation snapshots); batch 2, Adam lr 2e-4, 600 iterations (25.8 s on the GPU), no input normalisation.
Training is finite and the validation MSE falls monotonically (4.98e-3 at iteration 1, 3.00e-3 at 100, 5.70e-4
at 600) but has not converged:

| Predictor (validation set) | relative squared error | standard SNR | paper-style SNR | cosine |
|---|---|---|---|---|
| AutoEncoder | 2.628 | -4.2 dB | 2.24e5 | 0.005 |
| zero | 1.000 | 0.0 dB | 5.90e5 | - |
| train mean (input-independent) | 1.49e-4 | +38.3 dB | 3.97e9 | 0.99993 |

The AE is worse than predicting zero. The snapshots are almost identical (dominated by the shared random A
initialisation, section 5), so the input-independent train mean is a near-perfect predictor that the AE does not
approach within 600 iterations. Candidate causes for Phase 3 to separate: the reconstructed single-channel ReLU
stem (it discards part of a signed input's information), no input normalisation, only 4 training snapshots, the
iteration budget, and the representation (`adapter_delta` inputs are ~100x smaller). The paper-style
SNR (summed signal energy / per-element MSE) of the zero predictor equals the element count (589,824), which
shows how that definition inflates values. On the CPU learnability control (structured synthetic matrices)
the same training loop fits well (`test_ae_training.py`), so this is not a broken optimiser. Whether input
normalisation, the representation, a longer budget or more snapshots change this is a Phase-3 question; the
paper does not specify normalisation, batch size or iterations (`evidence_ledger.md`). Nothing was tuned in
Phase 2 to improve these numbers.

**Per TGAP source and representation** (same AE configuration and budget; `results/phase2/smoke/`; PHASE2-SMOKE):

| AE training data (8 snapshots, temporal split 4/4) | AE val. rel. sq. error | zero | train mean | file |
|---|---|---|---|---|
| `local_pretrain`, `adapter_state` (smoke default; used by FAF-AE) | 2.628 | 1.0 | 1.49e-4 | `smoke_summary.json` |
| `federated_pretrain`, `adapter_state` | 2.947 | 1.0 | 1.04e-4 | `ae_metrics_federated_state.json` |
| `local_pretrain`, `adapter_delta` | 5.04e4 | 1.0 | 0.746 | `ae_metrics_local_delta.json` |

Unnormalised `adapter_delta` inputs (per-element magnitude ~1e-4) sit far below the AE's output error floor, so the
relative error explodes; for `adapter_state` the two source modes behave alike. None of these runs says which
source mode or representation the paper used.

## 7. GPU feasibility (LOCAL-MICROBENCH)

`results/phase2/gpu_microbench.json` (`cgfed bench-gpu --config configs/feasible/gpu_microbench.yaml`, commit
`e132491`, only docs and this result file uncommitted). LoRA r 8 on q,k,v,o (fp32 adapters), synthetic sequences
of exactly 512 tokens (the `cutoff_len` upper bound), the same loss as the trainer (full-vocabulary logits cast to
fp32), AdamW, 2 warm-up + 5 timed steps, sdpa attention. The PyTorch allocator was capped at the free dedicated
VRAM minus 256 MiB = **6.69 GiB** (of 8.00 GiB), so a configuration that does not fit fails with OOM instead of
spilling silently into shared memory. A repeat run gave identical memory and throughput within +-6 %.

| Model | Base precision | Grad. ckpt | Micro-batch | Status | Peak allocated / reserved (GiB) | Tokens/s |
|---|---|---|---|---|---|---|
| llama-160m | bf16 | no | 1 / 4 / 8 / 16 | ok / ok / ok / OOM | 0.79 / 2.13 / 3.91 (reserved 0.83 / 2.30 / 4.22) | 8,287 / 17,189 / 17,124 |
| Qwen1.5-0.5B | bf16 | no | 1 / 2 / 4 / 8 | ok / ok / OOM / OOM | 2.41 / 3.90 (reserved 2.62 / 4.27) | 3,691 / 5,135 |
| Qwen1.5-0.5B | bf16 | yes | 1 / 2 / 4 / 8 | ok / ok / ok / OOM | 1.81 / 2.71 / 4.49 (reserved 2.00 / 3.07 / 5.17) | 2,602 / 4,247 / 4,362 |
| Qwen1.5-1.8B | bf16 | no | 1 / 2 / 4 / 8 | ok / OOM / OOM / OOM | 5.60 (reserved 5.82) | 2,156 |
| Qwen1.5-1.8B | bf16 | yes | 1 / 2 / 4 / 8 | ok / ok / OOM / OOM | 4.43 / 5.35 (reserved 4.77 / 5.81) | 1,709 / 1,899 |
| Qwen1.5-1.8B | nf4 (bf16 compute) | yes | 1 / 2 / 4 / 8 | ok / ok / ok / OOM | 2.73 / 3.65 / 5.49 (reserved 2.96 / 4.09 / 6.21) | 1,619 / 1,802 / 1,967 |
| Qwen1.5-1.8B | int8 (LLM.int8) | yes | 1 / 2 / 4 / 8 | ok / ok / OOM / OOM | 3.69 / 4.98 (reserved 4.09 / 5.55) | 1,088 / 1,563 |

Weights resident after loading: llama-160m 0.31 GiB, Qwen1.5-0.5B 0.91 GiB, Qwen1.5-1.8B 3.49 GiB (bf16), 1.79 GiB
(nf4), 2.32 GiB (int8). With Qwen's 151,936-token vocabulary the fp32 logits (0.31 GB per 512-token sequence) and
their gradient dominate activation memory, which is why micro-batches above 2-4 do not fit even with checkpointing.

Other measured GPU numbers: the Tier-C smoke takes 88 s; evaluator scoring of Qwen1.5-0.5B in fp32 runs at
7.9-8.2 k tokens/s (section 4.1).

**DERIVED projection for Phase 3** (padded training tokens of one 20-round Dolly FL run, from the real
Qwen-tokenised client data: 10,076 examples, 1.92 M real tokens, mean 182 tokens per example; divided by the
measured 512-token throughput, so shorter real sequences may run less efficiently; local training only):

| Configuration | Padded tokens | One FL run | One pass over all D1 (0.81 M real tokens) |
|---|---|---|---|
| Qwen1.5-1.8B bf16, checkpointing, micro-batch 2 | 2.64 M | ~23 min | ~10 min |
| Qwen1.5-1.8B bf16, no checkpointing, micro-batch 1 | 1.95 M | ~15 min | ~6 min |
| Qwen1.5-1.8B nf4, checkpointing, micro-batch 2 / 4 | 2.64 / 3.38 M | ~24 / ~29 min | ~10 / ~12 min |
| Qwen1.5-1.8B int8, checkpointing, micro-batch 2 | 2.64 M | ~28 min | ~12 min |
| Qwen1.5-0.5B bf16, no checkpointing, micro-batch 2 | 2.64 M | ~9 min | ~4 min |
