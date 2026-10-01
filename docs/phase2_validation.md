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
| 7 | MMLU scorer matches lm-eval within 0.5 pp | see §4.2 | `results/phase2/mmlu_lmeval_crosscheck.json` |
| 8 | local LoRA smoke succeeds | PASS | §5 (`lora_ft`, N = 1 check) |
| 9 | federated baseline smoke succeeds | PASS | §5 |
| 10 | Phi round trip exact | PASS | `tests/unit/test_representation_layout.py` (bitwise, 4 geometries) |
| 11 | ResNet AE shape/compression tests pass | PASS | `tests/unit/test_autoencoder.py` |
| 12 | TGAP snapshot collection works | PASS | both source modes, §5 |
| 13 | AE trains on smoke snapshots / control without NaNs | PASS | §6 |
| 14 | FAF runs end to end | PASS | §5 |
| 15 | CPU IdentityCodec equivalence exact | PASS | bitwise (`test_identity_codec_equals_uncompressed_baseline`, CPU smoke) |
| 16 | GPU equivalence within documented tolerance | PASS | §5 (bitwise; delta path 1.9e-7 <= 1e-6) |
| 17 | resume test passes | PASS | CPU test and GPU smoke: bitwise |
| 18 | logical byte accounting verified | PASS | `test_codecs_metrics.py`; smoke: AE uplink x 64 = identity uplink |
| 19 | llama-160m Tier-C smoke completes | PASS | §5 |
| 20 | Qwen0.5B feasibility recorded | see §7 | `results/phase2/gpu_microbench.json` |
| 21 | Qwen1.8B feasibility recorded (downloaded) | see §7 | `results/phase2/gpu_microbench.json` |
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

Commands: `pytest -q -m "not gpu and not model"` (CI and local CPU), `pytest -q -m gpu` (local GPU),
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

_Filled from `results/phase2/mmlu_lmeval_crosscheck.json` when the run completes._

### 4.3 C-Eval structural validation

Official answer-only prompt with the Chinese subject name, first-5 dev shots, single-token `A`-`D`
continuations, subject-macro categories/average, Hard = 8 subjects; verified by prompt fixtures,
mapping/aggregation tests and batching-invariance tests. No C-Eval number is compared with a target.

## 5. Tier-C end-to-end smoke (PHASE2-SMOKE)

_Filled from the clean-commit GPU smoke run (`results/phase2/smoke/`)._

## 6. AutoEncoder

DERIVED (exact, from the implementation): encoder 246,785 / decoder 246,698 parameters (0.9414 / 0.9411
MiB fp32; paper Table 3: 0.94 MB each). MACs at the paper's LLaMA-7B input [1, 4096, 2048]: encoder 0.755 G
(paper 0.81 G); decoder 1.091 G counted on the input grid of each transposed convolution, 1.770 G on its
output grid (paper 1.80 G). Latent [64, H/64, W/64]: compression ratio exactly 1/64 = 1.5625 % for every
tested geometry.

_Smoke training behaviour: filled from the clean-commit smoke run._

## 7. GPU feasibility (LOCAL-MICROBENCH)

_Filled from `results/phase2/gpu_microbench.json`._
