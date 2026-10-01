# CG-FedLLM Reproduction

This repository is for reproducing the paper **CG-FedLLM: How to Compress Gradients in Federated Fine-Tuning for Large Language Models**.

## 1. Paper

- **Title:** CG-FedLLM: How to Compress Gradients in Federated Fine-Tuning for Large Language Models
- **Authors:** Huiwen Wu, Xiaogang Xu, Deyi Zhang, Xiaohan Li, Jiafei Wu, Zhe Liu
- **Venue:** ECAI 2025, Frontiers in Artificial Intelligence and Applications vol. 413, pp. 4257-4264
  (IOS Press), DOI [10.3233/FAIA251320](https://doi.org/10.3233/FAIA251320), CC BY-NC 4.0
- **Extended version:** [arXiv:2405.13746](https://arxiv.org/abs/2405.13746) v3 (main body identical to the
  ECAI text, plus appendices with hyperparameters and AutoEncoder details); v1/v2 are earlier drafts
- Citation metadata: `CITATION.cff`; version notes: `docs/paper_notes.md`, `docs/provenance.md`

## 2. What Problem Does This Paper Solve?

> TODO: Read the paper and summarize this part in my own words.

Things I want to clarify while reading:

- Why is communication expensive in federated fine-tuning of LLMs?
- Why is LoRA alone not enough to solve the communication problem?
- What exactly is being transmitted between clients and the server?
- What trade-off does gradient compression introduce?

## 3. Core Workflow of CG-FedLLM

> TODO: Complete this section after reading the methodology.

Current outline:

1. Stage I: TGAP
   - TODO

2. Stage II: FAF
   - TODO

3. Client side
   - TODO

4. Server side
   - TODO

## 4. Reproduction Plan

The reproduction will be carried out incrementally.

### Phase 1: Understand and prepare

- [x] Create the GitHub repository
- [x] Clone the repository locally
- [x] Record the local hardware/software environment
- [ ] Read and summarize the CG-FedLLM paper
- [ ] Investigate related open-source implementations and baselines
- [ ] Determine a feasible reproduction setting for the available GPU

### Phase 2: Baseline

- [x] Set up a clean Python environment
- [x] Run a basic LoRA fine-tuning experiment (smoke scale)
- [x] Build or adapt a simple federated LoRA baseline (re-implemented FedIT/Shepherd semantics)
- [x] Verify client/server aggregation (regression-tested against the Shepherd formula)

### Phase 3: CG-FedLLM

- [ ] Reproduce gradient / LoRA update collection
- [ ] Reproduce TGAP
- [ ] Train and evaluate the AutoEncoder
- [ ] Reproduce FAF
- [ ] Measure gradient reconstruction quality
- [ ] Measure compression ratio
- [ ] Compare against the federated LoRA baseline

### Phase 4: Experiments

- [ ] Reproduce selected experiments from the paper
- [ ] Compare communication cost
- [ ] Compare downstream model performance
- [ ] Record differences from the original paper

## 5. Related Projects and Baselines

Projects mentioned in or related to the paper:

- FederatedScope-LLM
- OpenFedLLM
- FedIT / FederatedGPT-Shepherd
- LoRA
- QLoRA

Links and notes will be added after checking the corresponding repositories.

## 6. Local Environment

Current machine (re-verified live on 2026-09-30; an earlier version of this section misreported the GPU):

- **OS:** Windows 11 Home (China) 10.0.26100, 64-bit, native (no WSL). Python's `platform` reports "10".
- **CPU / RAM:** Intel Core i9-14900HX (24 cores / 32 threads), 31.7 GB RAM
- **GPU:** NVIDIA GeForce **RTX 4060 Laptop GPU, 8 GB** (8,188 MiB), compute capability 8.9
- **NVIDIA Driver:** 595.97; CUDA version reported by `nvidia-smi`: 13.2; CUDA Toolkit / nvcc: not installed (not needed)
- **Project environment:** dedicated conda env `cgfedllm` (Python 3.11.16) with PyTorch 2.14.0+cu130 (CUDA runtime 13.0),
  transformers 5.17.0, peft 0.21.1, accelerate 1.15.0, datasets 5.0.1, bitsandbytes 0.50.2, lm-eval 0.4.13.
  Exact pins: `requirements/base-win-cu130.txt`; full freeze: `requirements/lock-win-py311-cu130.txt`.
  (The shared Anaconda base env, Python 3.13.9, is not used by this project.)

### Original Paper Environment

The extended version of the paper reports:

- **OS:** Ubuntu Linux 22.04
- **GPU:** NVIDIA RTX 4090, 24 GB
- **PyTorch:** 2.2.1
- **CUDA:** 12.4

Because the local GPU has only 8 GB VRAM, reproducing the complete 7B-model experiments may require smaller-scale experiments or memory-saving methods. The reproduction scope will be decided after studying the paper and baseline implementations.

## 7. Progress

- [x] Repository initialized
- [x] Repository cloned locally
- [x] Local environment inspected
- [ ] Paper reading
- [x] Environment setup
- [x] Baseline implementation
- [ ] CG-FedLLM implementation
- [ ] Experiments

## 8. Phase-2 Status (foundation, baselines, compression core, end-to-end smoke)

Branch `phase2-foundation-smoke`. Phase 2 builds and validates the machinery at smoke scale; it does **not**
reproduce any number from the paper. Every reported number is labelled PAPER-REPORTED, PHASE2-SMOKE,
LOCAL-MICROBENCH, DERIVED or UNKNOWN.

Implemented (package `src/cg_fedllm`, see `docs/architecture.md`): strict configs with provenance;
Shepherd-compatible Dolly partition (oracle-equivalent) and per-client D1/D2 manifests; `reference_eval_v1`
(C-Eval / MMLU log-likelihood evaluator, cross-checked against lm-eval); local LoRA training and a
FedIT/Shepherd-style simulator with resume; `adapter_state | adapter_delta` representations and the Phi layout;
the reconstructed ResNet-3 AutoEncoder (1/64 compression); TGAP collection (`local_pretrain` and
`federated_pretrain`); FAF with Identity / AutoEncoder / ConstantMean / GaussianNoise codecs.

Quickstart (Windows, conda; see `docs/reproduction_protocol.md`):

```bash
conda create -n cgfedllm python=3.11 -y && conda activate cgfedllm
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements/base-win-cu130.txt && python -m pip install -e . --no-deps
pytest -q -m "not gpu and not model"                      # CPU suite (also run in CI)
cgfed smoke --config configs/smoke/llama160m_smoke.yaml   # Tier-C GPU smoke (llama-160m)
```

Documentation: `docs/phase2_validation.md` (exit gates, smoke, evaluator, GPU measurements),
`docs/reproduction_protocol.md`, `docs/architecture.md`, `docs/paper_notes.md`, `docs/evidence_ledger.md`
(resolved / inferred / unknown), `docs/deviations.md`, `docs/discrepancies.md`, `docs/provenance.md`.
Results: `results/phase2/`. Licenses: Apache-2.0 (`LICENSE`), third-party notices in `NOTICE`; models and
datasets are downloaded at pinned revisions and never committed.
