# CG-FedLLM Reproduction

This repository is for reproducing the paper **CG-FedLLM: How to Compress Gradients in Federated Fine-Tuning for Large Language Models**.

## 1. Paper

- **Title:** CG-FedLLM: How to Compress Gradients in Federated Fine-Tuning for Large Language Models
- **Authors:** Huiwen Wu et al.
- **Paper:** arXiv:2405.13746
- **Venue:** ECAI 2025

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

- [ ] Set up a clean Python environment
- [ ] Run a basic LoRA fine-tuning experiment
- [ ] Build or adapt a simple federated LoRA baseline
- [ ] Verify client/server aggregation

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
- [ ] Environment setup
- [ ] Baseline implementation
- [ ] CG-FedLLM implementation
- [ ] Experiments
