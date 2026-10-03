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

Federated fine-tuning avoids centralizing private client data, but it still
requires repeated communication between clients and the server. LoRA reduces
the number of trainable parameters compared with full-model fine-tuning, yet
the LoRA factors still have to be uploaded during federated training.

CG-FedLLM proposes compressing the transmitted LoRA representation with a
learned convolutional AutoEncoder:

- the client trains its local LoRA adapter;
- the client maps the LoRA factors into an AutoEncoder input and uploads only
  the latent representation;
- the server decodes the latent representation;
- the decoded LoRA factors are aggregated to update the global adapter.

The intended trade-off is communication reduction versus reconstruction error
and any downstream degradation introduced by compression.

A central reproduction issue is that the paper calls the transmitted objects
"gradients" but does not fully specify whether they are absolute LoRA states,
client-local increments, or another representation. This repository therefore
keeps paper-reported facts separate from reconstructed or inferred semantics.

See `docs/paper_notes.md`, `docs/evidence_ledger.md`, and
`docs/representation_forensics.md`.

## 3. Core Workflow of CG-FedLLM

The paper describes a two-stage workflow.

### Stage I: TGAP

Temporal-ensemble Gradient-Aware Pre-training (TGAP) uses the D1 portion of the
training data (30 %) to collect LoRA A/B representations from multiple clients
and time steps.

Those representations are used to train the AutoEncoder with reconstruction
MSE. The paper reports Adam with learning rate `2e-4`.

The purpose of TGAP is to train the compressor before the compressed federated
fine-tuning stage begins.

### Stage II: FAF

Federated AutoEncoder-Involved Fine-tuning (FAF) uses D2 (70 %).

For each selected client:

1. start from the current global LoRA adapter;
2. perform local LoRA fine-tuning;
3. arrange the transmitted A/B factors into the compressor representation;
4. encode the representation on the client;
5. upload the compressed latent representation.

On the server:

1. decode each client latent;
2. reconstruct the corresponding LoRA factors;
3. aggregate the reconstructed client contributions;
4. update the global adapter;
5. continue to the next communication round.

The paper does not fully disambiguate the transmitted representation. The
repository therefore supports and tests multiple interpretations, including
adapter state and adapter delta, rather than silently treating one inference as
paper fact.

The reconstructed algorithm and evidence boundaries are documented in
`docs/paper_notes.md`, `docs/architecture.md`, and
`docs/representation_forensics.md`.

## 4. Reproduction Plan

The reproduction is carried out incrementally. A checked item means that the
repository has implemented, investigated, or measured that item at the scope
described here; it does not imply that every paper-reported result has been
reproduced.

### Phase 1: Understand and prepare

- [x] Create the GitHub repository
- [x] Clone the repository locally
- [x] Record the local hardware/software environment
- [x] Read and summarize the CG-FedLLM paper
- [x] Audit the related implementations and baselines used by this reproduction
- [x] Determine a feasible reproduction setting for the available GPU

### Phase 2: Baseline and infrastructure

- [x] Set up a clean Python environment
- [x] Run a basic LoRA fine-tuning experiment at smoke scale
- [x] Build a federated LoRA baseline with re-implemented FedIT/Shepherd semantics
- [x] Verify client/server aggregation against the Shepherd formula
- [x] Implement the compression, TGAP, FAF, evaluation, provenance, and resume machinery at smoke scale

Phase 2 validates infrastructure and smoke-scale behavior. It does not reproduce
a paper result.

### Phase 3: Tier-B compressor diagnosis

- [x] Collect Tier-B federated LoRA states/updates with Qwen1.5-1.8B
- [x] Exercise the TGAP collection and AutoEncoder training pipeline
- [x] Train and evaluate the reconstructed AutoEncoder at the 1/64 target ratio
- [x] Measure factor-aware reconstruction and update-direction quality
- [x] Measure compression ratio and logical communication
- [x] Run pre-registered representation and sensitivity diagnostics
- [ ] Run an operational AE-backed FAF core experiment

The operational compressed FAF experiment was not started because no
pre-registered Phase 3 compressor candidate passed the viability gates. That is
a recorded negative result, not an unfinished implementation task.

### Phase 4: Representation forensics and baseline evaluation

- [x] Audit the paper's representation semantics
- [x] Validate paper-style virtual micro-batching and document the remaining deviation
- [x] Implement and test gauge-invariant LoRA representations
- [x] Characterize state, update, and gradient representations
- [x] Screen candidate representations at the 1/64 target ratio
- [x] Run the seed-1 Tier-B LoRA-FT and FAF-Identity baseline
- [x] Measure logical communication cost
- [x] Run the planned MMLU and C-Eval downstream evaluations
- [x] Record deviations, discrepancies, negative results, and evidence provenance

Phase 4 still does not establish a viable operational AutoEncoder-backed FAF
compressor at the paper's target ratio.

## 5. Related Projects and Baselines

Projects mentioned in or related to the reproduction include:

- FederatedScope-LLM
- OpenFedLLM
- FedIT / FederatedGPT-Shepherd
- LoRA
- QLoRA

The repository uses re-implemented FedIT/Shepherd-style semantics as a baseline
reference where documented. Paper, implementation, and provenance decisions are
tracked in `docs/paper_notes.md`, `docs/provenance.md`,
`docs/evidence_ledger.md`, and `docs/representation_forensics.md`.

No external implementation is treated as authoritative for an ambiguity unless
the repository records the supporting evidence explicitly.

## 6. Local Environment

Current machine (re-verified live on 2026-09-30; an earlier version of this
section misreported the GPU):

- **OS:** Windows 11 Home (China) 10.0.26100, 64-bit, native (no WSL).
  Python's `platform` reports "10".
- **CPU / RAM:** Intel Core i9-14900HX (24 cores / 32 threads), 31.7 GB RAM
- **GPU:** NVIDIA GeForce **RTX 4060 Laptop GPU, 8 GB** (8,188 MiB),
  compute capability 8.9
- **NVIDIA Driver:** 595.97; CUDA version reported by `nvidia-smi`: 13.2;
  CUDA Toolkit / nvcc: not installed (not needed)
- **Project environment:** dedicated conda env `cgfedllm` (Python 3.11.16)
  with PyTorch 2.14.0+cu130 (CUDA runtime 13.0), transformers 5.17.0,
  peft 0.21.1, accelerate 1.15.0, datasets 5.0.1, bitsandbytes 0.50.2,
  lm-eval 0.4.13.
  Exact pins: `requirements/base-win-cu130.txt`; full freeze:
  `requirements/lock-win-py311-cu130.txt`.
  The shared Anaconda base environment (Python 3.13.9) is not used by this
  project.

### Original Paper Environment

The extended version of the paper reports:

- **OS:** Ubuntu Linux 22.04
- **GPU:** NVIDIA RTX 4090, 24 GB
- **PyTorch:** 2.2.1
- **CUDA:** 12.4

The local 8 GB GPU cannot reproduce the paper's full hardware setting directly.
The repository therefore uses explicitly labelled smoke and Tier-B settings,
including Qwen1.5-1.8B experiments, resource-feasible micro-batches, and
pre-registered deviations.

These resource adaptations are part of the reproduction evidence and are
recorded rather than presented as paper-faithful settings.

## 7. Progress

- [x] Repository and development environment established
- [x] Paper and method audit completed
- [x] Baseline federated LoRA implementation validated
- [x] CG-FedLLM data, compression, TGAP, FAF, and evaluation machinery implemented
- [x] Phase 2 smoke-scale validation completed
- [x] Phase 3 Tier-B AutoEncoder viability diagnosis completed
- [x] Phase 4 representation-forensics and seed-1 baseline evaluation completed
- [x] Communication and downstream baseline measurements recorded
- [x] Deviations, discrepancies, and evidence provenance recorded
- [ ] Demonstrate a viable operational AE-backed FAF compressor at the paper's target ratio
- [ ] Reproduce the paper-scale experimental claims across the required settings and seeds

## 8. Phase-2 Status (foundation, baselines, compression core, end-to-end smoke)

Published Phase 2 history originated on `phase2-foundation-smoke` and is integrated into `main`. 
Phase 2 builds and validates the machinery at smoke scale; it does **not**
reproduce any number from the paper. 
Every reported number is labelled PAPER-REPORTED, PHASE2-SMOKE,
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

## 9. Phase-3 Status (Tier-B AutoEncoder diagnosis; Phase 3A only)

Published Phase 3 history originated on `phase3-tierb-diagnosis-core` and is integrated into `main`. 
Phase 3A asks whether the reconstructed CG-FedLLM compressor can
reconstruct real federated LoRA states or updates of Qwen1.5-1.8B at the paper's 1/64 ratio. 
The protocol and
the minimum-viability gates were pre-registered in `docs/phase3_preregistration.md` before any data were
collected.
The configuration (bf16 + gradient checkpointing, micro-batch 1 x 32) is resource-feasible, NOT
paper-faithful. Labels: PHASE3-DIAGNOSTIC, PHASE3-SENSITIVITY, DERIVED.

**Outcome (seed 1).**
* *A5/A6:* none of the three pre-registered AE candidates passes the gate on `federated_pretrain + adapter_state`
  (`none`, `global_rms`, `factor_rms`). Their reconstructions carry no information about the client's update:
  innovation cosine about -0.02, aggregate-update cosine about -0.04. The verdict is **NO PRIMARY CODEC IS VIABLE**.
* *A7, Phase 3B:* consequently the one-round FAF probe, the 20-round core runs, the replicate seeds and the
  benchmark evaluation were not run.
* *A8:* the sensitivities (`local_pretrain` state; federated delta with `factor_rms`) fail too.
* *Interpretation:* the absolute LoRA state is about 99.96 % shared initialisation plus history, and a client
  update is 3.6e-4 of its energy.
  * Generic codes at the same element ratio keep at most 17 % of the A energy.
  * Even a decoder that memorises the training snapshots recovers only 0.15 of the update direction.

Details: `docs/phase3_findings.md`. Evidence: `results/phase3/`. Deviations and discrepancies:
`docs/deviations.md` (rows 25-34), `docs/discrepancies.md` (section C).

## 10. Phase-4 Status (representation forensics, micro-batch fidelity, seed-1 baseline)

Published Phase 4 history originated on `phase4-representation-forensics` and is integrated into `main`. 
Phase 4 audits what the CG-FedLLM AutoEncoder encodes, emulates the paper's micro-batch, and produces one trustworthy seed-1 Tier-B baseline.
* Protocol: pre-registered in `docs/phase4_preregistration.md`.
* Evidence audit: `docs/representation_forensics.md`.
* Labels: PHASE4-FORENSIC, PHASE4-BASELINE, PHASE4-DIAGNOSTIC, DERIVED.
* No benchmark result was used to select a representation, and no operational AE/FAF compressor run was started.

**Outcome (seed 1).**
* *Paper forensics (F0).* No official code or supplement was found. The PAPER-LITERAL object is the transmitted factor pair `[A_i, B_i]`, and whether it is a state, an increment or a gradient remains UNKNOWN. New paper inconsistencies: DR-22..28.
* *Micro-batch (F1).*
  * A virtual paper micro-batch (logical 16 x 2, streamed in chunks) is implemented and exact in code.
  * On the GPU it misses the pre-registered adapter criterion (5.0e-5 > 1e-5), through Adam first-step sign flips at near-zero gradients.
  * The baseline therefore keeps micro-batch 1 x 32, a PROMINENT deviation: gradient cosine with the paper's micro-batch 0.866 on a fixed batch.
* *Gauge (F2-F4).*
  * Raw LoRA factor norms (and the paper's 14.29) are gauge-dependent.
  * A balanced, gauge-invariant canonical representation is implemented and tested.
  * The per-round effective increment keeps 98.7 % of its energy at rank 8.
* *Gradients (F5).*
  * The real-gradient collection is bitwise identical to Phase 3.
  * Raw factor gradients have 0.69-0.72x the paper's per-element RMS; 1-3-step increments 0.08-0.15x.
* *Screen (F6).*
  * No representation passes (R0-R4).
  * For R2-R4 the fixed AE never reached the zero-output MSE on its own training data under the pre-registered `none` mode, so the screen does not measure their intrinsic 1/64 compressibility.
* *Baseline (F7).* LoRA-FT over 20 rounds: held-out loss 2.370 → 1.766. FAF-Identity is bitwise identical (13/13 checks). The resource-matched centralized reference reaches 1.756. Logical communication: 2,516,582,400 B two-way over 20 rounds.
* *Evaluation (F8).* Timed first (projected 23.2 / 28.7 min per model, under the 45-min rule), then the full run: MMLU test Base 45.25 % / LoRA-FT 45.71 % (paired McNemar p = 0.010); C-Eval val 59.08 % / 58.56 % (question-level accuracy identical, p = 1.0). One seed; not used for any representation decision.

Details: `docs/phase4_findings.md`. Evidence: `results/phase4/`. Deviations and discrepancies: `docs/deviations.md` (rows 35-43) and `docs/discrepancies.md` (DR-28, section D).
