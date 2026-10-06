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
- [x] Apply the pre-registered viability gates before operational AE-backed FAF

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

### Phase 5: AutoEncoder fit control (P5-A)

- [x] Pre-register the P5-A control for R2-R4 (v1, `docs/phase5_preregistration.md`; historical, never executed)
- [x] Approve the revised protocol v2 for implementation (`docs/phase5_preregistration_v2.md`)
- [x] Freeze the v2.1 addendum on recovery and publication states (`docs/phase5_preregistration_v2_1.md`)
- [x] Implement P5-A v2 and validate it on synthetic data (implementation review)
- [x] Run the separately authorised real CPU preflight 1 (passed)
- [x] Run the separately authorised P5-A v2 + v2.1 GPU invocation 1 (closed; six completed controls)
- [x] Record the negative fit outcomes and review the complete invocation-1 evidence

All six controls completed 3,000 iterations and received `FIT FAIL`. R2/R3
(primary) and R4 (secondary) each have `CAPACITY FIT NOT DEMONSTRATED` under
the frozen protocol; this does not establish incompressibility or insufficient
general AE capacity. No downstream experiment is authorised by these outcomes.
See `docs/phase5_p5a_v2_findings.md` and `results/phase5/p5a_v2/`.

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

## 7. Current status and evidence

The repository has completed the infrastructure, smoke validation, compressor
diagnosis, representation forensics, and a seed-1 Tier-B baseline evaluation.
These milestones do **not** establish full reproduction of the paper's
experimental claims.

- **Phase 2 — infrastructure and smoke validation:** the federated LoRA,
  compression, TGAP, FAF, evaluation, provenance, and resume machinery was
  implemented and validated at smoke scale. This phase does not reproduce a
  paper-reported result. See `docs/phase2_validation.md` and `results/phase2/`.
- **Phase 3 — Tier-B compressor diagnosis:** no pre-registered AutoEncoder
  candidate passed the viability gates at the 1/64 target ratio. The planned
  operational AE-backed FAF experiment was therefore not run. This is a
  recorded negative result, not an unfinished implementation task. See
  `docs/phase3_preregistration.md`, `docs/phase3_findings.md`, and
  `results/phase3/`.
- **Phase 4 — representation forensics and seed-1 baseline:** representation,
  micro-batch, gauge, state/update/gradient, screening, baseline, communication,
  and downstream evaluation work was completed. No representation passed the
  pre-registered R0-R4 screen, and the project still has no viable operational
  AE-backed compressor at the paper's target ratio. See
  `docs/phase4_preregistration.md`, `docs/phase4_findings.md`,
  `docs/representation_forensics.md`, and `results/phase4/`.

- **Phase 5 — P5-A fit control:** the separately authorised real preflight 1
  passed, and GPU invocation 1 completed all six 3,000-iteration controls
  under the frozen v2 + v2.1 protocol on 2026-10-06. All six controls are
  `FIT FAIL`; each of R2/R3 (primary) and R4 (secondary) has
  `CAPACITY FIT NOT DEMONSTRATED` under that fixed protocol. This result
  does not establish incompressibility, insufficient general AE capacity,
  or a cause of earlier screening failures, and authorises no downstream
  experiment. The complete artifact review found no discrepancy within its
  stated limits. See `docs/phase5_p5a_v2_findings.md` and
  `results/phase5/p5a_v2/`; v1 remains historical and was never executed.

The remaining project-level goals are to determine whether a viable operational
compressed FAF path can be established, and to reproduce the
paper-scale claims across the required settings and seeds.

Cross-phase evidence boundaries, deviations, and unresolved discrepancies are
tracked in `docs/evidence_ledger.md`, `docs/deviations.md`,
`docs/discrepancies.md`, and `docs/provenance.md`.

## 8. Quickstart

Windows, conda; see `docs/reproduction_protocol.md` for the full protocol and
experiment acceptance-record requirements.

```bash
conda create -n cgfedllm python=3.11 -y && conda activate cgfedllm
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements/base-win-cu130.txt
python -m pip install -e . --no-deps
pytest -q -m "not gpu and not model"
cgfed smoke --config configs/smoke/llama160m_smoke.yaml
```

Passing the Linux CPU CI or the local CPU test suite is a repository integration
gate; it is not a substitute for GPU, model-level, or scientific experiment
validation.
