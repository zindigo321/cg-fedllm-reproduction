# Source and provenance inventory

Everything a result in this repository depends on is pinned here. Hashes are SHA-256 unless stated.
Verification dates: 2026-09-30.

## 1. Paper versions (method/claims reference — reviewer decision R1)

| Version | Identifier | Date (authoritative source) | Role here |
|---|---|---|---|
| **ECAI 2025 (published)** | doi:10.3233/FAIA251320; *Frontiers in Artificial Intelligence and Applications* vol. 413, pp. 4257–4264; IOS Press; ISBN 9781643686318; ISSN 0922-6389 / 1879-8314 | Crossref `issued`/`published-print`: **2025-10-21**; Crossref record created 2025-10-22; publisher page `citation_online_date`: 2025/10/22 | **Primary** method/claims reference |
| arXiv v3 "Extended Version" | arXiv:2405.13746v3 | 2025-11-17 (arXiv API `updated`) | Extended/supplementary reference (appendix: hyperparameters, AE architecture, noise/SNR tables) |
| arXiv v1 / v2 | arXiv:2405.13746v1 / v2 | 2024-05-22 / 2024-05-24 | Ambiguity analysis only |

Licenses: ECAI version CC BY-NC 4.0 (Crossref license record from 2025-10-21); arXiv license CC BY-NC-SA 4.0.
No paper text or figures are redistributed here. Authors (ECAI/Crossref order): Huiwen Wu (ORCID
0000-0001-8471-4219), Xiaogang Xu, Deyi Zhang, Xiaohan Li, Jiafei Wu, Zhe Liu. The ECAI body text was diffed
word-by-word against the arXiv v3 main body during Phase 1: identical apart from reference numbering,
hyphenation and the abstract.

**Official code:** none public (Phase-1 search of all paper versions, GitHub search, Hugging Face Papers,
awesome-lists and author accounts). Everything here is an independent re-implementation.

## 2. Baseline framework whose behaviour is reconstructed

| Item | Pin | License | Use |
|---|---|---|---|
| FedIT / FederatedGPT-Shepherd | `JayZhang42/FederatedGPT-Shepherd@bcffa00e9642990ecc6210363a7f0dab91bef4dc` | Apache-2.0 | Semantics of client sampling, FedAvg weighting, Dolly partition, local Trainer defaults are re-implemented (no code copied) |
| `client_data_allocation.py` | SHA-256 `4f250129f8ffd6ed7df59cf4da2ee936cf54847d874dba0848fb9b2ec524d84c` | Apache-2.0 | Executed **only offline** to produce the partition oracle (below) |

### Partition oracle (how `tests/fixtures/shepherd_oracle_*.json` were produced)

1. Download the pinned `client_data_allocation.py` and `new-databricks-dolly-15k.json` and verify their hashes.
2. Insert one line after the JSON is read: `df['source_id'] = list(range(len(df)))`. Sorting and sampling
   use only `category`, so assignments are unchanged; this was verified by comparing every client file of an
   unmodified run with the instrumented run (identical content and order).
3. Run under **pandas 2.3.3 / numpy 2.4.6 / Python 3.11.16** (throw-away venv), modes `100 1` (Dirichlet) and
   `10 0` (shards), and record the per-client `source_id` lists.

`scripts/make_shepherd_oracle.py` automates this and regenerates both fixtures **byte-for-byte**.
Discovered facts (see `discrepancies.md`, P2-D1): under pandas ≥ 3 the shard partition differs (stable sort)
and the Dirichlet branch raises `ValueError: array is read-only` (copy-on-write).

## 3. Datasets (never committed; downloaded at run time)

| Dataset | Source / revision | Hash / size | License |
|---|---|---|---|
| Databricks-Dolly-15k (Shepherd's copy = first release) | `.../FederatedGPT-Shepherd/bcffa00.../new-databricks-dolly-15k.json` | `52e0c44e2155bfc920927f94f28cb06fd27122338ce3f0bafb2955d86f58e066`, 13,554,176 B, **15,015 records** (HF current release has 15,011) | CC BY-SA 3.0 |
| C-Eval | HF `ceval/ceval-exam@617524a00b307ff6f9933702f724131fe12ca7ce` (52 subjects; dev 260 / val 1,346 / test 12,342; test labels public since 2025-07-27) | per-file SHA-256 recorded in every evaluation result | CC BY-NC-SA 4.0 (data), MIT (code) |
| MMLU | HF `cais/mmlu@c30699e8356da336a370243923dbaf21066bb9fe` (57 subjects; dev 285 / val 1,531 / test 14,042; only per-subject files are downloaded, never `auxiliary_train`) | per-file SHA-256 recorded in every evaluation result | MIT |

Committed derivatives (labels and integer IDs only): `tests/fixtures/dolly_categories.json`,
`tests/fixtures/shepherd_oracle_*.json`, `manifests/*.json`.

| Manifest | SHA-256 | Content |
|---|---|---|
| `manifests/dolly_shepherd_dirichlet100_seed42.json` | `d4e1d12c3231ca4f0034c7e751132d75d87771f6dbdf3a6de1d6e89cf43e79df` | 100 clients (47–303 examples), 80 held-out, per-client 30/70 D1/D2 |
| `manifests/dolly_smoke_subset12_4clients.json` | `7451335b2430bd3a13aff328cfd333dfd6f6c16c1cb01e6fbef4ebe32a5dfb88` | 96-record subset (12/category), 4 clients (10–25), 16 held-out |

## 4. Models (never committed; HF cache outside the repository, `HF_HOME=D:\cgfed-cache\hf`)

| Model | Revision | Files downloaded (allow-list) | License | Phase-2 use |
|---|---|---|---|---|
| `JackFram/llama-160m` | `aca9b687d1425f863dcf5de9a4c96e3fe36266dd` | config, generation_config, **model.safetensors**, tokenizer files (not `optimizer.pt`, `pytorch_model.bin`, RNG states) | Apache-2.0 | Tier-C smoke, micro-benchmark |
| `Qwen/Qwen1.5-0.5B` | `8f445e3628f3500ee69f24e1303c9f10f5342a39` | config, generation_config, model.safetensors, tokenizer/vocab/merges, LICENSE | Tongyi Qianwen **Research** License (non-commercial research use; no use of outputs to improve other LLMs) | Evaluator validation, micro-benchmark |
| `Qwen/Qwen1.5-1.8B` | `7846de7ed421727b318d6605a0bfab659da2c067` | same pattern | Tongyi Qianwen Research License | Bounded micro-benchmark only (R8) |

Downloaded bytes (allow-listed files only, single safetensors file each): llama-160m 652,029,182; Qwen1.5-0.5B
1,250,659,408; Qwen1.5-1.8B 3,685,176,753 (downloaded only for the bounded micro-benchmark, after the GPU smoke
gates were green; 120 s).

No 7B model, no Alpaca, no original LLaMA weights were downloaded (R9/R19).

## 5. Evaluation resources (committed, small)

| File | Source | Upstream hash | License |
|---|---|---|---|
| `src/cg_fedllm/evaluation/resources/ceval_subject_mapping.json` | `hkust-nlp/ceval@cba65ae93bcf189149ced9f66ae0c958201faed9` (verbatim, CRLF preserved via `-text`) | `671018e9d1ac8e51e8c3ea02574c89be8ff7660ebe67c8ec31b30b9035e064a7` | MIT |
| `.../mmlu_categories.json` | `hendrycks/test@4450500f923c49f1fb1dd3d99108a0bd9717b660:categories.py` (transcribed) | `e977fffa...` (of `categories.py`) | MIT |
| `.../ceval_hard_subjects.json` | C-Eval paper definition (8 subjects) | n/a | n/a |

## 6. Software environment

Dedicated conda env `cgfedllm` (`D:\conda_envs\cgfedllm`), Python 3.11.16; torch 2.14.0+cu130 (CUDA 13.0),
transformers 5.17.0, peft 0.21.1, accelerate 1.15.0, datasets 5.0.1, bitsandbytes 0.50.2, safetensors 0.8.0,
lm-eval 0.4.13, huggingface_hub 1.33.0, tokenizers 0.23.2, numpy 2.4.6, pandas 3.0.6, pyarrow 25.0.1,
PyYAML 6.0.3, pytest 9.1.1, ruff 0.16.9. All candidate pins resolved and passed the compatibility gate (CPU
fp32 and CUDA bf16 LoRA forward/backward, bitsandbytes NF4 + int8 kernels on CUDA 13, lm-eval task registry)
with **no version substitutions**. Full freeze: `requirements/lock-win-py311-cu130.txt`.

Hardware: Windows 11 Home (China) 10.0.26100, Intel i9-14900HX, 31.7 GB RAM, NVIDIA RTX 4060 Laptop GPU
8,188 MiB (CC 8.9), driver 595.97 (CUDA 13.2). The paper used one RTX 4090 24 GB, Ubuntu 22.04,
PyTorch 2.2.1, CUDA 12.4 (v3 appendix).

## 7. Run provenance

Every run directory contains `config.resolved.yaml`, `config.sha256.json` (full and resume-identity hashes)
and `run_metadata.json` (Git commit/branch/dirty state and dirty files, Python and package versions,
CUDA/GPU facts incl. `nvidia-smi`, model ID/revision/files, dataset/manifest hashes, all seeds,
representation, layout, aggregation and TGAP source mode). Evaluation results additionally record the
benchmark repo, revision and per-file SHA-256 digest.
