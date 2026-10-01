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

## Scope (reviewer decisions)

U-Former and 1-D CNN AEs, DP, all 7B experiments, Qwen-7B rows, formal replicate policy and mechanistic
control experiments are deferred (R5, R9, R10, R15, R16). Controls are implemented and unit-tested.
