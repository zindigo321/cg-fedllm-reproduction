# Evidence ledger

Confidence: **H** = directly specified by primary evidence; **M** = strongly implied or inherited from the
identified baseline (Shepherd); **L** = engineering inference requiring validation.
Status column: how Phase 2 implements it (config key in brackets).

| ID | Claim / parameter | Value / interpretation | Source | Conf. | Phase-2 status |
|---|---|---|---|---|---|
| L01 | Target version | ECAI 2025 (= v3 main body) + v3 appendix | Crossref, arXiv, word diff | H | adopted (R1) |
| L02 | Official code | none public | Phase-1 search | H | full re-implementation |
| L03 | FL framework | FedIT / Shepherd | paper §4.1 | H | semantics re-implemented |
| L05 | LoRA targets | q, k, v, o | appendix + figure filenames | H | `lora.target_modules` |
| L06 | LoRA rank | 8 (Dolly, MMLU); 16 (C-Eval-dev) | appendix table | H | `lora.r` |
| L07 | LoRA alpha / dropout | UNKNOWN -> 16 / 0.05 | Shepherd defaults | M | `lora.alpha`, `lora.dropout` |
| L08 | Base precision | 8-bit (LLM.int8) | Shepherd; Table 3 LLM memory | M | `model.quantization` (fp32 smoke) |
| L09 | Local optimizer | AdamW, linear decay, 0 warm-up, wd 0, clip 1.0, fresh per round | Shepherd + Trainer defaults (verified in installed transformers) | M | explicit `local_train.*` |
| L10 | Rounds / fraction / batch / lr / epochs | see `paper_notes.md` §3 | appendix table | H | `federated.*`, `local_train.*` |
| L11 | cutoff_len / template / train_on_inputs | 512 / Alpaca / True | Shepherd | M | `data.*` |
| L12 | Client sampling | K = max(int(fN), 1), RandomState(round) | Shepherd | M | `federated.sampler` (regression-tested) |
| L13 | Dolly partition | Dirichlet(0.5) per category, 100 clients, min size 40, 10/category held out, seed 42 | paper + Shepherd | H (alpha) / M | `data.*` (oracle-equivalent) |
| L14 | D1/D2 | 30/70; granularity UNKNOWN | paper §4.8 | M / L | per-client split (R7, INFERRED) |
| L16 | Transmitted object | factor pair (A_i, B_i) (H); full state vs delta UNKNOWN | Alg. 1; Shepherd | H / L | `federated.representation` (R2) |
| L17 | Aggregation | A and B independently; normalised sample-weighted mean | Alg. 1 "LoRA subspace"; Shepherd | H / M | `federated.aggregation` (R3) |
| L18 | Global update | aggregate replaces the adapter (eta == 1) | Shepherd | M | implemented; `literal_sum` diagnostic only |
| L20 | Optimizer state | reset every client round | Shepherd | M | enforced (`reset_optimizer_each_round`) |
| L21 | AE sharing | one AE for all clients and layers | v1 text; Table 1 shape | H / M | one codec |
| L22 | AE input layout | [1, 4096, 2048] (H); block ordering UNKNOWN | appendix + Table 1 | H / L | `layer_major_qkvo_AtB` + alternative (R4) |
| L23 | ResNet-3 AE | 6 stride-2 stages to 64 ch, 3+3 residual blocks, Tanh | appendix table + diagram; parameter/MAC match | H / M | implemented (R5) |
| L26 | Compression ratio | latent / input elements, uplink per client | paper §3.2.3 | H | exactly 1/64 (tested) |
| L27 | TGAP training | MSE, Adam, 2e-4 (H); batch / iterations / split / normalisation UNKNOWN | appendix | H / L | `autoencoder.*` defaults INFERRED |
| L28 | TGAP source | FL on D1 (ECAI/v3) vs local training without FL (v1) | text versions | L | both modes (R6); smoke default local |
| L30 | TGAP/FAF LoRA-init coupling | UNKNOWN; Shepherd leaves A unseeded | code audit | L | seeded init `lora.init_seed` |
| L31 | SNR definition | sum-of-squares / per-element MSE (derived from the denoising table) | arithmetic | M | `snr_paper` + `snr_db` reported |
| L33 | DP | GDP, eps in {0.25, 2, 8}; delta, C, placement UNKNOWN | paper §4.4–4.5 | L | deferred |
| L34 | Evaluation protocol | "test sets"; shots/prompt/scoring UNKNOWN | paper | L | `reference_eval_v1` (R11), ours |
| L35 | C-Eval averages | categories/average = subject mean; Hard = 8 subjects | official C-Eval | H | implemented + tested |
| L36 | MMLU averages | question-weighted | official evaluate.py; lm-eval | H | implemented + lm-eval cross-check |
| L37 | Cent | centralized fine-tuning ("contrastive" is a typo); configuration UNKNOWN | text versions | H / L | `cent_smoke_sample_matched` only (R13) |
| L38 | Checkpoint choice | paper mixes epochs (10/13/17/19) | appendix | L | final round only (R14) |
| L40 | Seeds | UNKNOWN | — | L | explicit derived seeds |

## UNKNOWN items and how Phase 2 handles them

| Item | Least-assumptive choice | How sensitivity will be tested (Phase 3+) |
|---|---|---|
| state vs delta | both implemented; Shepherd-compatible `adapter_state` default for baselines | identical pipeline with `representation=adapter_delta` |
| local vs federated TGAP | both implemented, same schema | AE trained on each; FAF compared |
| Phi block ordering | `layer_major_qkvo_AtB` | `module_major_qkvo_AtB` and others via layout registry |
| LoRA alpha/dropout | 16 / 0.05 | alpha in {8, 16} |
| AE batch/iterations/split/normalisation | 4 / 600 / temporal / none | sweeps |
| evaluation protocol | `reference_eval_v1` | 0-shot and generation diagnostics |
| Cent configuration | sample-matched (smoke only) | formal Cent variants |
| checkpoint policy | final round | report all-round held-out loss |
