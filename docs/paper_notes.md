# Paper notes: CG-FedLLM (ECAI 2025; arXiv:2405.13746)

Reference hierarchy (reviewer decision R1): **ECAI 2025** published text defines the method and claims;
**arXiv v3** ("Extended Version") supplies the appendix (hyperparameters, AutoEncoder architecture,
noise/SNR tables); **v1/v2** are used only to analyse ambiguities. The ECAI body equals the v3 main body.

## 1. What the paper proposes

Federated LoRA fine-tuning (FedIT/Shepherd style) in which every client's uploaded LoRA factors are
compressed by a convolutional **AutoEncoder**: the **encoder** runs on the client, the **decoder** on the
server. Training has two stages:

* **TGAP** (Temporal-ensemble Gradient-Aware Pre-training): a dataset split **D1** (30 %) is used to collect
  LoRA "gradients" (A/B factors) across clients and time; the AutoEncoder is trained on them with an MSE
  reconstruction loss (Adam, lr 2e-4).
* **FAF** (Federated AutoEncoder-Involved Fine-tuning): federated fine-tuning on **D2** (70 %) where each
  selected client sends `Enc([A_i, B_i])`; the server decodes, aggregates in the "LoRA subspace" and updates
  the global model.

Claims: per-client uplink payload reduced to **1.56 %** (latent `[64, 64, 32]` vs input `[1, 4096, 2048]`
for LLaMA-7B, r = 8, q/k/v/o, 32 layers = 8,388,608 elements), equal or better C-Eval/MMLU scores than
uncompressed federated LoRA ("LoRA-FT") and centralized training ("Cent"), robustness to DP noise, and AE
generalisation across datasets.

## 2. Reconstructed one-round algorithm (as implemented in Phase 2)

```
server global adapter G_t (A, B of q/k/v/o in all layers)
  -> Shepherd sampler: K = max(int(f N), 1) clients, RandomState(t)
  -> each selected client trains LoRA from exactly G_t (fresh AdamW, linear decay, clip 1.0)
  -> representation: adapter_state (end state) | adapter_delta (end - G_t)       [R2]
  -> Phi (layer_major_qkvo_AtB): [A^T | B] blocks -> X in R^{1 x d x 2*4*L*r}  [R4]
  -> Enc(X) on client -> latent [64, d/64, W/64] (1/64 of X)                    [R5]
  -> server: Dec(latent) -> Phi^-1 -> (G_t + delta if delta)
  -> aggregation of A and B independently: sample_weighted_mean (primary)      [R3]
  -> the aggregate replaces the global adapter; checkpoint; next round
```

## 3. Hyperparameters stated by the paper (v3 appendix)

| Setting | LLaMA/Alpaca + Dolly | LLaMA + C-Eval-dev | Qwen + MMLU |
|---|---|---|---|
| communication rounds | 20 | 5 | 8 |
| client selection fraction | 0.05 (100 clients) | 1.0 (3 clients) | 1.0 (3 clients) |
| local batch / micro-batch | 32 / 16 | 24 / 8 | 8 / 4 |
| local learning rate | 1.5e-4 | 2.0e-4 | 3.5e-4 |
| LoRA rank | 8 | 16 | 8 |
| local epochs | 1 | 3 | 2 |

Also stated: Dirichlet α = 0.5 over Dolly categories into 100 segments; D1:D2 = 3:7; AE loss = MSE, Adam,
lr 2e-4; LoRA targets Q/K/V/O; the 1-D CNN / ResNet / U-Former AE variants; 1 × RTX 4090 24 GB.

Not stated (see `evidence_ledger.md` UNKNOWN items): LoRA α/dropout, the transmitted representation
(state vs delta), TGAP batch size/iterations/split, the evaluation protocol, the Cent configuration,
checkpoint selection, seeds, exact checkpoints.

## 4. What Phase 2 does and does not establish

Phase 2 builds and validates the machinery (data, evaluation, baselines, Phi, AE, TGAP, FAF, controls) at
smoke scale. It does **not** reproduce any paper number and does **not** resolve which TGAP source mode or
representation the paper used. Smoke-scale AE diagnostics (`phase2_validation.md`) are recorded as
evidence for Phase 3, not as conclusions about the paper.
