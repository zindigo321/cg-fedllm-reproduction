# Tier-C smoke summary (PHASE2-SMOKE — correctness run, not a paper reproduction)

clients: 4; total stage time: 74.84 s

## Stage status

- lora_ft: complete
- faf_identity: complete
- faf_identity_delta: complete
- faf_autoencoder: complete
- cent: complete

## Equivalence checks

| check | bitwise | relative L2 | tolerance | pass |
|---|---|---|---|---|
| identity_state_vs_lora_ft | True | 0.000e+00 | 1e-06 | True |
| identity_delta_vs_lora_ft | False | 1.861e-07 | 1e-06 | True |
| resume_vs_uninterrupted | True | 0.000e+00 | None | True |
| n1_fl_vs_local | True | 0.000e+00 | 1e-06 | True |

## Held-out loss (PHASE2-SMOKE)

| stage | initial | final round |
|---|---|---|
| lora_ft | 3.5882388952387427 | 3.4785504438504353 |
| faf_identity | 3.5882388952387427 | 3.4785504438504353 |
| faf_identity_delta | 3.5882388952387427 | 3.4785503621160445 |
| faf_autoencoder | 3.5882388952387427 | 8.02149263017346 |
| cent | 3.5882388952387427 | 3.4666561283792268 |

## Uplink bytes (logical) per run

| stage | logical | raw fp32 |
|---|---|---|
| lora_ft | 9437184 | 9437184 |
| faf_identity | 9437184 | 9437184 |
| faf_autoencoder | 147456 | 9437184 |
| cent | 2359296 | 2359296 |

## AutoEncoder (PHASE2-SMOKE)

input [1, 768, 768] -> latent [64, 12, 12]; CR 0.015625
- train/autoencoder: pooled rel. sq. error 2.6283351321279995, mean innovation ratio 50634.1908025089
- train/zero: pooled rel. sq. error 1.0, mean innovation ratio 19264.73919239673
- train/train_mean: pooled rel. sq. error 2.0332525948341185e-05, mean innovation ratio 0.3917013100002654
- val/autoencoder: pooled rel. sq. error 2.628375825948026, mean innovation ratio 25403.90284472696
- val/zero: pooled rel. sq. error 1.0, mean innovation ratio 9665.24791262329
- val/train_mean: pooled rel. sq. error 0.00014930018778755238, mean innovation ratio 1.4430255394076064

## Stage timings (s)

- lora_ft: 6.18
- tgap_local: 4.02
- tgap_federated: 4.03
- ae: 25.78
- faf_identity: 6.23
- faf_identity_delta: 6.27
- faf_autoencoder: 6.26
- cent: 4.15
- resume: 6.28
- eval: 5.64
