# Representation forensics: what is CG-FedLLM's `G`? (Phase 4, stage F0)

This is an evidence audit, committed **before** any Phase-4 experiment was run or interpreted. It records what each source says about the object
that the CG-FedLLM AutoEncoder encodes, which is called `G`, `G_i`, `G_i^t`, "gradients", "low-rank gradients", `ΔW`, `[A_i, B_i]` or
`B_i A_i` in different places.

Wording follows the reviewer's categories:

| Label | Meaning |
|---|---|
| PAPER-LITERAL | stated verbatim by the paper |
| RECOVERED-BASELINE | Shepherd behaviour |
| PHASE4-FORENSIC | an alternative under investigation |
| DERIVED | our arithmetic on reported numbers |
| UNKNOWN | not established |

Phase-3 results (the paper-literal state reconstruction was not viable) are frozen and are not reinterpreted here.

## 1. Sources examined

**Paper versions:**
* arXiv v1, v2 and v3: LaTeX sources and PDFs.
* The ECAI 2025 published PDF (text extraction). Its body equals v3 apart from numbering and hyphenation (Phase 1 diff).

The files were downloaded in Phase 1 and kept outside Git. Locations below are in the form "version, file or section".

**Baseline code:** FedIT/Shepherd @ `bcffa00`, files `fed_utils/client.py` and `fed_utils/model_aggregation.py`.

**Later work by the same authors:** DR-Encoder (arXiv 2412.17053, AAAI-25; identical author list).

### 1.1 Search record (2026-10-01)

| Where | Query | Result |
|---|---|---|
| GitHub repository search API | `CG-FedLLM` | 1 hit: this reproduction |
| GitHub repository search API | `CG-FedLLM in:readme` | 4 hits: this reproduction, `Lydia-yang/Awesome-Federated-LoRA` (paper link only), `solidlabnetwork/awesome-distributed-LLM` (code column "N/A"), a systematic-review list |
| GitHub repository search API | `cgfedllm`; `compress gradients federated LLM autoencoder` | 0 hits |
| Web search | "CG-FedLLM github code Compress Gradients ..." | the paper (arXiv / ResearchGate / review sites); this reproduction; no author code |
| Web search | author names + "federated fine-tuning LLM LoRA AutoEncoder" | DR-Encoder (AAAI-25, arXiv 2412.17053) |
| arXiv API | author `Wu_Huiwen` | 15 entries. Only CG-FedLLM and DR-Encoder concern this topic; the others (Iter-AHMCL, DP papers, unrelated fields) do not |
| Hugging Face papers | `/papers/2405.13746` | HTTP 404 (no page) |
| DR-Encoder full text | — | no code link |

**Conclusion:** no official code, supplementary source or issue discussion was found. "Absence" here means absence from these searches only.

## 2. Evidence table

Notation follows each source. "Factor stack" means the per-module LoRA factors concatenated into one 2-D input of 8,388,608 elements for
LLaMA-7B (r = 8; q/k/v/o; 32 layers).

| Source / version (location) | Exact mathematical object implied | Shape implied | Used in TGAP? | Used in FAF? | Confidence | Contradiction |
|---|---|---|---|---|---|---|
| v1/v2 §3.1 Overview | `G_i` = "a set of gradients to update the model"; combined into the "polymerized gradient G to update the central model weight, as ΔW_global"; clients receive `ΔW_i` | unspecified | yes | yes | low (generic) | "gradient" and "model update ΔW" are used as synonyms |
| v1/v2 §3.2, loss | `L = ‖G − G̃‖₂`; commented-out continuation `= ‖[A, B] − [Ã, B̃]‖₂` | factor pair `[A, B]` | yes | – | medium (author source comment) | the loss is called a gradient reconstruction; the comment equates `G` with the factors |
| v1/v2 §3.2, TGAP | "train `M_i` with its own `D_i` **without FL**", "collect the gradient sequences `G_{i,t}`", t = iteration | unspecified | yes | – | medium | v3 says FedLLM pre-training (DR-16) |
| v1/v2 §3.3, communication | transmitted parameters `4096 x 8 x 2 x 4 x 32 = 8,388,608`; "ensures accurate reconstruction of the model parameter increment ΔW" | factor stack | yes | yes | high (count) / medium (ΔW) | the AE reconstructs factors, not ΔW (= B A, 4096 x 4096 per module) |
| v1/v2 §3.4 FAF text + Alg. 1 | "updates the local model parameters in a low-rank format, specifically `ΔW_i^t = B_i A_i`"; `[Ā_i, B̄_i] = Enc[A_i, B_i]`; `[Ã, B̃] = [Σ Ã_i, Σ B̃_i]`; `W^{t+1} = W^t + ΔW_i^t = W^t + B̃Ã` | factor pair per client | – | yes | high (as written) | per-client `ΔW_i^t` is used for the global update; factor sums do not give Σ B_i A_i (DR-15) |
| v1 §4.4 SNR table | first row `‖[A, B]‖²₂ = 14.29`: "the power of the ℓ2 norm for the **low-rank decomposition of input gradients**"; second row `‖[Ã, B̃] − [A, B]‖²₂`: "the power of MSE loss" | factor stack | yes | – | high (labels) | `‖·‖²` is a squared norm, but the second row is an MSE (per element) |
| v1 §4.4, commented-out paragraph (removed in v2) | "Assume `G = Concatenate[A_i, B_i]_{i=1}^N` represents the combined transmitted A's and B's" | N clients' factor stacks | yes | yes | medium (comment) | a concatenation over N clients would make ‖G‖² N times larger again |
| v1 §4.4 / v3 App. "Surface" figure | text: max values "vary from 0.0018 to 0.0010", min symmetric, "on the order of 10⁻³ ... converging toward 0 ... The zero gradient value indicates the convergence" | factor stack over 20 iterations, 128 matrices | yes | – | text vs figure conflict | the figure's z-axis is 0.010-0.018, 10x the text (DR-09) |
| v2 | identical to v1 apart from layout, the removed commented paragraph and affiliations | – | – | – | – | – |
| v3 = ECAI §3.1 | `G_i` = "a set of gradients to update the model"; polymerized gradient updates `W_global`; "the gradient can be first compressed by other methods like Low-rank decomposition in LoRA" | unspecified | yes | yes | low (generic) | "gradient" vs "update" |
| v3 = ECAI §3.2, TGAP | "`D1` is utilized for the pre-training of FedLLM to collect the intermediate gradients"; `G = [G_i^t]`, i = client, t = "iteration step to record" | unspecified | yes | – | medium | v1 says "without FL" (DR-16) |
| v3 = ECAI, Alg. 1 and §3.3 FAF | "obtain `G_i^t = B_i A_i`"; `[Ā_i, B̄_i] = Enc[A_i, B_i]`; sum of factors; `W^{t+1} = W^t + η G̃^t = W^t + η B̃Ã`; commented line `W_i^{t+1} = W_i^t + ΔW_i^t = W_i^t + B_i A_i` | `G_i^t`: d x d product; encoded: factor pair | – | yes | high (as written) | `G` is defined as the product, but the factors are encoded; v1 called the same object `ΔW_i^t` |
| v3 = ECAI Table 1 | `G` and `G̃` are `[1, 4096, 2048]` (ResNet); `‖G‖²₂ = 14.29` for all three AEs; `‖G − G̃‖²₂ = 5.06e-12` (ResNet); SNR = row 1 / row 2 | factor stack, 8,388,608 elements | yes | – | high | the shape is the factor count, not the product (d x d); the same 14.29 appears for every AE (DR-10) |
| v3 = ECAI §3.2, communication | the ΔW sentence of v1 is commented out; CR counts encoder output / input (uplink) | factor stack | – | yes | high | see the downlink row below |
| v3 = ECAI §3.5, security | "local gradients shared during federated aggregation"; "clients are equipped with both the encoder and decoder. Consequently, the only data transmitted during both the uplink and downlink procedures are the encrypted gradients" | latent | – | yes | medium | the communication analysis counts only the uplink (new DR-26) |
| v3 App. B.1 "Gradients Low-rank Decomposition Distributions" | "we decompose the transformer gradients `G_K, G_Q, G_V, G_O` into the product of two low-rank matrices `G_K = B_K A_K`"; A is 8 x 4096, B is 4096 x 8, 128 of each, "**collected over 20 training epochs** ... used as input for training an AutoEncoder"; distributions "resemble a Gaussian distribution, predominantly centered around 0, with an x-axis range from −0.01 to 0.01 ... as iterations increase, the matrix values tend to cluster more tightly around 0"; histogram x-axis "Gradient Value", y-axis epochs 0-19 | factor stack per epoch | yes | – | high (figures and text agree) | inconsistent with PEFT's Kaiming-uniform A state (flat over ±0.0156) under an absolute-state reading (new DR-24) |
| v3 App. "AutoEncoder structure" | "we compress the gradients processed with LoRA, as A and B"; commented-out "the difference between two sequential model parameters in low rank structure, `ΔW = B A`"; concatenation of 8 x 4096 blocks gives 2048 x 4096 | factor stack (transposed relative to Table 1) | yes | yes | medium | 2048 x 4096 vs Table 1 `[1, 4096, 2048]` (same elements; new DR-27) |
| v3 App. "Surface Visualization of Low-rank Gradients" | "matrix visualizations of **B A** during the federated fine-tuning ... a slight change in the updated model parameters, roughly 10⁻³. The reconstruction loss was estimated to be around 10⁻⁷"; images are 16 x 16 grids, colour scale ±2.5e-5 | product (visualised) | – | yes | low (crop/downsampling unknown) | reconstruction loss about 1e-7 vs Table 1's 5.06e-12 (new DR-22) |
| v3 App. noise table + text | `‖[A, B]‖²₂ = 14.29` at every noise level; LoRA-FT MSE = σ²; Compress-FT MSE 4.59e-7 to 5.25e-7 for σ = 5e-5 ... 5e-1; "[A, B] are the original low-rank gradients"; "restoration of the original, noise-free **local updates** A and B"; DP: "clip the local gradients to make the sensitivity equal 1" | factor stack | – | yes (DP variant) | high (numbers) | the output error is nearly independent of the input noise (new DR-23); vs Table 1 MSE (new DR-22) |
| ECAI §2, related work | "the client train its own model with a local dataset and send the **increment gradients** to the server" | – | – | – | low (generic FL) | – |
| Shepherd @ `bcffa00` (RECOVERED-BASELINE) | client uploads `get_peft_model_state_dict(model)`, the full current A/B **state**; server: sample-weighted average of the A and B states, then `set_peft_model_state_dict` | absolute adapter state | – | (baseline) | high (code) | CG-FedLLM's Alg. 1 replaces averaging with sums and `W + η B̃Ã` |
| DR-Encoder (AAAI-25, same authors) | "intermediate gradients are initially collected using low-rank decomposition (LoRA)"; `G_i^t = A_i^t B_i^t`; per-layer, per-epoch mean/variance of the "low-rank gradients" used to sample synthetic Gaussian training data for the AE; cites CG-FedLLM as "FedCG" | LoRA factors (naming swapped) | yes (statistics) | – | medium | whether A/B are current weights or changes is **not stated**; LoRA initialisation not stated |

## 3. Geometry-normalised, gauge-aware comparisons (DERIVED)

**Geometry of the paper's numbers.**
* LLaMA-7B (d = 4096, 32 layers), r = 8, q/k/v/o: n = 8,388,608 factor elements (A: 4,194,304; B: 4,194,304).
* LoRA scaling s = α/r: **UNKNOWN**, because the paper does not state α.
* The representation is the factor stack, as Table 1's shape shows.

**Gauge.**
* LoRA factors are non-identifiable: for any invertible r x r matrix Q, (BQ)(Q⁻¹A) = BA.
* Factor norms therefore depend on the gauge. The smallest possible `‖A‖² + ‖B‖²` over all gauges equals `2 ‖B A‖_*`, twice the nuclear norm; the balanced factorisation attains it.
* `‖G‖² = 14.29` is a statement about one specific (unknown) factorisation, not about ΔW.

**The comparisons:**

| Quantity | Value | Gauge-invariant? | Note |
|---|---|---|---|
| `‖G‖²₂ = 14.29` over n = 8,388,608 | per-element mean square 1.70e-6, **RMS 1.305e-3** | no | PAPER-REPORTED norm; RMS DERIVED |
| PEFT default A (Kaiming uniform, a = √5) at d = 4096 | A ~ U(±1/√4096 = ±0.0156); per-element mean square 8.14e-5; **A alone: 341.3** | no | 24x more than 14.29 for one client's A alone; N clients' concatenation (v1 comment) would be N times larger again (DR-10) |
| Text max/min ("0.0018 ... 0.0010") | the same order as the RMS of 1.3e-3 | no | consistent with `‖G‖²` only if the values are concentrated near their maximum (a near-two-point, "sign-like" distribution) |
| Figure max/min axis (0.010 ... 0.018) | matches the Kaiming bound 0.0156 | no | consistent with an absolute A state, inconsistent with the text (DR-09) |
| Histogram description (Gaussian-like, ±0.01, narrowing over epochs) | inconsistent with the flat Kaiming-uniform A; consistent with sums of many optimizer steps or with gradients | – | DR-24 |
| Our Tier-B Qwen1.5-1.8B data (Phase 3, d = 2048, n = 3,145,728) | state: A RMS 0.0128; one-round delta: RMS 1.8e-4, max 3.0e-4 (1-3 Adam steps) | no | the paper's RMS 1.3e-3 would need about 10-20 Adam steps per collected increment at lr 1.5e-4 |
| Noise table: Compress-FT MSE 4.59e-7 at σ = 5e-5 and 5.21e-7 at σ = 5e-1 | input noise power grows 10⁸x (2.5e-9 to 0.25 per element, against 1.7e-6 signal power) while the output error changes by 14 % | – | the decoder output is nearly insensitive to its input (DR-23); 5e-7 is 30 % of the per-element signal power |

## 4. Consistency matrix: which reading survives which clue?

Candidate meanings of the encoded `[A_i, B_i]`:
* **S:** absolute LoRA state, PEFT-default initialisation (the Phase-3 primary).
* **D:** per-round or per-epoch factor increment (Phase-3 sensitivity).
* **Gr:** raw factor gradients `∂L/∂A, ∂L/∂B`.
* **P:** the product `B A` (ΔW).

Marks: ✓ consistent, ✗ inconsistent, ~ possible with extra assumptions, – not informative.

| Clue | S | D | Gr | P |
|---|---|---|---|---|
| Alg. 1 transmits `[A_i, B_i]` "after LoRA training"; Shepherd baseline transmits states | ✓ | ~ (if A_i, B_i denote changes) | ✗ | ✗ |
| Table 1 shape = factor count | ✓ | ✓ | ✓ | ✗ |
| `‖G‖² = 14.29` (RMS 1.3e-3) | ✗ (PEFT init) | ✓ (with ~10-20 steps) | ~ (unknown loss scale) | – |
| Histograms: Gaussian, ±0.01, narrowing, "Gradient Value" | ✗ (A would be flat) | ✓ | ✓ | – |
| Text: values ~1e-3 "converging toward 0", "zero gradient ... convergence" | ✗ | ✓ | ✓ | – |
| Figure axis 0.010-0.018 | ✓ | ~ | ~ | – |
| B·A visualisation magnitude ~1e-5 | ✓ (A ~1e-2 x B ~1e-3) | ~ | – | ✓ |
| "noise-free local updates A and B"; "increment gradients"; "ΔW = BA", "difference between two sequential model parameters" | ~ | ✓ | ~ | ✓ |
| DP: "clip the local gradients to sensitivity 1" | ~ | ✓ | ✓ | – |

**Synthesis:**
* No single reading is consistent with every statement in the paper.
* The magnitude and distribution statements (norm, histograms, text ranges, "converging to zero") favour increment- or gradient-like objects over PEFT-default absolute states.
* The literal transmission in Algorithm 1, the max-surface axis and the B·A visualisation scale are compatible with absolute states.
* The PAPER-LITERAL object remains the factor pair `[A_i, B_i]` exactly as Algorithm 1 transmits it. Whether it is a state, an increment or a gradient is **UNKNOWN**.
* Per reviewer decision R5, 14.29 is a discrepancy clue, not an oracle: it is gauge-dependent and its α/r scaling is unknown.

## 5. Consequences for Phase 4 (no result is anticipated)

The pre-registered screen (F6) covers:

| ID | Representation | Basis |
|---|---|---|
| R0 | `adapter_state` | PAPER-LITERAL reading of Alg. 1 + RECOVERED-BASELINE; Phase-3 negative result reused, frozen |
| R1 | `adapter_delta` | consistent with the magnitude clues; Phase-3 negative result at our 1-3-step scale reused |
| R2 | `balanced_effective_state` | PHASE4-FORENSIC: a gauge-invariant state |
| R3 | `balanced_effective_delta_r8` | PHASE4-FORENSIC: a gauge-invariant increment, truncated to rank 8 |
| R4 | `mean_step_gradient` | PHASE4-FORENSIC: the literal "gradient" reading |

None of R2-R4 is paper-specified, and a positive screen would not make any of them PAPER-LITERAL.

## 6. New paper inconsistencies

DR-09 and DR-10 already exist in `discrepancies.md`. New entries:
* **DR-22:** reconstruction error of the ResNet AE. Table 1 gives 5.06e-12; the appendix noise table gives 4.59e-7-5.25e-7; the surface appendix gives "around 10⁻⁷".
* **DR-23:** the Compress-FT reconstruction MSE barely changes (4.59e-7 to 5.25e-7) while input noise grows from σ = 5e-5 to 5e-1. DERIVED: the decoder output is nearly insensitive to its input. The paper reads this as denoising.
* **DR-24:** the appendix histograms (Gaussian-like, centred at 0, ±0.01, narrowing, labelled "Gradient Value") are incompatible with Kaiming-uniform A states (flat over ±0.0156) under the absolute-state reading.
* **DR-25:** Algorithm 1 defines `G_i^t = B_i A_i` (d x d product) but encodes `[A_i, B_i]`, and Table 1's shape is the factor stack. v1 named the same object `ΔW_i^t`.
* **DR-26:** the security section states that the downlink also carries encoded data (clients hold the encoder and decoder), while the communication analysis and the compression ratio count only the uplink.
* **DR-27:** the appendix concatenates 8 x 4096 blocks into 2048 x 4096, the transpose of Table 1's `[1, 4096, 2048]` (same elements). It states "Aᵀ and B are 8 x 4096", a dimension slip.
