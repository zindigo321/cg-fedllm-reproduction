# Discrepancy register

## A. Inside the paper (found in Phase 1; unchanged by Phase 2)

| ID | Discrepancy | Handling |
|---|---|---|
| DR-01 | Table 2 Row 1 "Compress-FT-LLaMA" (26.8/26/26.8/26.5/26.6/26.9) differs from Row 7 "Compress-FT-LLaMA" (26.6/26.5/25.5/25.7/26.2/26.8), which matches the RQ2 text and v1; Row 1 equals the D-Compress row. | Row 7 / text values are the paper reference for Compress-FT-LLaMA. |
| DR-02 | Several "different" runs report identical numbers (U-Former-0 = Row 3 = noised sigma=5e-4; U-Former-1 = Upsampling-2 = ResNet-3 = epochs 13/19; identical matrix entropy for Cent/Compress/LoRA). | Not treated as independent targets. |
| DR-03 | Dolly partition: main text "100 Dirichlet(0.5) segments" vs appendix "10 clients, equal quantity" (Shepherd shard mode). | 100-client Dirichlet is primary; both modes implemented and oracle-tested. |
| DR-04 | "Three clients" is attributed to Qwen (main text), C-Eval-dev (appendix) and ChatGLM (v1). | 3 clients for C-Eval-dev and Qwen+MMLU configurations (Phase 3+). |
| DR-06 | AE shape assumes r = 8 but LLaMA + C-Eval uses r = 16 -> [1, 4096, 4096]. | Fully-convolutional AE accepts it; Phi tested at r = 16. |
| DR-07 | 1-D CNN: stated shapes give CR 3.125 %, not 3.21 %; tables and diagram disagree. | 1-D CNN deferred (R5). |
| DR-08 | ResNet table lists a 7th deconvolution row ("deconv1 4->32"); text says six. | Six (parameter/MAC counts then match Table 3). |
| DR-09 | Max/min value text "0.0018..0.0010" vs figure axis 0.018..0.010; caption "2x4x32=128" (= 256). | Figure values. |
| DR-10 | ||G||^2 = 14.29 for every AE; inconsistent with a PEFT-initialised raw A (~341 for 128 A's at LLaMA-7B scale). | Representation left configurable (R2). |
| DR-11 | SNR = summed signal energy / per-element MSE (inflated by the element count). | Both `snr_paper` and standard `snr_db` reported. |
| DR-12 | DP noise "before compression" vs mechanism Dec∘(Aggre∘Noise∘Clip)∘Enc; Poisson vs fixed-size sampling; delta/C missing. | DP deferred. |
| DR-13 | "Generalize-Compress-Qwen" described as fine-tuned on MMLU-600 and, elsewhere, on Dolly. | Deferred (R10). |
| DR-14 | "Cent" expanded as "contrastive learning" (ECAI/v3) vs "centralized learning (CL)". | Cent = centralized. |
| DR-15 | Aggregation written as a raw sum and W + eta B~A~ vs Shepherd's normalised FedAvg + adapter replacement. | `sample_weighted_mean` primary; `literal_sum` diagnostic (R3). |
| DR-16 | TGAP data from "training without FL" (v1) vs "pre-training of FedLLM" (ECAI/v3). | Both modes (R6). |
| DR-21 | "PyTorch 2.2.1 with CUDA 12.4" (no official 2.2.1 cu124 wheel). | Recorded as reported. |

## B. Discovered in Phase 2

| ID | Finding | Evidence | Consequence |
|---|---|---|---|
| P2-D1 | **Shepherd's partition depends on the pandas version.** `DataFrame.sort_values` uses numpy quicksort (unstable) in pandas < 3 but a stable sort in pandas >= 3, changing the within-category order, the held-out sample and every client's data. Under pandas 3 the Dirichlet branch raises `ValueError: array is read-only` (copy-on-write makes `Index.values` read-only). | Oracle runs: shard partitions differ between pandas 2.3.3 and 3.0.6; Dirichlet crashes on 3.0.6. | We reproduce the pandas < 3 semantics explicitly (`argsort(kind="quicksort")`) and test against the pandas-2.3.3 oracle; the manifests pin the result by hash. |
| P2-D2 | Shepherd's Dolly file is the **first** Dolly release: 15,015 records (HF current: 15,011), including 8 exact duplicate records. | Source file hash `52e0c44e...`. | Records are identified by position (`source_id`), never by content. |
| P2-D3 | The official C-Eval `subject_mapping.json` uses CRLF line endings; Git `eol=lf` normalisation would silently alter the "verbatim" copy. | blob hash before/after normalisation. | File marked `-text`; committed bytes = upstream SHA-256 `671018e9...`. |
| P2-D4 | Publication dates differ by source: Crossref `issued` 2025-10-21 vs publisher `citation_online_date` 2025-10-22 (Phase-1 report used the latter). | Crossref API; IOS Press page. | CITATION.cff uses the Crossref date; both recorded in `provenance.md`. |
| P2-D5 | On this Windows/WDDM machine, exceeding VRAM spills silently into shared system memory instead of raising OOM (> 10x slow-down). Padded evaluation batches materialise an O(B * L^2) attention mask. | Scorer profiling (46 s for one 7 x 2055 batch). | `eval.max_batch_attention` (B * L_max^2) budget; predictions unchanged (171/171 identical, max |delta logprob| 1.9e-5). |
