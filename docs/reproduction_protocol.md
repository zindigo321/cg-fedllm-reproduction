# Reproduction protocol

## 1. Environment (verified Phase-2 stack)

```bash
conda create -n cgfedllm python=3.11 -y
conda activate cgfedllm
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements/base-win-cu130.txt
python -m pip install -e . --no-deps
# keep caches and outputs outside the repository (example used on the reference machine):
conda env config vars set HF_HOME='D:\cgfed-cache\hf' CGFED_CACHE='D:\cgfed-cache' CGFED_RUNS='D:\cgfed-runs' PYTHONUTF8=1
```

Full transitive freeze: `requirements/lock-win-py311-cu130.txt`. CI (Linux, CPU): `requirements/ci-cpu.txt`.

## 2. Commands

| Purpose | Command |
|---|---|
| Unit + integration tests (CPU) | `pytest -q -m "not gpu and not model"` |
| Rebuild/verify the committed partition manifest | `cgfed prepare-data --config configs/base/shepherd_dolly.yaml` |
| Tier-C end-to-end smoke (GPU) | `cgfed smoke --config configs/smoke/llama160m_smoke.yaml` |
| Evaluator validation | `cgfed evaluate --config configs/eval/qwen15_0p5b_validation.yaml` |
| MMLU cross-check vs lm-eval | `python scripts/crosscheck_mmlu_lmeval.py --ours <run>/mmlu_test_5shot.json --model-path <snapshot> --dtype float32 --batch-size 1 --out results/phase2/mmlu_lmeval_crosscheck.json` |
| GPU micro-benchmarks | `cgfed bench-gpu --config configs/feasible/gpu_microbench.yaml` |
| Federated / centralized run | `cgfed run-fl --config <cfg>` (`federated.pooled: true` for centralized) |
| TGAP collection / AE training | `cgfed collect-tgap --config <cfg>`; `cgfed train-ae --config <cfg> --snapshots <dir>` |

Overrides: `--set key.path=value` (repeatable). A run directory is `<output_root>/<run.name>/<stage>`.

## 3. reference_eval_v1 (our stable protocol — not the paper's)

* **C-Eval** (`ceval/ceval-exam@617524a0…`): official answer-only prompt
  `以下是中国关于{中文科目}考试的单项选择题，请选出其中的正确答案。\n\n` + 5 dev shots
  (`{q}\nA. …\nB. …\nC. …\nD. …\n答案：{X}\n\n`) + the question ending in `答案：`; continuations `A`–`D`;
  per-subject accuracy; STEM / Social Science / Humanities / Other and Average are **unweighted means over
  subjects**; Hard = mean over the 8 C-Eval Hard subjects. Splits: `val` and the public `test`.
* **MMLU** (`cais/mmlu@c30699e8…`): lm-eval 0.4.x template, first 5 dev items, continuations ` A`–` D`;
  overall and category accuracy **question-weighted**; unweighted subject mean reported as secondary.
* **Scoring:** lm-eval `loglikelihood` semantics (`context_ids = enc(ctx)`, continuation = tail of
  `enc(ctx + cont)`), argmax with first-index tie-break; single-token fast path reads only the last
  position's logits; float32 log-softmax; left padding with explicit position IDs.
* **Context limit:** if `max_context` is set and exceeded, shots are dropped from the end (logged per
  question); this never triggered for Qwen1.5 (32k context).
* **Correctness oracles:** MMLU agreement with lm-eval within 0.5 pp (hard gate); C-Eval validated by
  structure checks, prompt fixtures and batching-invariance tests. Published model scores are sanity
  references only.

## 4. Gates and tolerances

| Gate | Criterion |
|---|---|
| Phi round trip | bitwise exact (`adapter_state`) |
| IdentityCodec FAF vs uncompressed FL | CPU: bitwise; GPU: bitwise or relative L2 <= 1e-6 |
| adapter_delta identity path | relative L2 <= 1e-6 (fp32 rounding of `G + (end - G)`) |
| Resume | identical final adapter hash |
| N = 1 FL vs local training | CPU bitwise; GPU bitwise or relative L2 <= 1e-6 |
| MMLU scorer vs lm-eval | |delta overall| <= 0.5 percentage points |
| Compression ratio | latent / input elements = 1/64 exactly |

## 5. Result labels

Every reported number carries one label: **PAPER-REPORTED**, **PHASE2-SMOKE**, **LOCAL-MICROBENCH**,
**DERIVED** or **UNKNOWN**. Smoke numbers are correctness evidence only and must never be compared with the
paper's tables as reproduction results.
