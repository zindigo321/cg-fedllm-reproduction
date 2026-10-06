# P5-A v2 launch preparation record

**Status: READY FOR DOCUMENTATION-ONLY COMMIT (external document audit, section 13). REAL PREFLIGHT: NOT AUTHORIZED. TRAINING: NOT AUTHORIZED.**

This document prepares the launch record that `docs/phase5_preregistration_v2.md` §16, the v2.1 addendum
`docs/phase5_preregistration_v2_1.md` and `docs/reproduction_protocol.md` §6 require before any real P5-A step. It
authorises nothing. After review it is meant to be committed alone, in a documentation-only commit, **before** the
first real preflight (section 7). Facts that exist only at execution time (the execution HEAD, the preflight record,
the authorisations) are written later into an external execution record outside the repository and outside every run
root (section 7.3). This file never contains the SHA of the commit that contains it.

Every real step needs its own dated, written human authorisation. No flag, file or command-line option replaces it.
`--launch-record` only cites a file by path and SHA-256; the software cannot verify an authorisation (v2 §13.1).

Labels: VERIFIED (checked in this repository on the stated date from Git objects, committed text or a static
repository-only check), DERIVED (computed from committed values), REPORTED (stated by an earlier record, not
re-checked here), PENDING (needs a later event or decision), NOT EXECUTED / UNKNOWN (does not exist yet). Nothing here
was derived from a real payload, a run-root file or a GPU.

## 1. Research question and decision criteria (frozen; v2 §1, §10, §11)

For each of R2, R3 and R4 separately: can the fixed ResNet-3 AutoEncoder with the exact 1/64 bottleneck meet the fit
standards on (a) one fixed training snapshot and (b) four fixed training snapshots, after dividing every input of the
representation's training population by one frozen scalar that puts all of them inside the decoder's Tanh range?

| Control | Required criteria (all inclusive, float64, literal thresholds, no epsilon) |
|---|---|
| `single_snapshot` (memorisation standard) | C1 every reconstruction element finite and every gated value defined and finite; C2 `RSE <= 0.01`; C3 `cos >= 0.99` |
| `fixed_subset4` (subset fit standard) | C1; C2 `RSE_i <= 0.50` and C3 `cos_i >= 0.90` for **every** snapshot; C4 pooled `RSE_S(AE) <= 0.8 * RSE_S(subset mean)`, evaluated as written also when the subset-mean error is 0 |

RSE is the relative **squared** error `sse / sig`; pooled values are ratios of summed float64 sums. C0 (the Tanh-range
ceiling meets the control's own gate) is checked before training. A control is `FIT PASS` iff every required
criterion holds, otherwise `FIT FAIL`; the seven `INCOMPLETE` / `NOT INFORMATIVE` statuses of v2 §11.1 are not gate
results.

| `single_snapshot` | `fixed_subset4` | Outcome |
|---|---|---|
| `FIT PASS` | `FIT PASS` | `CAPACITY FIT DEMONSTRATED` |
| `FIT FAIL` | any | `CAPACITY FIT NOT DEMONSTRATED` |
| any | `FIT FAIL` | `CAPACITY FIT NOT DEMONSTRATED` |
| every other combination | | `INCONCLUSIVE` |

R2 and R3 are primary; R4 is a prespecified secondary that is always scheduled. Representations are never aggregated.
Required qualifier: "under the P5-A v2 protocol (fixed ResNet-3, exact training-population max-abs scaling to 0.95,
Adam 2e-4, 3,000 iterations, final checkpoint, AE seed 1, training snapshots only)". No outcome establishes
compressibility, general capacity, generalisation, seed stability, a cause of the F6 failures, or anything about FAF.
`CAPACITY FIT DEMONSTRATED` authorises only **drafting** a separately reviewed scale-normalised re-screen
preregistration for that representation. It does not authorise executing a re-screen, P5-B, P5-C, P5-D, FAF or any
downstream experiment. The other outcomes authorise nothing (v2 §11.5).

## 2. Identities and fingerprints

### 2.1 Commits and protocol documents (VERIFIED from local Git objects, this round)

| Item | Value |
|---|---|
| Protocol v2 (frozen) | commit `dd4bf3f4348fd3b2b1b2c4ad23e7204fbef8369c` (parent `52a0dd4`); `docs/phase5_preregistration_v2.md` blob `e80d21aeda653ab6d1339d34f7b88118588855df`, SHA-256 `ed03ac67e1c608c5c4de17bd00abfcd057dff1f18ea5fd9a8efafda6e476c112` |
| Protocol addendum v2.1 | commit `33e51fd63c6cd0679e76e12380ae5d418158d698` (parent `dd4bf3f`); `docs/phase5_preregistration_v2_1.md` blob `28e05f9ee33ed13386d346fce6e47e9108c937d7`, SHA-256 `a445d3476948e3c558805b36f7b6cd8b8f30d34e45ea42093f1c45e742e5686d`; supersedes only the v2 text named in its A1 |
| Protocol identity of every record | the pair (v2, v2.1): `p5a.protocol_identity()` holds both commits, paths and blobs above (v2.1 A1) |
| v2 companion | `docs/phase5_v2_audit_response.md` blob `7164034fe7a652f204e7120d59f638f2bb831ffd`, SHA-256 `a2a40a509449b82f7cdea5121673bf168b9f24967776a6515005d1f67172a441` |
| v1 (historical, never executed) | commit `52a0dd4e7bb8521752ac80ff8f77072dc0e0761c`; `docs/phase5_preregistration.md` blob `00f2477d164b795c4f35f717a3017888baee9b4e` |
| Baseline `main` | `afc9e333b5b865b32d47f5634aa7db3ddbfa97ef` (local `main` and `origin/main`) |
| **Reviewed implementation commit** | **`603acfce3fcfbe6724f7900c78dc03e28071f487`** (parent `33e51fd…`, tree `f8768545856dd71f058914e5958480c7efc332a1`, committed 2026-10-05 20:10:39 -0400); 26 files, including the F10/F11 fixes and their six regression cases; pushed (the remote-tracking ref `origin/worktree-p5a-engineering-support` is `603acfc…`; not re-fetched in this round). Both protocol blobs are unchanged in it. It is never the protocol SHA |
| Implementation review record | `docs/phase5_implementation_review.md` at `603acfc`: blob `e3a2798496ea24403bc0b43f290f5a3aa75e6d79`, SHA-256 `ede7fdb98df40498557e9fad04062389998efb477741212eab8089c7f3f4b12e` |
| Git state at this revision | branch `worktree-p5a-engineering-support`, HEAD = `603acfc…` = its remote-tracking ref, index empty; the only untracked file is this document |
| Launch-preparation commit | PENDING: the documentation-only commit of this file (section 7.4). Its SHA is never written into this file |
| Execution HEAD | PENDING: resolved after that commit and recorded in the external execution record (section 7.3) |
| Cited launch-record identity | PENDING: path and SHA-256 of the external execution record, recorded by the code at each real step |

### 2.2 Source fingerprints of the reviewed implementation commit (VERIFIED, this round)

Git blob = `git rev-parse 603acfc:<path>`; SHA-256 = SHA-256 of that blob's bytes. The working-tree bytes equal these
blobs (`git hash-object --no-filters` and SHA-256 of the working-tree files; `.gitattributes` normalises text to LF).
The coverage is that of the previous draft. Two values changed since it, both by the post-audit fixes: `cli.py`
(F10, only `cmd_p5a`, 6 lines added) and `phase5/p5a_run.py` (F11, one hunk in `production_closure`, 8 lines added,
1 removed), verified by diffing the old and new blobs. The other ten values are unchanged.

| Path | Git blob | SHA-256 |
|---|---|---|
| `configs/phase5/p5a_v2_seed1.yaml` | `f6385eb0c3a5ac710e102cd94a808f06b4096e26` | `dde9b89a479363a1bbe879b967f89d291345cefdf480c04767c04e042f5513d6` |
| `src/cg_fedllm/cli.py` | `cf2a1511c0159cb28c80ed40f1e24a1a59d09706` | `20cac35ce5d6e992ebbdd4a9bc1e3c35d0eb57c4ca703b110470b004c7116603` |
| `src/cg_fedllm/compression/normalization.py` | `fe916cb3dd818aaac8b2e7eb8951858104c1adc5` | `9d14fac2aef463ffe53e1966bc2464a3d924bf339608c3e5a7b925fd1c6d1f7a` |
| `src/cg_fedllm/config.py` | `c50f12b4904f16ac00d804bc53315509e5278855` | `5e8a1a711875479141e3360b72515ecde1054c1e0f85574d177cc9fbe6145e14` |
| `src/cg_fedllm/phase5/__init__.py` | `ba4bd6687fc0ec0f1a39faf15f32b25049a5d5cb` | `a872dfe94d1a372661b974cb8a6fc00c4d26c82f7eb9e888c1886fc9e36bff18` |
| `src/cg_fedllm/phase5/p5a.py` | `b6b0477beef4a70d1aa6766624765134a9bfe382` | `bd709b2e80037e03cdd03ce048a5222f18c5fd63043a5ea997ce70ce5fd21265` |
| `src/cg_fedllm/phase5/p5a_artifacts.py` | `1184b0ea617159411e42b7d80436beaa4032160b` | `93634c152c26bcbb77a47ea1cf9e13d8a127b555afc80b387dc31514c022773b` |
| `src/cg_fedllm/phase5/p5a_inputs.py` | `556d990c53b149a85cdc47724364006fc8ec0c96` | `ca57204ffb43873425649f409b38fc221e0a63c787d6e3db4be88f8dc72c305a` |
| `src/cg_fedllm/phase5/p5a_metrics.py` | `e3ef0523c9d3a0fa1e9ab751c23fe3527608fe70` | `1cc8d743eb479a08ba04e55206b69bcd72c99af63a690c8a54a5704b3f7665da` |
| `src/cg_fedllm/phase5/p5a_preflight.py` | `49c265766d534e160a30f1822dca7336c9cfd9ec` | `b8c0140452445b19a0a46fa5f76eff71585b11a624bd88cfa20b141507dcba44` |
| `src/cg_fedllm/phase5/p5a_run.py` | `412547f0c0dcdbc28c1ee890617fa6eef8b51473` | `5aa4f3be27e18bb92083287452b2df5ad9f900fba357aabb43e34f7e5689ce53` |
| `src/cg_fedllm/phase5/p5a_state.py` | `339f53edca59d97339489184ed6a3a990acac696` | `84fa30075970b19417e7892d67602e2c9e81ce7c0eeac3dfc9c084dbe52036d7` |

Added in this revision, unchanged since `afc9e33`: the four ancestor YAML files (with `p5a_v2_seed1.yaml` and the
dataclass defaults of `config.py`, the only inputs of the resolved configuration hash) and three shared helpers on the
launch path (`configure_determinism`, `git_info`, which records `execution_commit`, and the canonical hash functions).

| Path | Git blob | SHA-256 |
|---|---|---|
| `configs/phase3/tierb_seed1.yaml` | `4922d49c9ef7e9f7112fa15d044af57a253154db` | `444da8392e31da47981380cb689680c8246f0a15db5baf3fcb1bcee0613a4bf5` |
| `configs/feasible/tierB_qwen15_1p8b_candidate.yaml` | `0f746d275d7efb8e918ac76f703f783897be32d9` | `68c1ec26a51af3e0514b3f8eeef60cb1ed24cc2c78af18f98186a288f3b35449` |
| `configs/base/shepherd_dolly.yaml` | `b031c86d9d718a9abb27ea07728bf1a5f30c4b7e` | `9d44ee74698f7c86a4e10be0821e6878437e1a349ea48387b039d921388d6711` |
| `configs/base/defaults.yaml` | `00a0cc2d36a3e3c9b0217ea1f48f75154160bf73` | `b303b7971f5fa06eba64f953d220a32534f72d920995d34561d249460284051e` |
| `src/cg_fedllm/utils/seeding.py` | `f1f1c226eca777f800d43251c045044592808735` | `fbe46723ee76839c1bcce6d2d7b15dae0f5674f0d7931e7437b259764d4d6678` |
| `src/cg_fedllm/utils/provenance.py` | `5b01c18e5bac279f42b8a6d17b683d97c41b38f8` | `ca9798bfd7473857f93ce918669dd80765ba690ad80ec1577cb156f33d4c35db` |
| `src/cg_fedllm/utils/hashing.py` | `0e60c0b3891bf1a1bd90c8c1893416bb007aeb2a` | `abfd214c6d13c9bafe3d928808a5eae930b38c20d7c0e3e93f8ef07d43f10fd7` |

Tests and documentation are not execution inputs and are not fingerprinted, apart from the one test cited in 2.3.
The execution HEAD identifies every byte; these tables are a cross-check, not a substitute for it.

### 2.3 Three different configuration hashes (kept apart)

| Hash | Definition | Value | Where it is used |
|---|---|---|---|
| Source-byte SHA-256 of the YAML file | SHA-256 of the bytes of `configs/phase5/p5a_v2_seed1.yaml` only | `dde9b89a…f5513d6` (table 2.2) | file fingerprint; **not** a configuration hash |
| Resolved configuration hash `config.sha256()` | `canonical_json_sha256(cfg.to_dict())` (sorted keys, compact separators, UTF-8, trailing newline) of the dataclass built by `load_config` from the YAML inheritance chain of 2.2 plus dataclass defaults, with no override; `run.output_root` stays the unexpanded string `${CGFED_RUNS:-runs}`, so the value does not depend on any environment variable | `e5ac614c5f176f39b40ddd96cfe311f50c8209151578a1535794cd55d87bb122` | `config_sha256` of the preflight record and of every invocation; compared preflight -> training (`validate_against_preflight`) and invocation 1 -> invocations 2-3 and closure-only (`check_same_run`) |
| Identity configuration hash `identity_config_sha256(cfg)` | the same canonical hash after setting `federated.stop_after_round` to `null` (`pipeline.py`) | `e5ac614c5f176f39b40ddd96cfe311f50c8209151578a1535794cd55d87bb122` | written next to `config_sha256` in each control's `config.sha256.json`; not compared by P5-A |

The two configuration hashes coincide here only because the resolved `federated.stop_after_round` is already `null`;
they are different definitions and are not interchangeable in general.

Evidence for the resolved value: the committed test
`tests/unit/test_p5a_preflight.py::test_production_preflight_route_on_an_empty_synthetic_root` (blob
`a31d221968fefc258d94058726db0c5eed4836ec` at `603acfc`) asserts this literal on the real CLI route, and implementation
review §13.5 records it. Recomputed in this round (2026-10-06) with `D:\conda_envs\cgfedllm\python.exe`,
`CUDA_VISIBLE_DEVICES=-1`, `HF_HUB_OFFLINE=1`, imports from this worktree's `src`: `load_config(...)`,
`p5a.check_config(cfg)` (passes) and `cfg.sha256()` give the value above; the identity hash was recomputed by applying
the `pipeline.py` definition directly, without importing `pipeline`, `peft` or a model module. Inputs: the five
repository YAML files of 2.2 (the configuration and its four ancestors) and the `config.py` defaults only. CUDA was
not initialised. The value is a property of the committed bytes. The
code recomputes it at every real step, so it is confirmed again by each record.

## 3. Configuration

| Item | Value |
|---|---|
| Path | `configs/phase5/p5a_v2_seed1.yaml`, loaded relative to the current directory, so every command runs from the worktree root. Inheritance: `p5a_v2_seed1.yaml` -> `phase3/tierb_seed1.yaml` -> `feasible/tierB_qwen15_1p8b_candidate.yaml` -> `base/shepherd_dolly.yaml` -> `base/defaults.yaml` |
| Hashes | section 2.3 |
| Overrides | **none permitted**: both production routes refuse every `--set`, and `p5a.check_config` refuses any locked field that differs (`AE_LOCK`, `RUN_LOCK`, `LORA_LOCK`) |
| Run | `name phase5_p5a_v2_seed1`, `device cuda`, `deterministic true`, `num_threads null`, `result_label PHASE5-DIAGNOSTIC`, `allocator_cap_margin_mb 256`, `seed 1` |
| AE | `resnet3`, layout `layer_major_qkvo_AtB`, `batch_size 4` (applied as `min(4, k)`), `iterations 3000`, `learning_rate 2e-4`, betas `(0.9, 0.999)`, `epsilon 1e-8`, `weight_decay 0`, `init_seed 1`, `eval_every 50` (curve only), `checkpoint_policy final`, `normalization global_exact_maxabs_train` |
| LoRA geometry | `r 8`, `alpha 16` (s = 2); Qwen1.5-1.8B q/k/v/o, 24 layers, hidden 2048; Phi `[1, 2048, 1536]` fp32; latent `[64, 32, 24]` |
| Run root | `D:\cgfed-runs\phase5_p5a_v2_seed1\`, given by `--runs-root D:\cgfed-runs` (absolute, existing; `run.output_root` is not used by P5-A). It must be absent before the first preflight. `D:\cgfed-runs\phase5_p5a_seed1\` (v1) must never exist; every P5-A entry point refuses to run if it does |

## 4. Environment

### 4.1 Machine and software

| Item | Value | Source |
|---|---|---|
| Interpreter | `D:\conda_envs\cgfedllm\python.exe`, Python 3.11.16 (`sys.executable` prints exactly this path) | VERIFIED 2026-10-06 (no torch import) |
| OS | Windows-10-10.0.26100-SP0 (native Windows); Windows PowerShell 5.1.26100 | VERIFIED 2026-10-06 |
| Packages | torch 2.14.0+cu130, numpy 2.4.6, safetensors 0.8.0, peft 0.21.1, transformers 5.17.0 (installed distribution metadata); full freeze `requirements/lock-win-py311-cu130.txt` (blob `271b55e62384a2b5331e723bc080c8cf49aeb410` at `603acfc`; not re-diffed against the installation) | VERIFIED 2026-10-06 (metadata only) |
| Package location | the editable install `__editable__.cg_fedllm_repro-0.2.0.dev0.pth` names this worktree's `src`. The commands do not rely on it: they set `PYTHONPATH` to this worktree's `src` and stop unless `cg_fedllm` resolves there. This matters twice: the code must be the reviewed bytes, and `git_info()` finds the repository from the imported source file, so `execution_commit` is the HEAD of the checkout whose code is imported | VERIFIED 2026-10-06 (`.pth` content; `find_spec` origin with the command environment) |
| `CGFED_RUNS` | `D:\cgfed-runs`, passed explicitly as `--runs-root` | REPORTED (`reproduction_protocol.md` §1); not opened in this round |
| GPU, driver, CUDA | NVIDIA GeForce RTX 4060 Laptop GPU, compute 8.9, 8,585,216,000 bytes; driver 595.97; torch CUDA 13.0, cuDNN 92400 | REPORTED (F7 diagnostic of 2026-10-05, implementation review §13.3). The training invocation records `nvidia-smi` output and the CUDA facts in `run_metadata.json`, and its allocator cap (free VRAM - 256 MiB) as `vram_guard`: PENDING |

### 4.2 Pinned process environment (v2.1 A5 P1)

Both routes call `configure_determinism(cfg.run.deterministic, cfg.run.num_threads)` from the locked configuration,
that is `configure_determinism(True, None)`, before building inputs. The preflight records `cpu_environment()`; every
training invocation recomputes it before it rebuilds the inputs and treats any difference as stale preflight evidence
(`INCOMPLETE (PROVENANCE)` for that representation, before any start). The blocks of section 8 hold every field fixed:

| `cpu_environment()` field | Required value | How it is held fixed |
|---|---|---|
| `interpreter` | `D:\conda_envs\cgfedllm\python.exe` | the same literal path in every block |
| `python`, `torch`, `torch_git_version`, `torch_cuda_build`, `numpy`, `safetensors` | 3.11.16; 2.14.0+cu130; the build's git version; 13.0; 2.4.6; 0.8.0 | the same environment; no install, update or removal between the first preflight and closure |
| `deterministic_algorithms`, `deterministic_warn_only`, `cudnn_deterministic`, `cudnn_benchmark`, `cuda_matmul_allow_tf32`, `cudnn_allow_tf32` | `true`, `false`, `true`, `false`, `false`, `false` | set by `configure_determinism(True, None)`; strict determinism, never warn-only, TF32 off |
| `float32_matmul_precision`, `default_dtype` | `highest`, `torch.float32` | torch defaults; nothing under `src/` sets them (static search, this round) |
| `torch_num_threads`, `torch_num_interop_threads` | the torch default of this machine, because `run.num_threads: null` | no thread count is set and the thread variables below are absent. 24 / 24 was measured on 2026-10-05 (REPORTED, review §13.2): a measurement, not a requirement or a promise. The requirement is equality between the preflight and each training invocation, on the same machine |
| `env.CUBLAS_WORKSPACE_CONFIG` | `:4096:8` | set explicitly in every block (`configure_determinism` would only `setdefault` the same value) |
| `env.OMP_NUM_THREADS`, `env.MKL_NUM_THREADS`, `env.OPENBLAS_NUM_THREADS`, `env.NUMEXPR_NUM_THREADS`, `env.VECLIB_MAXIMUM_THREADS`, `env.MKL_DYNAMIC`, `env.OMP_DYNAMIC`, `env.MKL_CBWR`, `env.KMP_AFFINITY`, `env.KMP_BLOCKTIME` | absent (`null`) | removed from the process environment in every block, so a user or system value is not inherited |

Not part of the fingerprint, also set by every block: `HF_HUB_OFFLINE=1`; `PYTHONPATH=<worktree>\src`;
`CUDA_VISIBLE_DEVICES=-1` for the preflight and the closure-only step, and **absent** for training. A Python process
cannot be repaired by changing its parent shell afterwards, so the GPU step always starts a new process from a shell
whose `CUDA_VISIBLE_DEVICES` has been removed. The test fixture `configure_determinism(True, 2)`
(`tests/conftest.py`) is never used for a launch.

## 5. Frozen inputs (committed text and constants; real bytes PENDING)

Cross-checked in this round against the committed v2 text and the code constants of `603acfc` only: the R4
allowlist below equals `p5a_inputs.R4_GRADIENT_FILES` and v2 §3 entry for entry (14 / 14 / 14); the index hashes
equal `p5a_inputs.INDEX_SHA256`; `FROZEN_MAX`, `FROZEN_SCALE` and `F6_TRAIN_RMS` equal the tables below, and
`m_r / 0.95` in float64 equals `FROZEN_SCALE` for all three; `p5a.control_positions` gives the positions below (the
(t, client) pairs are copied from v2 §6 and were not re-derived in this round).
**Real-byte identity: PENDING until the authorised preflight.** No real file was opened or hashed.

| Root (under `D:\cgfed-runs\`) | `index.jsonl` SHA-256 | Population | Used for |
|---|---|---|---|
| `phase3_tierb_qwen15_1p8b_seed1/tgap_federated_state` | `b6834e476f21f3d6b07f6489b5113576e566ab3987a2f4593405664476d07240` | time 0-15, 80 records, `load_states(verify=True)` | R2, R3 |
| `phase4_gradient_forensics_seed1/gradient_forensics` | `560cb0f8153fc0a44e2910ae119ebdccdf228271f963969ff4352401833af9b2` | time 0, 5 records, the 14 files below only | R4 |

R4 allowlist (complete), under `gradients/`:

| File | Bytes | SHA-256 |
|---|---|---|
| `t0000_c0002/meta.json` | 280 | `796d4c1c3b81040fdd2150d3556093b75d1f1ca4eca42e77fe0454960ae3b0ee` |
| `t0000_c0002/step000_grad.safetensors` | 12,612,736 | `79a309a868ed8428fe7ab2040a67825abf37aff353fcc2bca1904deef0f89f99` |
| `t0000_c0002/step001_grad.safetensors` | 12,612,736 | `0cfdfc3bc9a05d9105ac04d08b9fd9a8ead3852d3c13c3575edb2e53bd1a4370` |
| `t0000_c0026/meta.json` | 393 | `2f475d8a86f8c337e111414c0d68e6a0d9718271c2d631dfe76860b03ba94e4e` |
| `t0000_c0026/step000_grad.safetensors` | 12,612,736 | `4f3da4a516c76f629f5e7e6b417eb95de97bb638ece738689d45431fab2b5892` |
| `t0000_c0026/step001_grad.safetensors` | 12,612,736 | `2b2fccb15830b444dae70f47c0c580f9f3eb9140ec925adaa45d4ee9c9fa28b1` |
| `t0000_c0026/step002_grad.safetensors` | 12,612,736 | `d8cee48f6a0278728a63325a1687bb0541f4683265fd53c6a081970302e2e9b6` |
| `t0000_c0055/meta.json` | 170 | `6da7293551ad7eade1fd8d7846d6f4298ca08164ea9bda34c20369bc125b1884` |
| `t0000_c0055/step000_grad.safetensors` | 12,612,736 | `a271c7d09bfe38a2b6347088ba95dfb189443b05912f8e3cc095e1ab7c5cb808` |
| `t0000_c0075/meta.json` | 168 | `43eacf9cd7cbb5b1dfb7fc5af6c031ca818de6a475895c1213edbe1bbdcc9ac0` |
| `t0000_c0075/step000_grad.safetensors` | 12,612,736 | `d9f5a9f12f8bb181d5fab1118ce2c2ee619b80678165b8751eee10c6eaeab5d9` |
| `t0000_c0086/meta.json` | 282 | `5bd53055f71e2ed906f2b1f3cb3ceccdffb840fb00864f7663df520fce062fa7` |
| `t0000_c0086/step000_grad.safetensors` | 12,612,736 | `6167d489cf105cf77e71dab957a13ac313eea68c79d8724598a77b8b7d8998a7` |
| `t0000_c0086/step001_grad.safetensors` | 12,612,736 | `236d29f16d13369cd2e9792ba469ce716c3aadce1b48748134fac1108fd1001e` |

Step counts 2 / 3 / 1 / 1 / 2. No `*_grad_clipped`, `*_delta`, time-index-1 or R4 start/end state file is opened.

**Frozen scales** (DERIVED from committed F6 `input_stats` at evidence commit
`d3c4eb3d631e0280533f35eb652fb24f68870079`; F6 runtime provenance `e8100e9dd19ed4df98777646d6907d718790014e`,
`dirty: true`):

| R | `m_r` (exact max \|x\|) | `s_r = m_r / 0.95` (float64) | F6 `train_rms` (secondary check) |
|---|---|---|---|
| R2 | 0.032825905829668045 | 0.034553585083861103 | 0.004050884395837784 |
| R3 | 0.014964050613343716 | 0.015751632224572334 | 0.0020410344004631042 |
| R4 | 0.0643031895160675 | 0.06768756791165001 | 0.0010228796163573861 |

The R2 literal and its shortest round-trip form `0.0345535850838611` (Python `repr`) parse to the same float64 (v2
§5.3); the code uses the long literal. Tolerances:
`|m_rec - m_r| <= 1e-6 * m_r`; RMS likewise; every normalised snapshot max `<= 0.95 (1 + 2e-6)`, the population max
`>= 0.95 (1 - 2e-6)`. A mismatch stops that representation; the frozen value is never replaced.

**Selected records** (DERIVED from committed metadata, v2 §6; the preflight re-derives them):

| R | `single_snapshot` (position -> (t, client)) | `fixed_subset4` |
|---|---|---|
| R2, R3 | 39 -> (7, 91) | 0, 26, 53, 79 -> (0, 2), (5, 32), (10, 43), (15, 84) |
| R4 | 2 -> (0, 55) | 0, 1, 3, 4 -> (0, 2), (0, 26), (0, 75), (0, 86) |

Payload identities, Phi SHA-256 and recomputed statistics: **NOT EXECUTED / UNKNOWN** until the preflight writes them.

## 6. Plan, budgets and evidence (frozen; v2 §7, §12, §13; v2.1 A3, A4)

* Fixed order: R2 `single_snapshot`, R2 `fixed_subset4`, R3 `single_snapshot`, R3 `fixed_subset4`, R4
  `single_snapshot`, R4 `fixed_subset4`. No outcome skips, reorders or changes another control; only the frozen §7
  prerequisite rules prevent a start. The iteration-3,000 (final) checkpoint is the only one evaluated and gated.
* Time: at most 6 trainings; 1,800 s per training (from before `ae.to(device)` to after the final evaluation, checked
  before every iteration, before evaluation and at the end); 10,800 s aggregate charged; a start needs
  `trainings_started < 6` and `charged + 1,800 <= 10,800`. An unobserved process death is charged 1,800 s, labelled
  `unknown_process_death`, and is never reported as a measured time.
* No retraining: a control with a finalized start record is never trained again; recovery publishes evidence only.
  At most 3 preflights (none after `ledger\invocation_1.json` exists) and at most 3 training invocations; a
  closure-only step completes a run whose invocation 3 died before `ledger\summary_3.json`.
* Bytes: one run-wide ceiling of 20,000,000 decimal bytes over the exact sizes of all regular files under the run
  root, temporaries included; the per-artifact caps of v2 §13.2; the reservation check `F + Σ caps(Q) <= 20,000,000`
  before every publication and training start. 18,990,000 bytes is the **planning estimate** (every cap at its maximum
  count, no temporary). It is not measured usage, not enforced, and the 1,010,000-byte difference is not a reserve.
  Retained temporaries are counted at their exact size for the life of the run and are never deleted (v2.1 A3). The
  first record of the invocation-1 slot is publishable iff `F <= 14,510,000` (v2.1 A3). Measured bytes: NOT EXECUTED.
  REPORTED synthetic size of one preflight record with production-length metadata: 141,283 bytes (cap 300,000).
* Status precedence for a started control (v2.1 A4.2): a training-stop status (S1) is kept; a completed training with
  every file finalized is `FIT PASS` / `FIT FAIL` (S2); a completed training with any file not finalized, for any
  cause, is `INCOMPLETE (PUBLICATION)` (S3), with the byte cause in `resource_stop_cause` (`per_file_cap`,
  `aggregate_byte_cap`, `evidence_reservation`). A byte refusal never makes a started control
  `INCOMPLETE (RESOURCE STOP)`. A reported-only gate result is listed beside the outcome and is never a gate result.

Run root and artifact paths (exact; under `D:\cgfed-runs\phase5_p5a_v2_seed1\`):

| Path | Content |
|---|---|
| `preflight\preflight_<k>.json`, k = 1-3 | preflight records |
| `ledger\invocation_<k>.json`, `ledger\summary_<k>.json`, k = 1-3 | invocation records and summaries |
| `ledger\start_<R>_<c>.json`, `ledger\end_<R>_<c>.json`, `ledger\failure_<R>_<c>.json` | per-control ledger (R in R2-R4, c in `single_snapshot`, `fixed_subset4`) |
| `R2_balanced_effective_state_<c>\`, `R3_balanced_effective_delta_r8_<c>\`, `R4_mean_step_gradient_<c>\` | the six control directories |
| `<control dir>\config.resolved.yaml`, `config.sha256.json`, `run_metadata.json` | provenance set (one 150,000-byte cap together) |
| `<control dir>\p5a_record.json` | control record |
| `<control dir>\autoencoder_p5a_final.safetensors` | final checkpoint (`FIT PASS` / `FIT FAIL` only) |
| `<dir>\.<name>.<12 hex>.p5a-tmp` | retained temporaries: counted, never presented, never deleted by a later invocation or operator. A publisher removes its own temporary after a successful link or its own failed write |

Committed evidence later, in a separate reviewed change after closure: `results/phase5/p5a_v2/` (every control record,
every preflight record, every summary and the ledger JSON files), a findings document, deviation rows, and a
byte-identical copy of every cited execution-record version (section 7.3). Negative, inconclusive, incomplete and
publication-failure results are kept with the same weight as passes.

## 7. Execution identity and launch-record sequencing

### 7.1 Three different identities

| Identity | Value | Role |
|---|---|---|
| Reviewed implementation commit | `603acfce3fcfbe6724f7900c78dc03e28071f487` | the reviewed code; cited by this document |
| Launch-preparation commit | PENDING: the documentation-only commit that adds this file, with parent `603acfc` | freezes this document before any real step; it does not exist yet |
| Execution HEAD | PENDING: `git rev-parse HEAD` of this worktree at the first preflight, expected to equal the launch-preparation commit | what the code records as `execution_commit` in every preflight record, invocation record, control record and summary |

The execution HEAD is a full SHA of its own. It is not interchangeable with `603acfc`, even though the code bytes are
identical: the code compares `execution_commit` as a string.

### 7.2 What the code compares (VERIFIED by reading the source at `603acfc`)

| Check | Code | Compared | Not compared |
|---|---|---|---|
| preflight -> training | `p5a_preflight.validate_against_preflight`, per representation, on the most recent preflight record only | schema; protocol pair; `execution_commit`; `config_sha256`; `cpu_environment`; that representation's `pass`; the full evidence (identities, Phi SHA-256, recomputed statistics, selection) | `launch_record`, `config_path`, `command`, `execution_git.dirty` |
| invocation 1 -> invocations 2, 3 and closure-only | `p5a_run.check_same_run` | protocol pair; `config_sha256`; `execution_commit` and `launch_record` (path as given, `Path(...).as_posix()`, and SHA-256): same commit requires the same launch record; a different commit requires a different launch record (the v2 §13.1 implementation-change rule) | `cpu_environment` (training invocations re-check it against the preflight; closure-only does not use it), `execution_git.dirty` |

Consequences:
* A Git commit, merge, pull, rebase, reset or checkout in this worktree between the preflight and training makes the
  preflight stale: every representation becomes `INCOMPLETE (PROVENANCE)` before any start, the invocation still
  counts, and no new preflight is permitted once `ledger\invocation_1.json` exists. Preflights are limited to 3, so
  rerunning one to repair a document-driven HEAD change is not a plan. The launch-preparation commit is therefore made
  **before** the first preflight, and nothing is committed in this worktree until the run is closed.
* The execution record may change between the preflight and training, because that comparison does not include
  `launch_record`.
* After training invocation 1 has cited the execution record, its path string and its bytes must stay identical for
  invocations 2 and 3 and for the closure-only step; otherwise `check_same_run` refuses them before they publish
  anything.
* A reviewed implementation change after invocation 1 is not a workaround. It needs a new reviewed commit and a new
  cited record (v2 §12, §13.1), and its invocations would find the most recent preflight stale for every
  representation still to train, because no new preflight is permitted (v2 §5.4.1).
* The code records the clean-tree state (`execution_git.dirty`, `dirty_files`) but does not compare it, and
  `git_info()` records no commit if `git` is not on `PATH`. The blocks of section 8 therefore check both before every
  step, and each record's `execution_commit` is reviewed afterwards.

### 7.3 Sequence (no Git commit between real steps)

| # | Step | Changes HEAD? | Status |
|---|---|---|---|
| S0 | External review of this document | no | COMPLETED 2026-10-06 (section 13); ready for S1 |
| S1 | Human-approved documentation-only commit of this one file in this worktree, on `worktree-p5a-engineering-support`, parent `603acfc`. Pushing it afterwards does not move HEAD | yes, once | PENDING |
| S2 | Resolve the execution HEAD: `git rev-parse HEAD`; confirm a clean tree and that `git diff --name-only 603acfc HEAD` lists only this file | no | PENDING |
| S3 | Create the external execution record, version A, at the fixed path below, with the S2 SHA, then review it | no | PENDING |
| S4 | Step A, the real CPU preflight, after its written authorisation (section 8.2) | no | NOT AUTHORIZED |
| S5 | Review the preflight record; update the execution record to version B: the preflight citation and the written authorisation of training invocation 1 | no | PENDING |
| S6 | Step B, training invocation 1, citing version B. From here on the record's path and bytes are frozen | no | NOT AUTHORIZED |
| S7 | Any invocation 2 or 3, or the closure-only step: each after its own written authorisation, recorded in a separate file, never by editing the cited record | no | NOT AUTHORIZED |
| S8 | After closure: the evidence commit (separate reviewed change) | yes (no invocation follows closure) | PENDING |

**External execution record** (proposed path, created only at S3, never in this round):
`D:\p5a_v2_execution\p5a_v2_execution_record.md`. It is outside this repository, outside the main checkout and
outside `D:\cgfed-runs\`. Every command uses exactly this literal path. Later authorisations (S7) go into
`D:\p5a_v2_execution\p5a_v2_later_authorisations.md`, which is never passed as `--launch-record`.

At S3, obtain the repository-only facts (S2 HEAD, branch, clean tree, this document's blob and SHA-256,
source location and planned environment) and the OS free-RAM reading independently, before completing version A.
Defining block 8.1 does not run its checks. Do not call `Assert-P5AState` to construct its own input record: that
function requires an already completed execution-record file. No P5-A CLI or run-root inspection is needed to
assemble these facts. The first real step performs the authorised run-root checks afterwards.

Version A contains: this document's path, its Git blob at the execution HEAD
(`git rev-parse <execution HEAD>:docs/phase5_p5a_v2_launch_draft.md`) and SHA-256; the reviewed implementation commit,
the protocol pair and the execution HEAD; those repository-only facts and the pre-step free RAM; and the dated
written authorisation naming the preflight number it covers. Complete and review version A before step A.
The output printed by `Assert-P5AState`, including the cited record's SHA-256, goes into a **separate console log**
after the record has been completed; it is not copied back into version A. The free-RAM reading printed by
step A is retained in that log and may be cited by version B. Never edit the cited record while a step is running.
Version B adds: the path and SHA-256 of every preflight
record, the reviewer's statement that the most recent one was read in full, its `pass` flag per representation, its
`execution_commit`, `config_sha256`, `cuda_initialized`, `trainings_started` and a full copy of its `cpu_environment`;
and the dated written authorisation of training invocation 1, citing that preflight record by path and SHA-256. Before
invocation 1 the record may receive further versions (for example after a repeated preflight); after invocation 1 it
receives none. Before each edit, copy the current version unchanged to `D:\p5a_v2_execution\versions\` with its SHA-256
in the file name, so that every cited version survives for the evidence commit. A file cannot contain its own SHA-256:
the code records the cited SHA-256 in each record, and each block prints it.

### 7.4 Proposed launch-preparation commit (not made)

One file, `docs/phase5_p5a_v2_launch_draft.md`, staged by explicit path, in this worktree on
`worktree-p5a-engineering-support`, after explicit human approval. Proposed subject:
`docs(phase5): prepare the P5-A v2.1 launch record`. No code, configuration, test, protocol or review file changes.

## 8. Commands (future instructions; not run; each needs its own written authorisation)

Derived from `cli.py` (`build_parser`, `cmd_p5a_preflight`, `cmd_p5a`) and the production routes
`p5a_preflight.production_preflight`, `p5a_run.production_main` and `p5a_run.production_closure` at `603acfc`. The
only accepted P5-A flags are `--config`, `--runs-root`, `--launch-record` and, for `p5a`, `--closure-only`. `--set`
exists in the parser but both routes refuse any override; `--stage` is accepted by the parser and ignored by both
routes; neither is used. No other flag exists. `python -m cg_fedllm.cli` is the same entry point as the `cgfed`
console script; the explicit interpreter and `PYTHONPATH` make the environment and the imported source certain.

Use a Windows PowerShell 5.1 console (`powershell.exe`); the blocks were parsed and dry-run in that console host only,
not in the ISE or PowerShell 7. The native P5-A call runs with `$ErrorActionPreference = 'Continue'`, so a stderr line
can never abort it, and its exit code is checked explicitly afterwards.
For each step, open a **new** window, paste block 8.1, edit the one required value, then paste exactly one step block.
Each step block is one `& { ... }` script block: any failed check throws and nothing after it runs, so a failure can
never fall through to a P5-A command. Every native command is followed by an explicit `$LASTEXITCODE` check. Do not
wrap a step in a timeout, a retry or a loop, and do not kill the Python process: Ctrl+C during a training is an
operator interrupt (`INCOMPLETE (INTERRUPTED)`, the invocation stops), and a killed process is an unobserved process
death (charged 1,800 s). The per-training limit is enforced by the code. Copy the full console text of each step into
`D:\p5a_v2_execution\` as a log; it is never passed as `--launch-record`.

### 8.1 Common definitions (no authorisation needed; runs no P5-A command and reads no run-root file)

`ExecutionHead` must be replaced by the full SHA recorded at S2; the check refuses the placeholder.

```powershell
# P5-A v2 launch, common definitions (section 8.1). Defines values and two check functions only.
$P5A = @{
    Python         = 'D:\conda_envs\cgfedllm\python.exe'
    Worktree       = 'D:\projects-cd\cg-fedllm-reproduction\.claude\worktrees\p5a-engineering-support'
    Branch         = 'worktree-p5a-engineering-support'
    Implementation = '603acfce3fcfbe6724f7900c78dc03e28071f487'
    LaunchDoc      = 'docs/phase5_p5a_v2_launch_draft.md'
    Config         = 'configs/phase5/p5a_v2_seed1.yaml'
    RunsRoot       = 'D:\cgfed-runs'
    RunRoot        = 'D:\cgfed-runs\phase5_p5a_v2_seed1'
    V1Root         = 'D:\cgfed-runs\phase5_p5a_seed1'
    LaunchRecord   = 'D:\p5a_v2_execution\p5a_v2_execution_record.md'
    ExecutionHead  = 'SET-FROM-EXECUTION-RECORD'    # replace with the full 40-hex execution HEAD of step S2
}
$P5A_ThreadVars = @('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS', 'MKL_DYNAMIC', 'OMP_DYNAMIC', 'MKL_CBWR', 'KMP_AFFINITY', 'KMP_BLOCKTIME')

function Assert-P5AState {
    # Checks the Git state, pins the process environment, returns the SHA-256 of the execution record.
    param([Parameter(Mandatory = $true)][ValidateSet('Hidden', 'Visible')][string]$Cuda)
    $ErrorActionPreference = 'Stop'
    if ($P5A.ExecutionHead -cnotmatch '^[0-9a-f]{40}$') { throw 'set $P5A.ExecutionHead to the S2 execution HEAD' }
    Set-Location -LiteralPath $P5A.Worktree
    $null = Get-Command git -CommandType Application    # without git on PATH no execution_commit is recorded
    $branch = & git rev-parse --abbrev-ref HEAD
    if ($LASTEXITCODE -ne 0 -or $branch -cne $P5A.Branch) { throw "unexpected branch: $branch" }
    $head = & git rev-parse HEAD
    if ($LASTEXITCODE -ne 0 -or $head -cne $P5A.ExecutionHead) { throw "HEAD $head is not the execution HEAD" }
    $dirty = @(& git status --porcelain=v1 --untracked-files=all)
    if ($LASTEXITCODE -ne 0 -or $dirty.Count -ne 0) { throw "the worktree is not clean: $($dirty -join '; ')" }
    & git merge-base --is-ancestor $P5A.Implementation HEAD
    if ($LASTEXITCODE -ne 0) { throw 'the reviewed implementation commit is not an ancestor of HEAD' }
    $changed = @(& git diff --name-only $P5A.Implementation HEAD)
    if ($LASTEXITCODE -ne 0 -or $changed.Count -ne 1 -or $changed[0] -cne $P5A.LaunchDoc) {
        throw "HEAD differs from the reviewed implementation beyond the launch document: $($changed -join '; ')"
    }
    $env:HF_HUB_OFFLINE = '1'
    $env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
    $env:PYTHONPATH = Join-Path $P5A.Worktree 'src'
    foreach ($name in $P5A_ThreadVars) {
        if (Test-Path -LiteralPath "Env:$name") { Remove-Item -LiteralPath "Env:$name" }
    }
    if ($Cuda -eq 'Hidden') {
        $env:CUDA_VISIBLE_DEVICES = '-1'
    } elseif (Test-Path -LiteralPath 'Env:CUDA_VISIBLE_DEVICES') {
        Remove-Item -LiteralPath 'Env:CUDA_VISIBLE_DEVICES'
    }
    $py = $P5A.Python
    $origin = & $py -c "import importlib.util as u; print(u.find_spec('cg_fedllm').origin)"
    if ($LASTEXITCODE -ne 0 -or $origin -ne (Join-Path $P5A.Worktree 'src\cg_fedllm\__init__.py')) {
        throw "cg_fedllm does not resolve to this worktree: $origin"
    }
    $busy = @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe' OR Name = 'cgfed.exe'" |
        Where-Object { $_.Name -eq 'cgfed.exe' -or $_.CommandLine -match 'cg_fedllm\.cli' })
    if ($busy.Count -ne 0) { throw "another P5-A process may be running: $($busy.ProcessId -join ', ')" }
    if (Test-Path -LiteralPath $P5A.V1Root) { throw 'the v1 run root exists: stop and report; never delete it' }
    if (-not (Test-Path -LiteralPath $P5A.LaunchRecord -PathType Leaf)) { throw 'the execution record is missing' }
    $sha = (Get-FileHash -LiteralPath $P5A.LaunchRecord -Algorithm SHA256).Hash.ToLowerInvariant()
    $docBlob = & git rev-parse "HEAD:$($P5A.LaunchDoc)"
    if ($LASTEXITCODE -ne 0) { throw 'cannot resolve the launch document at HEAD' }
    Write-Host "execution HEAD $head; launch document blob $docBlob; execution record SHA-256 $sha; CUDA $Cuda"
    return $sha
}

function Assert-P5ASameRun {
    # Invocations 2-3 and closure-only: invocation 1 must name this execution HEAD and this execution record.
    param([Parameter(Mandatory = $true)][string]$RecordSha)
    $ErrorActionPreference = 'Stop'
    $first = Join-Path $P5A.RunRoot 'ledger\invocation_1.json'
    $code = "import json, sys; i = json.load(open(sys.argv[1], encoding='utf-8'))['identity']; " +
        "print(i['execution_commit']); print(i['launch_record']['path']); print(i['launch_record']['sha256'])"
    $py = $P5A.Python
    $id = @(& $py -c $code $first)
    if ($LASTEXITCODE -ne 0 -or $id.Count -ne 3) { throw "cannot read the identity of $first" }
    if ($id[0] -cne $P5A.ExecutionHead) { throw "invocation 1 ran at $($id[0]), not at the execution HEAD" }
    if ($id[1] -cne ($P5A.LaunchRecord -replace '\\', '/') -or $id[2] -cne $RecordSha) {
        throw 'the execution record is not the path and bytes that invocation 1 cited; never edit it'
    }
}
```

### 8.2 Step A: CPU-only real input preflight (NOT AUTHORIZED)

Usable only after S1-S3 and a dated written authorisation that names this preflight number, recorded in the execution
record. Fresh window; CUDA hidden; no AE, no CUDA context, no training, no start record (v2 §5.4.1).

```powershell
# STEP A - real CPU input preflight. NOT AUTHORIZED until the written authorisation of this preflight exists.
& {
    $ErrorActionPreference = 'Stop'
    $recordSha = Assert-P5AState -Cuda Hidden
    if (Test-Path -LiteralPath (Join-Path $P5A.RunRoot 'ledger\invocation_1.json')) {
        throw 'ledger\invocation_1.json exists: no preflight is permitted any more'
    }
    $done = @(1..3 | Where-Object { Test-Path -LiteralPath (Join-Path $P5A.RunRoot "preflight\preflight_$_.json") })
    if ($done.Count -ge 3) { throw 'three preflights exist: no further preflight is permitted' }
    if ($done.Count -eq 0 -and (Test-Path -LiteralPath $P5A.RunRoot)) {
        throw 'the run root exists before preflight 1: stop and report; never delete or edit it'
    }
    $freeMiB = [math]::Floor((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1024)
    Write-Host "free physical memory: $freeMiB MiB (copy into the execution record)"
    Write-Host "this will be preflight $($done.Count + 1); its authorisation must name that number"
    $cliArgs = @('-m', 'cg_fedllm.cli', 'p5a-preflight', '--config', $P5A.Config,
        '--runs-root', $P5A.RunsRoot, '--launch-record', $P5A.LaunchRecord)
    $py = $P5A.Python
    $ErrorActionPreference = 'Continue'    # a stderr line must never interrupt the running process
    & $py @cliArgs
    $rc = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    Write-Host "p5a-preflight exit code: $rc (cited execution record SHA-256 $recordSha)"
    if ($rc -ne 0) { throw "p5a-preflight failed (exit $rc): keep every file, do not retry, report the output" }
    Write-Host 'exit 0 only means a record was published; it is NOT a passing preflight. Review the record (8.5).'
}
```

### 8.3 Step B: one GPU training invocation (NOT AUTHORIZED)

Usable only after a separate dated written authorisation that names the invocation number and cites the reviewed,
passing preflight record by path and SHA-256 (version B of the execution record for invocation 1; the separate
authorisation file for invocations 2 and 3). One block runs exactly one invocation; there is no loop, retry or automatic
re-entry. Run one P5-A process at a time. Fresh window; `CUDA_VISIBLE_DEVICES` removed, all other settings identical
to step A.

```powershell
# STEP B - one P5-A training invocation. NOT AUTHORIZED until its own written authorisation exists.
& {
    $ErrorActionPreference = 'Stop'
    $recordSha = Assert-P5AState -Cuda Visible
    $ledger = Join-Path $P5A.RunRoot 'ledger'
    $done = @(1..3 | Where-Object { Test-Path -LiteralPath (Join-Path $ledger "invocation_$_.json") })
    if ($done.Count -ge 3) {
        throw 'three invocations exist: only step C applies, and only if summary_3.json is missing'
    }
    if ($done.Count -ge 1) { Assert-P5ASameRun -RecordSha $recordSha }
    Write-Host "this will be training invocation $($done.Count + 1); its authorisation must name that number"
    $cliArgs = @('-m', 'cg_fedllm.cli', 'p5a', '--config', $P5A.Config,
        '--runs-root', $P5A.RunsRoot, '--launch-record', $P5A.LaunchRecord)
    $py = $P5A.Python
    $ErrorActionPreference = 'Continue'    # a stderr line must never interrupt the running training
    & $py @cliArgs
    $rc = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    Write-Host "p5a exit code: $rc (cited execution record SHA-256 $recordSha)"
    if ($rc -ne 0) { throw "p5a failed (exit $rc): keep every file, do not delete temporaries, do not rerun, report" }
    Write-Host 'exit 0 is not a result: read stop, closed and every control state in the summary (8.5).'
}
```

### 8.4 Step C: closure-only completion (NOT AUTHORIZED)

Usable only if `ledger\invocation_3.json` exists and `ledger\summary_3.json` does not (invocation 3 died before its
summary), and only after a dated written authorisation of this step. Fresh window with CUDA hidden: the F11 guard
refuses closure when CUDA is available or already initialised in the process. It reads no payload, builds no AE,
starts no control and publishes no invocation record (v2 §13.5). It runs at most once; after any nonzero exit it is
not repeated without a reviewed written decision.

```powershell
# STEP C - closure-only completion after invocation 3 died. NOT AUTHORIZED until its written authorisation exists.
& {
    $ErrorActionPreference = 'Stop'
    $recordSha = Assert-P5AState -Cuda Hidden
    $ledger = Join-Path $P5A.RunRoot 'ledger'
    if (-not (Test-Path -LiteralPath (Join-Path $ledger 'invocation_3.json'))) { throw 'closure-only does not apply' }
    if (Test-Path -LiteralPath (Join-Path $ledger 'summary_3.json')) {
        throw 'summary_3.json exists: the run is closed'
    }
    Assert-P5ASameRun -RecordSha $recordSha
    $cliArgs = @('-m', 'cg_fedllm.cli', 'p5a', '--closure-only', '--config', $P5A.Config,
        '--runs-root', $P5A.RunsRoot, '--launch-record', $P5A.LaunchRecord)
    $py = $P5A.Python
    $ErrorActionPreference = 'Continue'    # a stderr line must never interrupt the running process
    & $py @cliArgs
    $rc = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    Write-Host "p5a --closure-only exit code: $rc (cited execution record SHA-256 $recordSha)"
    if ($rc -ne 0) { throw "closure-only failed (exit $rc): keep every file, do not repeat, report" }
}
```

### 8.5 Reading the result of a step (exit codes and F10)

| Step and exit | Meaning | Action |
|---|---|---|
| A, nonzero | a refusal (configuration, `--set`, run root, v1 root, missing execution record, preflight count) or a publication failure. A preflight that could not publish its record published nothing and does not count, but it may leave residue that blocks every later preflight (section 9) | keep every file; report; a repeat needs a reviewed cause and a new written authorisation |
| A, 0 | a preflight record was published. It may be **failing**: `pass` is per representation and overall | review the record in full (below) before anything else |
| B, nonzero, no new finalized `invocation_<k>.json` | refused or failed before its invocation record was finalized (entry rules, `check_same_run`, capability or byte refusal, a death during the first publication); it does not count | keep every file, including empty directories and temporaries; report |
| B or C, nonzero, F10 message `P5-A evidence publication incomplete` | `summary_publication_error`: the summary could not be published; or `closure_issues`: a closure terminal record could not be published (for example an occupied path, left untouched and listed in the summary, v2 §13.5). `cmd_p5a` raises `P5AProtocolError` with the details, so the CLI exits nonzero instead of reporting success. Every finalized file stays; nothing was deleted, overwritten or republished | preserve the run root as it is; do not delete temporaries; report. A missing summary is recovered only by a separately authorised step under the frozen rules: a later permitted invocation, or, for invocation 3, the closure-only step (precondition: `invocation_3.json` without `summary_3.json`). Closure issues are reported, never repaired by hand |
| B, nonzero after the invocation record was finalized, other message or process death | an unexpected exception or a death of the process; the invocation counts; any started control is recovered (never retrained) by the next permitted invocation or the closure-only step | as for F10 |
| B, 0 | the invocation finished and published `summary_<k>.json`. `stop` may be set (byte or time ceiling, publication failure, operator interrupt) and controls may be `INCOMPLETE` | read the summary and every new record (below) |
| C, 0 | closure completed; `summary_3.json` is the closing summary with the outcomes | read it; then the evidence commit (S8) |

Review after step A (`preflight\preflight_<k>.json`): `schema` `cg_fedllm.p5a_v2_preflight/v1`; `preflight` = k;
`protocol` = the pair of 2.1; `execution_commit` = the execution HEAD and `execution_git.dirty` = `false`;
`config_sha256` = `e5ac614c…bb122`; `launch_record` = the execution record path (forward slashes) and the SHA-256
printed by the block; `trainings_started` 0, `ae_built` `false`, `cuda_initialized` `false`; `cpu_environment` as in
4.2; per representation `pass`, `failure`, `scale_check` (`bitwise_equal` is expected and recorded, not a gate),
`rms_check` and `controls_with_sig_zero`. A representation without `pass: true` cannot train in any invocation, and a
failure must be resolved before invocation 1 (v2 §5.4.1, §12). A preflight is repeated only after an engineering or
launch error, with a new written authorisation, never with a changed value; every record, passing or failing, is
kept and cited, and training uses only the most recent one.

Review after step B or C: the new `invocation_<k>.json` (`execution_commit`, `launch_record`, `entry` for k = 1) and
`summary_<k>.json` (`stop`, `closed`, `recovered`, every control's `state`, `status` and `pre_start`,
`resource_state`, `retained_bytes_before_this_file`, and at closure `outcomes`, `reported_only_gate_results` and
`closure_issues`), plus every new start, end, failure and control record. Record every failure, interruption and
deviation in a **separate post-run findings/log file** under `D:\p5a_v2_execution\`, citing the unchanged
version B by path and SHA-256. Never append to or edit the cited execution record after invocation 1: changing
its bytes would invalidate the identity needed by subsequent invocations and closure-only.

## 9. Preconditions, stop conditions, evidence handling and recovery

**Before step A** (remaining execution conditions PENDING): written decision H2 (section 10); S0-S3 done;
the execution record (version A) reviewed; the v1 root `D:\cgfed-runs\phase5_p5a_seed1\` absent. The v2 root
`D:\cgfed-runs\phase5_p5a_v2_seed1\` must be absent before **preflight 1**; before preflight 2 or 3 it must
satisfy the unchanged preflight entry rules (only the earlier consecutive preflight records, no training-stage
file or publication residue). These checks run only within the authorised step; no root is created, edited or
deleted by hand. Free RAM is recorded before the step. A dated written authorisation names step A and its
preflight number only.

**Before step B, invocation 1** (all PENDING): a reviewed preflight record that passed for every representation that
is to train, from the same execution HEAD, protocol pair, configuration hash and CPU environment fingerprint; the
execution record updated to version B; a dated written authorisation naming training invocation 1. The F7 synthetic
diagnostic is done (section 10); a further v2 §15 smoke test is optional and needs its own authorisation.

**Operating rules.** One P5-A process at a time, on this machine and interpreter (v2.1 A2: concurrent first-record
claims are resolved by the no-clobber link, but a later process could be counted as invocation 2). No package
change, no Git operation that moves this worktree's HEAD or branch (commit, merge, pull, rebase, reset, checkout,
`git branch -f` from any checkout) and no edit of tracked files until the run is closed. No cleanup, overwrite or
manual edit under the run root, ever.

**Stop conditions.** Provenance or input mismatch: that representation stops. `sig = 0`: that control. C0 failure:
that control. Reaching the **per-training** 1,800 s limit, including an evaluation overrun, ends that control
`INCOMPLETE (RESOURCE STOP)` without a checkpoint; the invocation may continue subject to the remaining budgets.
An aggregate time/allocation refusal, or a byte-policy refusal at a training start or during control-artifact
publication, stops the invocation and closes the run. A refused run-level artifact is not published and stops
the invocation under the run-level rules; it does not independently establish run closure. Filesystem publication
failure, operator interrupt or capability failure stops the invocation. A caught training exception (including
OOM or deterministic-backend refusal) or non-finite loss ends that control `INCOMPLETE`; the invocation may
continue if its evidence can be published. Process death stops the process, with recovery in a later permitted
step. Determinism is never disabled and warn-only is never used.

**Evidence.** Finalized files are never deleted or overwritten. A publisher removes only its own temporary after
its successful link or its own failed write; surviving temporaries are counted and left untouched. Keep every
finalized start, end, provenance, failure and control record. A training that ends without a completed final
evaluation publishes no checkpoint. An `INCOMPLETE (PUBLICATION)` control may already have a finalized checkpoint
when a later provenance or control-record publication fails: keep and cite that checkpoint too (v2.1 A4.4).
A started control is never retrained; its evidence is recovered by a later permitted invocation or closure-only.

**Invocation-1 recovery (v2.1 A2, A3).** If training invocation 1 dies before `ledger\invocation_1.json` is
finalized, the next invocation may take the invocation-1 slot only if the root holds nothing but consecutive preflight
records, empty `preflight\` / `ledger\` directories and recognized invocation-1 or preflight temporaries. Those
temporaries are kept, counted and never deleted; a started, foreign or unknown state is refused. The first record is
publishable iff the measured footprint is at most 14,510,000 bytes; a retained temporary may legitimately refuse it,
and nothing is deleted to avoid that. The most recent preflight record is re-validated as usual. Such a re-entry needs
its own written authorisation (in the separate authorisation file); the cited execution record is not edited.

**Preflight publication residue (disclosed limitation, unchanged rule).** The preflight entry rule of v2 §13.1 is not
changed by v2.1 (A6). If preflight 1 dies leaving an existing root without a finalized preflight record, or any
preflight leaves a temporary before its record is finalized, every later preflight is refused (the root holds no
record, or a non-record file). In the preflight-1 case, training has no preflight record and cannot start. If
preflight 2 or 3 leaves a recognized temporary, the earlier records stay, and v2.1 A2 accepts those temporaries at
invocation 1, so training uses the most recent finalized record. A death before creating any new residue does not
itself change eligibility; the unchanged entry checks decide from the actual retained state.
Neither state is repaired by hand: stop, preserve it and report it for a reviewed decision (implementation review §9,
F1 row).

**F5 (historical, preserved).** Every full CPU suite run before 2026-10-05, including the 308-pass runs of
2026-10-04, created a CUDA context through an inherited `peft` import. Those runs did no GPU computation and accessed
no data; they are not restated as CUDA-free. The real preflight and the closure-only step run with
`CUDA_VISIBLE_DEVICES=-1` in a fresh process; the preflight record reports `cuda_initialized` truthfully.

**F7 (synthetic compatibility only).** One authorised synthetic-only diagnostic on 2026-10-05 ran the real
`train_and_evaluate` with the strict v2 §8.3 settings at `[1, 2048, 1536]`, batch 1 and batch 4, 2 optimizer steps
each, on the pre-audit working copy (HEAD `33e51fd` plus the uncommitted implementation): exit 0, no
deterministic-kernel error, contrary to the PyTorch 2.14 documentation for `ReflectionPad2d` CUDA backward
(implementation review §13.3, REPORTED). The post-audit changes do not touch the trainer: the `p5a_run.py` blob diff
`fd6b666` -> `412547f` is one hunk inside `production_closure` (VERIFIED). F7 establishes compatibility for those
synthetic paths only. It does not establish real-data timing, memory, fit, full-run feasibility or run-to-run
repeatability. v2 §8.3 still applies: a refused kernel makes that training `INCOMPLETE (INTERRUPTED)`.

**Resources** (DERIVED from the code, not measured). One R2 or R3 population is 80 x 12,582,912 = 1,006,632,960 bytes
of fp32 inputs. The RMS check `torch.stack(xs).pow(2)` holds two more tensors of that size, so the peak is about three
copies (about 3.0 GB) for tensors alone, in the preflight and in every training invocation that rebuilds R2 or R3.
Implementation review F8 gives "about 2.0 GB"; by this arithmetic one copy is about 1.0 GB. The protocol prescribes no
minimum; the free RAM is recorded before each step.

**Residual risk (not verified, not a blocker).** Equality of `cpu_environment()` between the CUDA-hidden preflight
process and the training process, which also imports `pipeline` and initialises CUDA (`apply_vram_guard`) before it
reads the fingerprint, is supported by static reading only: the fields are explicit flags, versions, thread counts and
environment values, and no module-level thread, precision or determinism setter was found in `src/` or in the
installed `transformers`, `peft`, `accelerate`, `datasets`, `huggingface_hub`, `safetensors` and `tokenizers`
packages. It has not been observed on this machine. A difference would make every representation
`INCOMPLETE (PROVENANCE)` before any start, and that invocation would count. Whether to observe it beforehand with a
separately authorised synthetic check is a reviewer decision; this document does not authorise one.

## 10. Milestones, pending decisions and authorisations

| # | Item | Status |
|---|---|---|
| H1 | Implementation review, findings F1-F11 and their fixes, external diff audit (implementation review §§9-14) | DONE 2026-10-05, as recorded there. That record describes its reviews as AI-assisted engineering reviews, not a human acceptance; this document cites no human acceptance event beyond the commit itself (H5) |
| H2 | Written acceptance of engineering choices 1, 3, 4 and 5 (implementation review §13.4: implemented in `603acfc`, recommended for acceptance, stated there as pending). Choice 3 is the identity comparison that section 7 relies on | PENDING written decision, needed before step A. It accepts or rejects implemented behaviour; it does not reopen a frozen protocol choice |
| H3 | Engineering choice 2 | DECIDED 2026-10-05; frozen as v2.1 A4 (`33e51fd`) |
| H4 | Finding F1 | DECIDED 2026-10-05 (restricted option 1); frozen as v2.1 A2/A3 (`33e51fd`) |
| H4a | N2 (latest preflight only), N3 (exact agreement), P1 (pinned CPU environment) | DECIDED 2026-10-05; v2.1 A5 |
| H5 | Implementation commit | DONE: `603acfce3fcfbe6724f7900c78dc03e28071f487`, committed 2026-10-05 and pushed (remote-tracking ref at `603acfc` on 2026-10-06) |
| H6 | This launch-preparation document | External document audit COMPLETED 2026-10-06 (section 13); READY FOR DOCUMENTATION-ONLY COMMIT (S1, section 7.4) |
| H6a | External execution record, version A (S3) | PENDING; created only after S1 and S2 |
| H7 | Written authorisation of the real CPU preflight (step A), naming the preflight number | NOT GRANTED |
| H8 | Synthetic GPU compatibility diagnostic (F7) | GRANTED once and DONE 2026-10-05 (implementation review §13.3); not a real P5-A step, nothing under a run root |
| H9 | Written authorisation of training invocation 1 (step B), citing the reviewed passing preflight record; may be conditional on H7's result | NOT GRANTED |
| H10 | Each further invocation (2, 3), any invocation-1 re-entry and the closure-only step (step C) | NOT GRANTED; each needs its own written authorisation |
| H11 | Evidence commit after closure (`results/phase5/p5a_v2/`, findings, execution-record copies) | PENDING, separate reviewed change |

**Validation history** (REPORTED; nothing was rerun in this round). Pinned interpreter, imports from this worktree.

| Date | Code bytes | Result | Scope and limitations |
|---|---|---|---|
| 2026-10-04 | pre-review implementation | full CPU suite, 308 passed, 2 deselected | created a CUDA context through an inherited `peft` import (F5); historical |
| 2026-10-05 | after the first acceptance review | full CPU suite, 313 passed, 2 deselected; `CUDA_VISIBLE_DEVICES=-1`, session probe: CUDA never initialised | historical (review §12) |
| 2026-10-05 | after the v2.1 reconciliation, before the external audit | full CPU suite, 324 passed, 2 deselected, 88.8 s; probe: CUDA never initialised | historical evidence for the pre-audit bytes, not for the F10/F11 fixes (review §13.5, §14) |
| 2026-10-05 | post-audit bytes, as committed in `603acfc` | Phase-5-scoped: 188 passed in 54.19 s; Ruff check and Ruff format check exit 0; `git diff --check` clean (review §14); the staged whitespace check (`git diff --cached --check`) is reported clean by the session that made the commit | native Windows; earlier-phase suites not rerun; no CUDA-initialisation probe is reported for this run |

## 11. Not executed / unknown

The launch-preparation commit, the execution HEAD, the execution record and future real-step authorisations H7, H9 and H10;
preflight records; real input identities, payload hashes and Phi hashes; recomputed maxima and RMS; measured artifact
bytes; real-data GPU feasibility, full-run timing and memory; free VRAM and the allocator cap at launch; GPU
run-to-run repeatability; all training curves, gate values and outcomes: **NOT EXECUTED / UNKNOWN**. Measured only on
synthetic tensors: deterministic-kernel acceptance and memory of 2 optimizer steps at batch 1 and batch 4 (F7).

## 12. Changes from the previous draft

Previous draft: untracked, 2026-10-05, SHA-256 `b649722e41506673d6f83e8ccc742335a50979327359cf481351aa1eef7f709c`.

* Status, identities and milestones: the implementation commit `603acfc` (committed and pushed) replaces "PENDING";
  launch-preparation commit, execution HEAD and cited record identity separated (2.1, 7.1).
* Fingerprints refreshed from the committed bytes (`cli.py`, `p5a_run.py` changed by F10/F11; ten unchanged);
  inheritance chain and three shared helpers added; the three configuration hashes defined and kept apart (2.2, 2.3).
* The cited launch record is no longer a repository file. The previous plan cited
  `docs/phase5_p5a_v2_launch_record.md` and required an update of it after the preflight: committing that update
  would move HEAD and make the preflight stale, and leaving it uncommitted would run training from a dirty tree. It
  is replaced by the sequence of 7.3.
* Commands rebuilt as checked PowerShell blocks: worktree imports enforced, full `cpu_environment` pinning,
  `CUBLAS_WORKSPACE_CONFIG` set, explicit `$LASTEXITCODE` checks, CUDA hidden in a fresh process for steps A and C
  (F11), exit-code meaning and F10 handling (8.5).
* Added: preflight publication residue, the 14,510,000-byte first-record boundary, A4.2 precedence, artifact paths,
  outcome authorisations, F7 scope (pre-audit bytes, trainer unchanged), validation history, RAM arithmetic.

## 13. External launch-document audit (2026-10-06)

Scope: this launch document only, assessed against the protocol pair and relevant production source at
`603acfce3fcfbe6724f7900c78dc03e28071f487`. This is an AI-assisted document audit, not a new human authorisation,
software-suite run, preflight or GPU check. Original submitted bytes: SHA-256
`b8b37e309408330c13f775509c89f3604b52ecfdbd7c9a0d484bb82025777342`.

Verified independently from GitHub bytes at the pinned implementation commit: all 19 source/configuration/helper
rows in section 2.2 match both Git blob IDs and source-byte SHA-256; both protocol documents, the audit response and
implementation-review fingerprints match section 2.1; implementation parent/tree identities and the cited test blob
match. Static comparison confirms the complete 14-file R4 allowlist and the three frozen max/scale/RMS rows against
the committed Phase-5 constants. No earlier-phase tests or experiment evidence were re-audited.

Forward document corrections made before committing:

* Execution-record immutability: post-invocation findings go into a separate file, rather than changing the cited
  record and invalidating recovery identity (7.2, 8.5).
* Version A construction: capture repository/OS facts before completing it; log the subsequent record-hash output
  separately. `Assert-P5AState` requires the completed record and cannot construct that record itself (7.3).
* Stop/evidence semantics: a per-training timeout differs from aggregate run closure; run-level byte refusals keep
  their own rules; a publication-incomplete control may retain an already-finalized checkpoint (9; v2.1 A4.4).
* Entry/housekeeping wording: root absence applies to preflight 1; later preflights use their retained-state checks;
  successful publishers remove their own temporary link; process death blocks a later preflight only when the
  retained state fails the unchanged entry rules (6, 9).
* Readiness and history: this document is ready for its separate documentation-only commit; the completed synthetic
  authorisation is not relabelled unknown (10-11). Real-step authorisations remain pending.

All four PowerShell command blocks are byte-identical to the submitted document. The earlier reported Windows
parser/mock results therefore retain their stated scope; they were not rerun here. No code, configuration, tests,
frozen protocol or historical implementation-review bytes changed. No real input/run-root access, CUDA, preflight,
training, staging, commit, push, PR or merge occurred in this external review environment.

Verdict: **READY FOR DOCUMENTATION-ONLY COMMIT** after applying these document corrections. The external execution
record still needs the actual post-commit execution HEAD and its review. Real preflight and training remain
**NOT AUTHORIZED**; this document audit grants neither.
