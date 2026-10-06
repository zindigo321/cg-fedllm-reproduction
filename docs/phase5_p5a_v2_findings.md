# P5-A v2 + v2.1: invocation 1 findings

Date: 2026-10-06. Result label: `PHASE5-DIAGNOSTIC`; arithmetic and this artifact audit are `DERIVED` from the preserved production evidence.

**Invocation 1 is CLOSED. All six controls completed 3,000 iterations and are `RECORDED` / `FIT FAIL`. For each of R2, R3 and R4, the frozen outcome is `CAPACITY FIT NOT DEMONSTRATED`. The complete supplied artifact archive passes the checks described below and is ready for human archival review.**

This is a completed negative fit result under the frozen protocol. It is not an interrupted, resource-stopped, publication-failed or inconclusive experiment. R2/R3 are the prespecified primary representations; R4 is secondary. There is no combined P5-A pass or aggregated representation verdict.

## Execution and authorization identity

- Execution HEAD: `6603bd8d9156831562dd76c0bf9dba5a071e867f`, branch `worktree-p5a-engineering-support`. The producer records agree on a clean execution tree. The implementation parent is `603acfce3fcfbe6724f7900c78dc03e28071f487`. This archival change is made after run closure and does not replace that execution HEAD in historical records.
- Frozen v2 protocol: commit `dd4bf3f4348fd3b2b1b2c4ad23e7204fbef8369c`, blob `e80d21aeda653ab6d1339d34f7b88118588855df`, `docs/phase5_preregistration_v2.md`.
- Frozen v2.1 addendum: commit `33e51fd63c6cd0679e76e12380ae5d418158d698`, blob `28e05f9ee33ed13386d346fce6e47e9108c937d7`, `docs/phase5_preregistration_v2_1.md`. The protocol identity is the pair.
- Configuration: `configs/phase5/p5a_v2_seed1.yaml`, no overrides. Resolved SHA-256: `e5ac614c5f176f39b40ddd96cfe311f50c8209151578a1535794cd55d87bb122`. All six resolved YAML files independently hash to this value using the canonical JSON configuration definition.
- Accepted real preflight 1 SHA-256: `b058eb1bf01d101e1800bdf05d7701b5ac3de515ea71cd542f70403ee5d48cee`. Its complete CPU environment fingerprint exactly matches invocation 1 and every control's evidence.
- Immutable execution record B: `D:/p5a_v2_execution/p5a_v2_execution_record.md`, SHA-256 `fc7bc274efd7c705fb3d60d9a03ebd5b0a42620b7002d82305e18262a2976dba`. Its actual archived bytes match the finalized H9 record. Archived A SHA-256: `e36de843549ef2e169ba79e6883e265de69a0f91fac93aa66a0df8fbc08c0c02`.
- H9 authorized only training invocation 1 at **2026-10-06T11:25:52Z**. The external one-use claim is at 11:39:57.6139409Z; the invocation record starts at 11:40:02Z. The console reports exit 0.

The authorization did not grant automatic retries, re-entry, another invocation, closure-only or another GPU diagnostic. No such operation is part of this review or archival proposal. All original run-root files and cited execution records remain immutable.

## Frozen instrument and inputs

The six-control order is R2 single/subset4, R3 single/subset4, R4 single/subset4. Each uses the fixed ResNet-3, BatchNorm, Tanh output, Adam 2e-4, betas (0.9, 0.999), epsilon 1e-8, zero weight decay, AE seed 1 and final iteration-3,000 checkpoint. Batch size is `min(4, k)`: 1 for single and 4 for subset. Each checkpoint contains the final BatchNorm buffers; evaluation uses eval mode with running statistics.

Input geometry is `[1, 2048, 1536]` (3,145,728 elements). The recorded latent shape is `[64, 32, 24]` (49,152 elements): an exact element ratio of 1/64, not a measured operational network-byte ratio.

Selections, complete source provenance and selected Phi hashes match the accepted preflight entry for entry. R2/R3 each use the 80 training entries and positions 39 (single) / 0, 26, 53, 79 (subset). R4 uses the five frozen t=0 entries and positions 2 (single) / 0, 1, 3, 4 (subset). R4 evidence matches the exact 14-file gradient allowlist and contains gradient-only selected-record provenance. The original input payloads were not reopened during this artifact audit.

The exact max-abs and RMS evidence agrees bit for bit with preflight. Frozen scales are R2 0.034553585083861103, R3 0.015751632224572334 and R4 0.06768756791165001. Selected-range reports contain no values at or beyond absolute normalized value 1. All six C0 Tanh-range feasibility gates pass.

## Final gates and outcomes

Values shown here are rounded; checks and the committed JSON retain the original binary64 numbers. RSE means relative squared error. The table reports maximum per-snapshot RSE and minimum per-snapshot cosine; it does not average snapshot ratios.

| Representation | Control | Iterations | Charged GPU wall seconds | Max RSE | Min cosine | Status |
|---|---|---:|---:|---:|---:|---|
| R2 | single_snapshot | 3000 | 250.625 | 0.995254505 | 0.083535178 | FIT FAIL |
| R2 | fixed_subset4 | 3000 | 419.187 | 1.008728078 | 0.020734642 | FIT FAIL |
| R3 | single_snapshot | 3000 | 316.047 | 0.993808162 | 0.087180367 | FIT FAIL |
| R3 | fixed_subset4 | 3000 | 592.672 | 1.001282927 | 0.037887195 | FIT FAIL |
| R4 | single_snapshot | 3000 | 247.234 | 1.329541511 | 0.001951016 | FIT FAIL |
| R4 | fixed_subset4 | 3000 | 454.141 | 1.582031079 | 0.001244500 | FIT FAIL |

Single-snapshot gates require RSE <= 0.01 and cosine >= 0.99. Subset gates require RSE <= 0.50 and cosine >= 0.90 for every snapshot, plus C4. All six C1 fields report finite reconstructions and finite gated metrics. Every selected snapshot fails both C2 and C3; each subset fails C4:

| Representation | Pooled AE RSE | Pooled subset-mean RSE | C4 bound (0.8 x mean) | C4 |
|---|---:|---:|---:|---|
| R2 | 1.001818059 | 0.682059743 | 0.545647795 | FAIL |
| R3 | 0.997918891 | 0.750884144 | 0.600707316 | FAIL |
| R4 | 1.244580653 | 0.155917033 | 0.124733626 | FAIL |

The audit independently recomputed all **78 metric sets** (per snapshot and pooled for each recorded predictor) from their recorded SSE/SIG/HAT/DOT sums, using the full element counts. It checked exact pooling by `math.fsum`, the zero baseline, subset-mean consistency, and all **66 C0/final threshold comparisons**, margins, booleans and four-ULP near-threshold flags. No gated comparison is near its threshold. No epsilon was added to any gate.

The precise permitted conclusion is **CAPACITY FIT NOT DEMONSTRATED** for each representation, under the P5-A v2 protocol (fixed ResNet-3, exact training-population max-abs scaling to 0.95, Adam 2e-4, 3,000 iterations, final checkpoint, AE seed 1, training snapshots only).

This does not establish incompressibility, insufficient general AE capacity, unseen-snapshot generalization, seed stability, the cause of the prior F6 failures, operational FAF semantics or the paper's compression claims. All three outcome records have `authorises: nothing`; no downstream re-screen or FAF experiment follows automatically.

## Curves and interpretation limits

Each control has all 61 expected curve points: iteration 1 and every 50 iterations through 3,000. All losses are finite and nonnegative.

| Representation | Control | Training MSE at 1 | Training MSE at 3,000 |
|---|---|---:|---:|
| R2 | single_snapshot | 0.209050625563 | 0.0146431773901 |
| R2 | fixed_subset4 | 0.205292642117 | 0.0130754373968 |
| R3 | single_snapshot | 0.212001353502 | 0.0171431340277 |
| R3 | fixed_subset4 | 0.212845429778 | 0.0185282826424 |
| R4 | single_snapshot | 0.19030790031 | 0.00035542214755 |
| R4 | fixed_subset4 | 0.181653723121 | 0.000272098899586 |

These are normalized-space batch losses recorded before the optimizer update at the listed iteration, in training mode. The gates use original-space metrics from final eval-mode reconstruction. A descending training curve alone does not satisfy the frozen relative-error or direction gates. The archive does not identify a causal explanation for the failed fit; this review performs no additional model evaluation or experiment.

## Resource and publication accounting

- Six starts, in the frozen order; six completed end records. No failure records, recoveries, reported-only gate outcomes, retained temporaries, extra preflights, later invocations or closure issues appear in the archive.
- End/control timing fields agree exactly. Every control is below 1,800 seconds. Independent `math.fsum` of the six end records gives **2,279.905999999988 seconds** (displayed as **2,279.906 seconds**, or about 38 minutes), below 10,800 seconds. `may_start=false` follows reaching six starts.
- Reported input/setup time: **44.419804 seconds**. The previously reviewed console wrapper interval is 07:39:57–08:18:53 America/New_York, **2,336 seconds (38 minutes 56 seconds)**. Wrapper duration includes setup and publication; it is not GPU charge.
- The archive's run root has **45 files / 12,361,855 decimal bytes**, below the 20,000,000-byte ceiling. Before summary publication it had 12,339,252 bytes; the 22,603-byte summary brings the replay to the exact final inventory.
- All 45 publication steps were replayed in order, including their measured footprints, every producer receipt's path/size/footprint and the per-artifact/shared-provenance caps. The reservation formula was separately derived from the pinned rules and state at each publication, and fits throughout. Each provenance set is 7,613 bytes under its shared 150,000-byte cap; each final checkpoint is under 2,100,000 bytes.
- The archived provenance reports an RTX 4060 Laptop GPU with 8,585,216,000 bytes. The allocator guard is 7,182,745,600 bytes, equal to observed free VRAM minus 256 MiB. This is a recorded allocation cap, not a measurement of peak GPU memory.

## Checkpoint and archive integrity

The full review ZIP is `p5a_invocation1_full_review_20261006.zip`, **11,163,077 bytes**, SHA-256 `1e5cc8ed95c4dd13864aa53f66918c7984be9268e2d15bd4b3bac89ce9d3164f`, with 50 regular entries and no CRC error. All extracted bytes match archive entries. The earlier supplied summary, invocation and two log files are byte-identical to their bundle copies; native CLI output exactly projects the finalized summary.

Each checkpoint's complete safetensors header and data boundaries were checked without importing Torch. The six checkpoints share the same 152-tensor dtype/shape structure; every F32 element is finite, every running variance nonnegative, and all 25 BatchNorm `num_batches_tracked` buffers equal 3,000. Checkpoint metadata matches architecture, normalization, representation, control and final iteration. The six byte hashes are distinct. R3 single's extra eight header-padding bytes are valid.

| Representation | Control | Checkpoint bytes | SHA-256 |
|---|---|---:|---|
| R2 | single_snapshot | 1,996,500 | `7205d2c64650a0093e8cd13e29e37196cee1c892b5714d31ff18bb2ff244a2aa` |
| R2 | fixed_subset4 | 1,996,500 | `b82fae484e40c36bc06159c37439a6e2a82acd94ab570df1fe9aa1f5def6e155` |
| R3 | single_snapshot | 1,996,508 | `dd46dbccba65c53b3c709ea46cedaaea9a470177d2d72dc2cfe1a19cb2dbbbc0` |
| R3 | fixed_subset4 | 1,996,500 | `9a62a2c724a8db4611f54e2309718fcc5ed1e63655d0d0b1d579f7eb4641da36` |
| R4 | single_snapshot | 1,996,500 | `a3cd6259df4d3aa6516ac0c2e79cb6194016d2803846ef540ce2e28639345441` |
| R4 | fixed_subset4 | 1,996,500 | `4979cecd64b6914cc7f34fed673516929072cc865548d01e83d2ba4aac152ecd` |

No reconstruction is rerun here: C1 reconstruction-element finiteness remains a producer observation. The original input tensors and their raw SSE/SIG/HAT/DOT reductions are not independently recomputed. This audit verifies recorded arithmetic, identities and checkpoint structure/content, not full numerical replay, GPU peak memory or repeatability. The inventory describes the supplied archive; the import helper must confirm the live Windows originals still match before copying evidence.

## Archival scope and next boundary

Frozen v2 section 14 and launch step S8 require a separate reviewed evidence change after closure. `results/phase5/p5a_v2/` preserves all six control records, preflight 1, every invocation/start/end/summary JSON, and the six small provenance sets byte for byte. `audit.json` and `artifact_manifest.json` are new derived audit/index records. The complete ZIP, original execution records, logs and checkpoints remain external and are indexed by bytes and SHA-256.

This change updates only those result files, this findings document, the README's Phase-5 statements and the Phase-5 part of `docs/deviations.md`. Protocol texts, launch document, implementation, configuration and tests are unchanged. The historical F5 disclosure is retained. No new invocation-1 protocol deviation was identified within the received evidence and the stated review limits.

Validation for this evidence/documentation change consists of the complete artifact audit, byte-preservation checks, exact scope/hash checks and Git whitespace/diff inspection. Per the user's Phase-5-only instruction, previous-phase reviews and the already completed software test suites were not repeated. No new preflight, GPU work or model evaluation was run. Human acceptance of this concrete archival package and any future experiment authorization remain separate decisions.
