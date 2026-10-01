# Architecture (Phase 2 foundation + Phase 3 diagnostics)

```
src/cg_fedllm/
  config.py              strict dataclass schema, YAML inherit/overrides, ${VAR:-default} paths, hashing
  utils/                 hashing (canonical JSON, tensors), atomic IO, seed derivation, provenance,
                         GPU allocator cap against silent WDDM shared-memory spill (gpu.py)
  data/
    dolly.py             pinned source fetch + SHA-256 verification, records by source_id
    partition.py         Shepherd-compatible partition (pandas<3 semantics), per-client D1/D2 split
    manifest.py          canonical hash-stable manifests, validation, prepare/verify
    formatting.py        Alpaca template, tokenisation, DataCollatorForSeq2Seq-style collation
  evaluation/            reference_eval_v1: benchmarks (pinned parquet), prompts, scorer, aggregate,
                         orchestration; held-out loss
  models/
    loading.py           pinned revisions + file allow-lists; seeded tiny LLaMA for CPU tests
    adapter.py           AdapterState: canonical keys, explicit order, hashing, IO, arithmetic
    lora.py              explicit PEFT attachment, seeded init, LoRA-only trainability checks
  compression/
    representation.py    adapter_state | adapter_delta
    layout.py            Phi / Phi^-1 layouts + geometry validation (layer_major_qkvo_AtB, module_major_qkvo_AtB)
    autoencoder.py       reconstructed ResNet-3 AE (+ checkpoint IO)
    macs.py              meta-device MAC counting (input/output-grid conventions)
    codecs.py            Identity / AutoEncoder / ConstantMean / GaussianNoise codecs, payload accounting
    metrics.py           MSE, paper-style SNR, standard SNR (dB), cosine, innovation ratio
    normalization.py     [P3] frozen D1-train-fitted AE input normalisation: none | global_rms | factor_rms
    diagnostics.py       [P3] factor-aware report: transmitted/state/innovation x {A,B,all}, B*A product,
                         offline FedAvg replay, shift control
  federated/
    sampling.py          Shepherd round-seeded sampler
    aggregation.py       sample_weighted_mean | uniform_mean | literal_sum (diagnostic)
    client.py            LocalTrainer (one client, one round)
    simulator.py         single FL code path (LoRA-FT and FAF), checkpoints, resume, divergence status
  tgap/
    snapshots.py         common snapshot schema, writer/reader, tamper checks
    collect.py           local_pretrain | federated_pretrain
    train_ae.py          AE training (+ normaliser, final + best-D1-validation checkpoints) vs baselines
    stats.py             [P3] TGAP snapshot-set statistics (participation, distributions, cosines, split)
    viability.py         [P3] A1 report for AE / zero / train-mean / identity / Tanh-range ceiling, A6 gate
  pipeline.py            config -> model bundle / data bundle / codec / run directories
  smoke.py               Tier-C end-to-end pipeline + equivalence checks + summaries
  bench.py               bounded GPU micro-benchmarks
  calibration.py         [P3] realistic-sequence timing/memory, micro-batch decision rule, loss-normalisation
                         diagnostic
  cli.py                 `cgfed` command line (P3: calibrate-train, microbatch-diag, tgap-stats, ae-viability,
                         ae-select)
```

## Data flow of one FAF round

```
G_t ──► LocalTrainer.train(G_t, D2_i, seeds(t, i)) ──► end_i
          representation.to_representation(end_i, G_t)        (state | end_i - G_t)
          layout.forward(rep)              X_i ∈ R^{1×d×2·4·L·r}
          codec.encode(X_i, ctx)           payload_i (logical bytes logged)
          codec.decode(payload_i, ctx)     X̂_i                 (server)
          layout.inverse(X̂_i)              rep̂_i
          representation.recover_state     ŝ_i = rep̂_i | G_t + rep̂_i
aggregate([ŝ_i], [n_i], strategy) ──► G_{t+1}  (A and B independently) ──► atomic checkpoint
```

`codec=None` skips the boxed Phi/codec steps (the uncompressed baseline). The IdentityCodec path is
bitwise equal to it (CPU and, in the smoke run, GPU), which validates the machinery.

## Design rules

* One FL code path for every codec; controls differ only in the codec object.
* All randomness derives from `(seed, namespace, round, client, purpose)`; resume is bit-identical.
* Every scientific assumption is a config value; unknown keys are errors.
* Run outputs, checkpoints, snapshots, models and datasets live outside Git (`CGFED_RUNS`, `CGFED_CACHE`,
  `HF_HOME`); only manifests/fixtures with integer IDs and lightweight result summaries are committed.
* Every run writes its resolved config, config hashes and environment/Git/model/data provenance.

## Scripts (thin wrappers / audit tools)

`scripts/run_smoke.py`, `run_fl.py`, `collect_tgap.py`, `train_ae.py`, `evaluate.py`, `bench_gpu.py`, `capture_env.py` wrap the
`cgfed` commands. Audit tools: `prepare_data.py` (rebuild/verify manifests), `make_shepherd_oracle.py`
(re-run pinned Shepherd code under pandas < 3 to regenerate the partition oracle),
`crosscheck_mmlu_lmeval.py` (lm-eval reference, device-split), `summarize_eval_run.py` (condense an evaluation
run into a labelled, hash-referenced result file).
