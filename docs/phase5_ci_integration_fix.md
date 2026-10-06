# Phase-5 archive CI integration fix

## Trigger and resulting behavior

The archival commit `fcb6a603c86c44ef1a184eef0a653b3f5b474bf9` added the six
original P5-A `config.sha256.json` provenance files under
`results/phase5/p5a_v2/`. These files contain configuration hashes, not a
scientific result label. The existing all-results JSON label sweep nevertheless
required a label in every JSON file and failed on the first hash receipt.

The fix recognizes exactly those six relative paths and validates each receipt
instead: the exact two-field schema, lowercase SHA-256 syntax, equality of the
configuration and identity hashes, the canonical hash of the neighboring
resolved YAML, and the receipt's byte count and SHA-256 in the archival manifest.
All other JSON files continue through the existing result-label checks.

The original receipts, manifest, experimental records, frozen protocols, launch
record and implementation are unchanged. This is a forward CI integration fix;
the archival commit and historical execution HEAD
`6603bd8d9156831562dd76c0bf9dba5a071e867f` remain intact.

## Failure evidence

GitHub Actions run
[37470401823](https://github.com/zindigo321/cg-fedllm-reproduction/actions/runs/37470401823)
on archival HEAD completed with Ruff lint and format passing, and
`1 failed, 329 passed, 2 deselected` in the CPU pytest step. The sole failure was
`test_every_committed_result_stays_valid_under_the_current_label_schema`:
`config.sha256.json carries no result label`.

## Validation and scope

At patch preparation, an isolated pure-Python check reproduced that failure and
passed the updated label/hash sweep over the real 41-file Phase-5 archive plus
one synthetic historical-label fixture. It also exercised four invalid hash
metadata cases, two incorrect manifest entries, five disallowed paths, and an
unknown unlabelled JSON file. The four earlier-phase quantitative test functions
and the existing label migration assertions are unchanged.

Pytest and Ruff were unavailable in that isolated environment; those results
are not represented as pytest or Ruff passes. The provided Windows preparation
script uses the pinned interpreter and runs Ruff only on the changed test file,
then selects the affected regression and seven new Phase-5 cases:

```powershell
& "D:\conda_envs\cgfedllm\python.exe" -m pytest -q tests/unit/test_recorded_results.py -k "every_committed_result or p5a_config_hash" -m "not gpu and not model"
```

The expected targeted result is `8 passed, 4 deselected`. The four existing
quantitative tests are not run locally. GitHub CI must pass on the new pushed
head before merge. This fix authorizes no preflight, training, retry or new
experiment, and changes no Phase-5 scientific conclusion.
