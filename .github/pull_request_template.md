## Summary

<!-- Describe the single logical purpose of this pull request. -->

## Validation

- [ ] `git status` reviewed
- [ ] `git diff` reviewed
- [ ] `git diff --stat` reviewed
- [ ] `git diff --check` passes
- [ ] `ruff check src tests scripts` passes
- [ ] `pytest -q -m "not gpu and not model"` passes, or the reason it was not run is documented

## Review checklist

- [ ] The pull request has one coherent purpose
- [ ] No unrelated changes are included
- [ ] Only intended files were staged
- [ ] No caches, logs, checkpoints, downloaded models, or raw large artifacts were added unintentionally
- [ ] README, docs, configs, results, tests, requirements, and CI were updated where relevant
- [ ] Experimental criteria or protocols were not changed in response to observed results
- [ ] Claims clearly distinguish measured, derived/inferred, paper-reported, and unknown information
- [ ] Published history was not rewritten unexpectedly

## Formatting

<!--
Repository-wide Ruff formatting is not yet a required gate because the current
baseline has not been normalized. Do not mix bulk formatting into unrelated PRs.
-->

- [ ] This PR does not introduce unrelated repository-wide formatting changes

## Evidence / experiment notes

<!--
For experimental changes, link the relevant config, result, preregistration,
deviation, or provenance record. Delete this section when not applicable.
-->

N/A
