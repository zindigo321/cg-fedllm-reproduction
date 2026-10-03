# Contributing

This repository is an independent, evidence-oriented reproduction of
CG-FedLLM. Changes should remain reviewable, reproducible, and explicit about
what is measured, inferred, paper-reported, or unknown.

## Development workflow

Use a branch for each logical change.

1. Start from the latest `main`.
2. Make one logical change at a time.
3. Inspect the diff before staging.
4. Run the relevant validation commands.
5. Stage only files related to the change.
6. Commit with a concise Conventional Commit message.
7. Push the branch and open a pull request.
8. Merge only after CI and manual review.

Do not commit directly to `main`.

## Commit boundaries

Use this rule:

> one commit = one logical change

A commit should:

- have one clear purpose;
- be describable accurately in one short sentence;
- leave the repository in a reasonable, reviewable state;
- avoid unrelated cleanup or opportunistic edits.

Do not split changes mechanically by file or line count. A logical change may
touch multiple files when those files are required to keep the repository
consistent.

Avoid `git add .` during normal development. Stage only intended paths.

## Commit messages

Use Conventional Commits:

- `feat(scope): ...`
- `fix(scope): ...`
- `test(scope): ...`
- `docs(scope): ...`
- `refactor(scope): ...`
- `perf(scope): ...`
- `build(scope): ...`
- `ci(scope): ...`
- `chore(scope): ...`

Subjects should be written in English, use imperative style, omit the trailing
period, and stay at or below 72 characters when practical.

Examples:

```text
feat(tgap): add snapshot collection
fix(eval): bound MMLU batch memory
test(codec): cover adapter round trip
docs(phase3): record AE viability result
```

Use subject-only commits by default. Use a short body only when the reason for
a change cannot be understood from the diff.

Detailed experiment results, measurements, provenance, and design arguments
belong in `docs/`, `results/`, pull requests, issues, or experiment records.

Do not add `Co-Authored-By` trailers for AI tools.

Custom types such as `bench:` or `exp:` must be documented here before use.

## AI-assisted changes

AI-generated changes are proposals, not automatically accepted changes.

Before committing, inspect at least:

```text
git status
git diff
git diff --stat
git diff --check
ruff check src tests scripts
ruff format --check src tests scripts
pytest -q -m "not gpu and not model"
```

Also check that:

- no unrelated files were changed;
- one commit does not contain multiple independent tasks;
- experimental criteria were not changed after seeing results;
- logs, caches, checkpoints, downloaded models, or raw large files were not
  accidentally committed;
- claims do not exceed the available evidence;
- README, docs, configs, results, tests, requirements, and CI remain consistent
  where relevant.

Do not commit merely because tests pass.

## Formatting

Ruff formatting is a required local and CI gate:

```text
ruff format --check src tests scripts
```

Do not mix repository-wide formatting with feature, experiment, or
documentation changes.

## Experimental integrity

Do not change an experimental protocol, metric, gate, representation choice,
or selection rule after observing results unless the change is explicitly
recorded as a new experiment or protocol revision.

Results and documentation must distinguish among:

- measured results from this repository;
- derived or inferred results;
- values reported by the paper or another external source;
- unknown or unresolved claims.

Negative results should be preserved when they are part of the experimental
record.

## Repository artifacts

Do not commit generated or raw artifacts unless they are intentionally part of
the reviewed evidence set.

Check changes for caches, temporary logs, downloaded models or datasets,
checkpoints, large raw outputs, local environment files, and machine-specific
files.

Small structured evidence files under `results/` may be committed when they
are required for reproducibility and documented appropriately.

## Documentation consistency

At the end of a phase or substantial experiment, review:

```text
README.md
docs/
configs/
results/
tests/
requirements/
.github/workflows/
```

The README should describe the project's current public state and should not
retain stale TODOs that contradict completed work.

## Published history

Do not rewrite published history without explicit approval.

In particular:

- do not force-push published branches for cosmetic cleanup;
- do not rebase or squash published evidence solely to improve appearance;
- preserve referenced commit SHAs when practical;
- prefer forward fixes over rewriting already-public history.

When integrating stacked or phase branches, merge them in dependency order.

The priority is:

> preserve published evidence over cosmetic history cleanup

## Pull request review

Before merging, confirm that:

- the diff has one coherent purpose;
- required CI checks pass on the current pull request head;
- the commit history is understandable;
- documentation matches the implementation and evidence;
- no experimental claim is stronger than its supporting evidence;
- no published history has been rewritten unexpectedly.

A human review must be completed after the final substantive push and after
required CI passes.

If an independent reviewer is available, use a normal pull request review.
For single-maintainer work, leave a pull request comment recording the
self-review before merging.

The review record should confirm at least:

- the full diff was reviewed;
- commit scope and changed files were reviewed;
- no unrelated changes are included;
- experimental protocols, gates, and interpretation were not changed in
  response to observed results;
- relevant README, docs, configs, results, tests, requirements, and CI are
  consistent;
- required CI passed on the current pull request head.

Do not merge merely because the merge button is enabled.

`main` should represent the actual public state of the reproduction project.
