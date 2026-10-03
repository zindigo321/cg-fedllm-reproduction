# Claude Code project instructions

Read and follow `CONTRIBUTING.md` before making any change. It is the
repository's source of truth for development, experiment, review, and Git
workflow rules.

In particular:

- keep one logical change per commit;
- do not make unrelated or opportunistic changes;
- do not use `git add .` by default;
- inspect `git status`, `git diff`, `git diff --stat`, and `git diff --check`;
- run the required Ruff and pytest gates before proposing a commit;
- do not commit, push, merge, or rewrite published history without explicit
  human approval;
- do not change an experimental protocol, metric, gate, representation choice,
  or selection rule after observing results;
- preserve negative results and published experiment evidence;
- distinguish measured, derived/inferred, paper-reported, and unknown claims;
- do not add generated, raw, temporary, or machine-specific artifacts unless
  they are explicitly approved as part of the evidence set;
- keep README, docs, configs, results, tests, requirements, and CI consistent
  where relevant.

AI-generated changes are proposals. Stop at a human review boundary before
commit or merge.
