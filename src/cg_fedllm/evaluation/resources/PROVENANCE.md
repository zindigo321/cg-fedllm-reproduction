# Evaluator resource provenance

| File | Source | Pinned revision | Upstream SHA-256 | License |
|---|---|---|---|---|
| `ceval_subject_mapping.json` | `hkust-nlp/ceval` `subject_mapping.json` (verbatim copy) | `cba65ae93bcf189149ced9f66ae0c958201faed9` | `671018e9d1ac8e51e8c3ea02574c89be8ff7660ebe67c8ec31b30b9035e064a7` | MIT |
| `mmlu_categories.json` | `hendrycks/test` `categories.py`, dictionaries transcribed to JSON (values unchanged) | `4450500f923c49f1fb1dd3d99108a0bd9717b660` | `e977fffa5356ab73c7da5e55003af48c2dcc6bf32155c787d0c0f42e76844f76` (of `categories.py`) | MIT |
| `ceval_hard_subjects.json` | C-Eval paper definition of C-Eval Hard (8 subjects) | arXiv:2305.08322 | n/a | n/a (list of names) |

MIT license texts for the two files above: `LICENSES-THIRD-PARTY.md`.

The benchmark *data* are never stored in this repository. They are downloaded at runtime from the
pinned Hugging Face revisions recorded in `configs/eval/*.yaml` (C-Eval: CC BY-NC-SA 4.0; MMLU: MIT).
