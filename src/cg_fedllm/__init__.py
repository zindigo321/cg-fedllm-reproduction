"""Independent, auditable reproduction of CG-FedLLM (Wu et al., ECAI 2025).

The package is organised by scientific component:

* :mod:`cg_fedllm.data` -- dataset provenance, Shepherd-compatible partitioning, D1/D2 splits.
* :mod:`cg_fedllm.models` -- pinned model loading and explicit LoRA attachment.
* :mod:`cg_fedllm.federated` -- local LoRA training, client sampling, aggregation, FL simulator.
* :mod:`cg_fedllm.compression` -- LoRA representations, the Phi layout, ResNet-3 AutoEncoder, codecs.
* :mod:`cg_fedllm.tgap` -- TGAP snapshot collection and AutoEncoder training.
* :mod:`cg_fedllm.evaluation` -- ``reference_eval_v1`` (C-Eval / MMLU) and held-out loss.
"""

__version__ = "0.2.0.dev0"
