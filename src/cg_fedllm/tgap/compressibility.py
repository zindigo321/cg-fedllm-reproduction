"""Reference codes at the AE's element ratio (Phase 3A interpretation aid; DERIVED, never a gate).

How much of a snapshot can simple codes keep when they transmit as many numbers as the AE latent (1/64 of the
Phi elements), and what does that imply for the innovation? Each reconstruction goes through the same factor-aware
report as the AutoEncoder (:mod:`cg_fedllm.compression.diagnostics`), so the numbers are directly comparable.

* ``lowpass_dct``  -- fixed linear code: the lowest-frequency (H/8 x W/8) block of the orthonormal 2-D DCT-II of
                      the Phi image (exactly 1/64 of the coefficients, no side information).
* ``topk_dct``     -- adaptive: the N/64 largest-magnitude DCT coefficients (index cost ignored, so optimistic).
* ``topk_raw``     -- adaptive: the N/64 largest-magnitude Phi elements (index cost ignored, so optimistic).
* ``train_pca``    -- a decoder that memorises the training split: ``mean + projection onto the span of the
                      centred training snapshots`` (at most n_train - 1 numbers per snapshot). This is far below
                      1/64, but the decoder stores n_train full snapshots.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from scipy.fft import dctn, idctn

from cg_fedllm.compression.diagnostics import evaluate_predictors
from cg_fedllm.compression.layout import Layout, LoRAGeometry
from cg_fedllm.compression.representation import to_representation
from cg_fedllm.tgap.snapshots import load_states
from cg_fedllm.tgap.train_ae import split_indices
from cg_fedllm.tgap.viability import error_stride, iter_groups

RATIO = 64


def lowpass_dct(x: np.ndarray, ratio: int = RATIO) -> np.ndarray:
    side = int(round(ratio**0.5))
    h, w = x.shape[0] // side, x.shape[1] // side
    c = dctn(x, type=2, norm="ortho")
    kept = np.zeros_like(c)
    kept[:h, :w] = c[:h, :w]
    return idctn(kept, type=2, norm="ortho")


def _topk(c: np.ndarray, k: int) -> np.ndarray:
    flat = c.reshape(-1)
    idx = np.argpartition(np.abs(flat), flat.size - k)[flat.size - k :]
    out = np.zeros_like(flat)
    out[idx] = flat[idx]
    return out.reshape(c.shape)


def topk_dct(x: np.ndarray, ratio: int = RATIO) -> np.ndarray:
    return idctn(_topk(dctn(x, type=2, norm="ortho"), x.size // ratio), type=2, norm="ortho")


def topk_raw(x: np.ndarray, ratio: int = RATIO) -> np.ndarray:
    return _topk(x, x.size // ratio)


class TrainPCA:
    """``x -> mean + P (x - mean)``, P = orthogonal projection onto the span of the centred training rows."""

    def __init__(self, train: Sequence[torch.Tensor], rel_eig_floor: float = 1e-10) -> None:
        rows = torch.stack([t.reshape(-1).double() for t in train])
        self.mean = rows.mean(0)
        self.basis_rows = rows - self.mean  # [n, N]
        gram = self.basis_rows @ self.basis_rows.T
        evals, evecs = torch.linalg.eigh(gram)
        keep = evals > rel_eig_floor * float(evals.max())
        self.evecs, self.evals = evecs[:, keep], evals[keep]
        self.rank = int(keep.sum())

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        d = x.reshape(-1).double() - self.mean
        coef = (self.evecs.T @ (self.basis_rows @ d)) / self.evals  # coordinates in the eigenbasis of X X^T
        return (self.mean + self.basis_rows.T @ (self.evecs @ coef)).reshape(x.shape).to(torch.float32)


def reference_code_report(
    snapshot_dir,
    records: Sequence[dict],
    *,
    representation: str,
    layout: Layout,
    geom: LoRAGeometry,
    split: str = "temporal",
    val_fraction: float = 0.2,
    split_seed: int = 0,
) -> dict[str, Any]:
    train_idx, val_idx = split_indices(records, split, val_fraction, split_seed)
    train = []
    for i in train_idx:
        start, end = load_states(snapshot_dir, records[i])
        train.append(layout.forward(to_representation(end, start, representation), geom))
    pca = TrainPCA(train)
    del train

    def np_code(fn):
        return lambda x, it: torch.from_numpy(fn(x[0].double().numpy())).unsqueeze(0).to(torch.float32)

    predictors = {
        "lowpass_dct": np_code(lowpass_dct),
        "topk_dct": np_code(topk_dct),
        "topk_raw": np_code(topk_raw),
        "train_pca": lambda x, it: pca(x),
    }
    rep = evaluate_predictors(
        iter_groups(snapshot_dir, records, val_idx),
        representation,
        layout,
        geom,
        predictors,
        error_stride=error_stride(len(val_idx), geom),
    )
    keep = ("transmitted", "innovation", "aggregate", "product", "innovation_energy_fraction")
    return {
        "label": "DERIVED",
        "representation": representation,
        "split": "validation",
        "num_snapshots": len(val_idx),
        "element_ratio": f"1/{RATIO} (train_pca: <= {pca.rank} numbers per snapshot, decoder stores the training snapshots)",
        "train_pca_rank": pca.rank,
        "codes": {name: {k: v for k, v in r.items() if k in keep} for name, r in rep.items()},
    }
