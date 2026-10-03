"""Reference codes at the AE's element ratio (DERIVED interpretation aid): sanity of each code on data whose
compressibility is known analytically."""

from __future__ import annotations

import math

import numpy as np
import torch

from cg_fedllm.tgap.compressibility import TrainPCA, lowpass_dct, topk_dct, topk_raw


def _rse(x, x_hat):
    return float(((x_hat - x) ** 2).sum() / (x**2).sum())


def test_white_noise_is_incompressible_at_one_sixty_fourth():
    x = np.random.default_rng(0).standard_normal((256, 192))
    assert (
        abs(_rse(x, lowpass_dct(x)) - 63 / 64) < 0.02
    )  # a fixed linear 1/64 code keeps ~1/64 of i.i.d. energy
    # keeping the largest 1/64 of the values of a Gaussian keeps ~12 % of its energy (two-sided tail at 2.42 sigma)
    assert 0.85 < _rse(x, topk_raw(x)) < 0.91
    assert 0.85 < _rse(x, topk_dct(x)) < 0.91  # the orthonormal DCT of white noise is white noise
    assert np.count_nonzero(topk_raw(x)) == x.size // 64


def test_smooth_images_are_compressible():
    u, v = np.meshgrid(np.linspace(0, 1, 192), np.linspace(0, 1, 256))
    x = np.sin(2 * np.pi * u) * np.cos(3 * np.pi * v) + u * v
    assert _rse(x, lowpass_dct(x)) < 1e-3 and _rse(x, topk_dct(x)) < 1e-3


def test_train_pca_reconstructs_its_span_and_nothing_else():
    g = torch.Generator().manual_seed(0)
    basis = torch.randn(3, 1, 32, 24, generator=g)
    train = [basis[0] + (i % 3) * basis[1] - (i % 2) * basis[2] for i in range(6)]
    pca = TrainPCA(train)
    assert pca.rank == 2
    inside = basis[0] + 0.5 * basis[1] + 2.0 * basis[2]
    assert torch.allclose(pca(inside), inside, atol=1e-4)
    # a probe orthogonal to the span of the centred training snapshots is mapped to the training mean
    mean = torch.stack(train).mean(0)
    centred = torch.stack([t - mean for t in train]).reshape(6, -1).double()
    q, _ = torch.linalg.qr(centred.T)
    o = torch.randn(32 * 24, generator=g, dtype=torch.float64)
    o = o - q @ (q.T @ o)
    probe = mean + o.reshape(1, 32, 24).float()
    assert math.isclose(float((pca(probe) - mean).norm()), 0.0, abs_tol=1e-3)
