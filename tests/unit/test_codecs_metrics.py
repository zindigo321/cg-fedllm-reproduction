"""Codec determinism (13), logical payload byte accounting (14), reconstruction metrics."""

from __future__ import annotations

import math

import torch

from cg_fedllm.compression.codecs import (
    CodecContext,
    ConstantMeanCodec,
    GaussianNoiseCodec,
    IdentityCodec,
)
from cg_fedllm.compression.metrics import compression_ratio, reconstruction_metrics

X = torch.randn(1, 128, 128, generator=torch.Generator().manual_seed(0)) * 1e-2
CTX = CodecContext(round_index=3, client_id=7, seed=11)


def test_identity_codec_is_lossless_and_accounts_raw_bytes():
    c = IdentityCodec()
    p = c.encode(X, CTX)
    assert torch.equal(c.decode(p, CTX), X)
    assert p.logical_nbytes() == X.numel() * 4 == 128 * 128 * 4
    assert p.serialized_nbytes() >= p.logical_nbytes()


def test_constant_mean_codec_is_input_independent_and_sends_nothing():
    mean = torch.full_like(X, 0.5)
    c = ConstantMeanCodec(mean)
    p = c.encode(X, CTX)
    assert p.logical_nbytes() == 0 and p.serialized_nbytes() == 0
    assert torch.equal(c.decode(p, CTX), mean)
    assert torch.equal(c.decode(c.encode(torch.zeros_like(X), CTX), CTX), mean)


def test_gaussian_noise_codec_is_seeded_per_round_and_client():
    c = GaussianNoiseCodec(0.1)
    a = c.decode(c.encode(X, CTX), CTX)
    b = c.decode(c.encode(X, CTX), CTX)
    other = c.decode(c.encode(X, CodecContext(3, 8, 11)), CTX)
    assert torch.equal(a, b) and not torch.equal(a, other)
    assert abs(float((a - X).std()) - 0.1) < 0.01
    assert c.encode(X, CTX).logical_nbytes() == X.numel() * 4
    assert torch.equal(GaussianNoiseCodec(0.0).decode(GaussianNoiseCodec(0.0).encode(X, CTX), CTX), X)


def test_snr_definitions_and_paper_arithmetic():
    x = torch.ones(10, 10, dtype=torch.float64)
    x_hat = x + 0.1
    m = reconstruction_metrics(x, x_hat)
    assert math.isclose(m["mse"], 0.01, rel_tol=1e-9)
    assert math.isclose(m["snr_standard"], 100.0, rel_tol=1e-9)
    assert math.isclose(m["snr_paper"], m["snr_standard"] * 100, rel_tol=1e-9)  # paper SNR = n * standard SNR
    assert math.isclose(m["snr_db"], 20.0, rel_tol=1e-9)
    # the paper's Table 1 arithmetic: SNR = 14.29 / e with e the per-element MSE
    assert math.isclose(14.29 / 5.06e-12, 2.82e12, rel_tol=5e-3)
    # Gaussian noise of std sigma gives MSE ~= sigma^2 (the denoising table's LoRA-FT column)
    g = torch.Generator().manual_seed(1)
    z = torch.zeros(1, 512, 512)
    assert abs(reconstruction_metrics(z + 1.0, z + 1.0 + 5e-3 * torch.randn(z.shape, generator=g))["mse"] / (5e-3) ** 2 - 1) < 0.02


def test_innovation_ratio_and_cosine():
    ref = torch.zeros(4, 4)
    x = torch.ones(4, 4)
    m = reconstruction_metrics(x, ref.clone(), reference=ref)  # predicting "no change"
    assert math.isclose(m["innovation_ratio"], 1.0)
    m2 = reconstruction_metrics(x, x * 0.5, reference=ref)
    assert m2["innovation_ratio"] < 1.0 and math.isclose(m2["cosine"], 1.0, rel_tol=1e-9)
    assert compression_ratio(8_388_608, 131_072) == 1 / 64
