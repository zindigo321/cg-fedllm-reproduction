"""ResNet-3 AutoEncoder: shapes, latent, compression ratio, parameter count, MACs, gradients (items 9-12)."""

from __future__ import annotations

import pytest
import torch

from cg_fedllm.compression.autoencoder import ResNetAutoEncoder, load_autoencoder, save_autoencoder
from cg_fedllm.compression.codecs import AutoEncoderCodec, CodecContext
from cg_fedllm.compression.macs import count_macs


@pytest.fixture(scope="module")
def ae():
    torch.manual_seed(0)
    return ResNetAutoEncoder()


@pytest.mark.parametrize(
    "hw,latent",
    [
        ((128, 128), (64, 2, 2)),
        ((768, 768), (64, 12, 12)),
        ((2048, 1536), (64, 32, 24)),
        ((4096, 2048), (64, 64, 32)),
        ((4096, 4096), (64, 64, 64)),
    ],
)
def test_latent_shape_and_exact_compression_ratio(ae, hw, latent):
    assert ae.latent_shape(*hw) == latent
    assert ae.compression_ratio(*hw) == 1 / 64  # 1.5625 %, paper Table 1 (ResNet)


def test_paper_llama7b_geometry_on_meta_device():
    fresh = ResNetAutoEncoder()
    enc = fresh.encoder.to("meta")
    dec = fresh.decoder.to("meta")
    z = enc(torch.empty(1, 1, 4096, 2048, device="meta"))
    assert tuple(z.shape) == (1, 64, 64, 32)  # paper Table 1: Enc(G) = [64, 64, 32]
    assert tuple(dec(z).shape) == (1, 1, 4096, 2048)


def test_parameter_count_matches_phase1_reconstruction():
    p = ResNetAutoEncoder().num_parameters()
    assert p == {"encoder": 246_785, "decoder": 246_698, "total": 493_483}
    # fp32 bytes per half ~= 0.94 MiB (paper Table 3: 0.94 MB GPU memory for Enc and Dec)
    assert abs(p["encoder"] * 4 / 2**20 - 0.94) < 0.005 and abs(p["decoder"] * 4 / 2**20 - 0.94) < 0.005


def test_mac_counts_and_paper_convention():
    ae = ResNetAutoEncoder()
    enc = count_macs(ae.encoder, (1, 1, 4096, 2048))
    dec = count_macs(ae.decoder, (1, 64, 64, 32))
    assert enc["macs_input_grid"] == 754_974_720
    assert dec["macs_input_grid"] == 1_090_519_040
    assert dec["macs_output_grid"] == 1_769_996_288
    # paper Table 3: Enc 0.81 G, Dec 1.80 G "FLOPS" -> consistent with MACs, output-grid convention for ConvT
    assert abs(enc["macs_output_grid"] / 0.81e9 - 1) < 0.10
    assert abs(dec["macs_output_grid"] / 1.80e9 - 1) < 0.05


def test_finite_forward_backward_and_eval_determinism():
    torch.manual_seed(1)
    ae = ResNetAutoEncoder()
    x = torch.randn(2, 1, 128, 128) * 1e-2
    x_hat, z = ae(x)
    assert tuple(x_hat.shape) == tuple(x.shape) and tuple(z.shape) == (2, 64, 2, 2)
    loss = torch.nn.functional.mse_loss(x_hat, x)
    loss.backward()
    grads = [p.grad for p in ae.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    ae.eval()
    with torch.no_grad():
        a, b = ae(x)[0], ae(x)[0]
    assert torch.equal(a, b)
    assert float(x_hat.detach().abs().max()) < 1.0  # Tanh head


def test_input_geometry_is_validated():
    ae = ResNetAutoEncoder()
    with pytest.raises(ValueError, match="divisible"):
        ae.latent_shape(100, 128)
    with pytest.raises(ValueError, match="too small"):
        ae.latent_shape(64, 64)


def test_checkpoint_round_trip(tmp_path):
    torch.manual_seed(2)
    ae = ResNetAutoEncoder().eval()
    path = save_autoencoder(tmp_path / "ae.safetensors", ae, {"note": "t"})
    back, meta = load_autoencoder(path)
    x = torch.randn(1, 1, 128, 128) * 1e-2
    with torch.no_grad():
        assert torch.equal(ae(x)[0], back(x)[0])
    assert meta["note"] == "t"


X_AE = torch.randn(1, 128, 128, generator=torch.Generator().manual_seed(0)) * 1e-2
CTX_AE = CodecContext(round_index=3, client_id=7, seed=11)


def test_autoencoder_codec_determinism_and_latent_bytes():
    torch.manual_seed(0)
    ae = ResNetAutoEncoder().eval()
    c32 = AutoEncoderCodec(ae, latent_dtype="float32")
    p1, p2 = c32.encode(X_AE, CTX_AE), c32.encode(X_AE, CTX_AE)
    assert torch.equal(p1.tensors["z"], p2.tensors["z"])
    assert torch.equal(c32.decode(p1, CTX_AE), c32.decode(p2, CTX_AE))
    assert p1.numel() == X_AE.numel() // 64  # CR = 1/64
    assert p1.logical_nbytes() == X_AE.numel() // 64 * 4
    c16 = AutoEncoderCodec(ae, latent_dtype="float16")
    assert c16.encode(X_AE, CTX_AE).logical_nbytes() == X_AE.numel() // 64 * 2
    assert tuple(c32.decode(p1, CTX_AE).shape) == tuple(X_AE.shape)
