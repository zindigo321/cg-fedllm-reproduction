"""Phase-3 A2: frozen, train-split-only AE input normalisation (no leakage, zero payload, exact inversion)."""

from __future__ import annotations

import math

import pytest
import torch

from cg_fedllm.compression.autoencoder import ResNetAutoEncoder
from cg_fedllm.compression.codecs import AutoEncoderCodec, CodecContext
from cg_fedllm.compression.layout import LoRAGeometry, get_layout
from cg_fedllm.compression.normalization import Normalizer, factor_column_mask, fit_normalizer, range_report
from cg_fedllm.models.adapter import AdapterState

GEOM = LoRAGeometry(num_layers=2, modules=("q_proj", "k_proj", "v_proj", "o_proj"), rank=8, hidden=128)


def _state(seed: int, a_scale: float = 1e-2, b_scale: float = 1e-4) -> AdapterState:
    g = torch.Generator().manual_seed(seed)
    t = {}
    for layer in range(GEOM.num_layers):
        for m in GEOM.modules:
            t[f"layers.{layer}.self_attn.{m}.lora_A.weight"] = torch.randn(8, 128, generator=g) * a_scale
            t[f"layers.{layer}.self_attn.{m}.lora_B.weight"] = torch.randn(128, 8, generator=g) * b_scale
    return AdapterState(t)


@pytest.mark.parametrize("layout_id", ["layer_major_qkvo_AtB", "module_major_qkvo_AtB"])
def test_factor_mask_matches_the_layout(layout_id):
    lay = get_layout(layout_id)
    st = _state(0, a_scale=1.0, b_scale=0.0)  # A = noise, B = exactly 0
    x = lay.forward(st, GEOM)
    mask = factor_column_mask(lay, GEOM)
    assert mask.sum() == mask.numel() // 2
    assert torch.all(x[..., ~mask] == 0) and torch.all(x[..., mask] != 0)


def test_fit_uses_only_the_training_split_and_is_frozen():
    lay = get_layout("layer_major_qkvo_AtB")
    xs = [lay.forward(_state(i), GEOM) for i in range(6)]
    train = [0, 1, 2, 3]
    for mode in ("global_rms", "factor_rms"):
        n1 = fit_normalizer(xs, train, mode, layout=lay, geom=GEOM)
        leaked = list(xs)
        leaked[4], leaked[5] = xs[4] * 1000.0, xs[5] * -50.0  # validation snapshots changed drastically
        n2 = fit_normalizer(leaked, train, mode, layout=lay, geom=GEOM)
        assert n1 == n2, mode  # validation data never enters the statistics
        assert n1.fit["train_indices"] == train
    allx = torch.cat([xs[i].reshape(-1) for i in train]).double()
    assert math.isclose(
        fit_normalizer(xs, train, "global_rms").scale_global, float(allx.pow(2).mean().sqrt()), rel_tol=1e-12
    )
    nf = fit_normalizer(xs, train, "factor_rms", layout=lay, geom=GEOM)
    mask = factor_column_mask(lay, GEOM)
    a = torch.cat([xs[i][..., mask].reshape(-1) for i in train]).double()
    b = torch.cat([xs[i][..., ~mask].reshape(-1) for i in train]).double()
    assert math.isclose(nf.scale_A, float(a.pow(2).mean().sqrt()), rel_tol=1e-12)
    assert math.isclose(nf.scale_B, float(b.pow(2).mean().sqrt()), rel_tol=1e-12)
    assert 50 < nf.scale_A / nf.scale_B < 200  # the 100x A/B scale gap is captured
    # after factor_rms both factors have unit RMS on the training split
    y = torch.cat([nf.normalize(xs[i]) for i in train], dim=0)
    assert math.isclose(float(y[..., mask].pow(2).mean().sqrt()), 1.0, rel_tol=1e-5)
    assert math.isclose(float(y[..., ~mask].pow(2).mean().sqrt()), 1.0, rel_tol=1e-5)


def test_normalize_denormalize_round_trip_and_serialisation():
    lay = get_layout("layer_major_qkvo_AtB")
    xs = [lay.forward(_state(i), GEOM) for i in range(3)]
    for mode in ("none", "global_rms", "factor_rms"):
        n = fit_normalizer(xs, [0, 1], mode, layout=lay, geom=GEOM)
        assert torch.allclose(n.denormalize(n.normalize(xs[2])), xs[2], rtol=1e-6, atol=0)
        assert Normalizer.from_dict(n.to_dict()) == n
    assert fit_normalizer(xs, [0], "none").normalize(xs[0]) is xs[0]  # paper-literal: no arithmetic at all
    assert Normalizer.from_dict(None).is_identity


def test_invalid_scales_and_missing_layout_are_rejected():
    with pytest.raises(ValueError):
        Normalizer("global_rms", scale_global=0.0)
    with pytest.raises(ValueError):
        Normalizer(
            "factor_rms",
            scale_A=1.0,
            scale_B=float("nan"),
            layout_id="layer_major_qkvo_AtB",
            geometry=GEOM.to_dict(),
        )
    lay = get_layout("layer_major_qkvo_AtB")
    zero_b = [lay.forward(_state(i, b_scale=0.0), GEOM) for i in range(2)]
    with pytest.raises(ValueError):  # B identically zero on the training split -> no finite scale
        fit_normalizer(zero_b, [0, 1], "factor_rms", layout=lay, geom=GEOM)
    with pytest.raises(ValueError):
        fit_normalizer(zero_b, [0], "factor_rms")
    with pytest.raises(ValueError):
        fit_normalizer(zero_b, [], "global_rms")


def test_normalisation_adds_zero_uplink_bytes_and_is_inverted_after_decoding():
    lay = get_layout("layer_major_qkvo_AtB")
    xs = [lay.forward(_state(i), GEOM) for i in range(3)]
    torch.manual_seed(0)
    ae = ResNetAutoEncoder()
    ctx = CodecContext(0, 0, 0)
    plain = AutoEncoderCodec(ae, normalizer=None)
    for mode in ("global_rms", "factor_rms"):
        n = fit_normalizer(xs, [0, 1], mode, layout=lay, geom=GEOM)
        c = AutoEncoderCodec(ae, normalizer=n)
        p = c.encode(xs[2], ctx)
        assert p.logical_nbytes() == plain.encode(xs[2], ctx).logical_nbytes() == 128 * 128 // 64 * 4
        assert set(p.tensors) == {"z"}  # nothing but the latent is transmitted
        # decode = denormalise(decoder(z)): the decoder output is in (-1, 1), so the result is bounded by the scale
        y = c.decode(p, ctx)
        bound = n.scale_global if mode == "global_rms" else max(n.scale_A, n.scale_B)
        assert float(y.abs().max()) <= bound * (1 + 1e-6)
        assert c.describe()["normalization"]["mode"] == mode
    # mode none behaves exactly like the Phase-2 codec
    none = AutoEncoderCodec(ae, normalizer=fit_normalizer(xs, [0, 1], "none"))
    assert torch.equal(none.decode(none.encode(xs[2], ctx), ctx), plain.decode(plain.encode(xs[2], ctx), ctx))


def test_range_report_measures_tanh_overflow_without_clipping():
    lay = get_layout("layer_major_qkvo_AtB")
    xs = [lay.forward(_state(i), GEOM) for i in range(4)]
    n = fit_normalizer(xs, [0, 1, 2], "factor_rms", layout=lay, geom=GEOM)
    r = range_report(xs, [3], n, lay, GEOM)
    # standard-normal entries at unit RMS: ~31.7 % have |v| >= 1 (they cannot be produced by a Tanh output)
    assert 0.25 < r["A"]["fraction_abs_ge_1"] < 0.40 and 0.25 < r["B"]["fraction_abs_ge_1"] < 0.40
    assert r["A"]["max"] > 2.0 and r["A"]["exact"] is True
    r0 = range_report(xs, [3], fit_normalizer(xs, [0, 1, 2], "none"), lay, GEOM)
    assert r0["all"]["fraction_abs_ge_1"] == 0.0  # raw LoRA values are far inside (-1, 1)
