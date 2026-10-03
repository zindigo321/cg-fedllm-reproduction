"""Representations (item 5), Phi / Phi^-1 exact round trip (6), key ordering (7), geometry rejection (8)."""

from __future__ import annotations

import random

import pytest
import torch

from cg_fedllm.compression.layout import (
    LAYOUTS,
    QKVO,
    LayoutGeometryError,
    LoRAGeometry,
    get_layout,
    infer_geometry,
)
from cg_fedllm.compression.representation import recover_state, to_representation
from cg_fedllm.models.adapter import AdapterState

GEOMETRIES = {
    "tiny": (128, 2, 8, (1, 128, 128)),
    "llama-160m": (768, 12, 8, (1, 768, 768)),
    "llama-7b-r8": (4096, 32, 8, (1, 4096, 2048)),
    "llama-7b-r16": (4096, 32, 16, (1, 4096, 4096)),
    "qwen1.5-1.8b": (2048, 24, 8, (1, 2048, 1536)),
}


def make_state(d: int, layers: int, r: int, seed: int = 0, shuffle_insertion: bool = False) -> AdapterState:
    g = torch.Generator().manual_seed(seed)
    items = []
    for layer in range(layers):
        for m in QKVO:
            items.append((f"layers.{layer}.self_attn.{m}.lora_A.weight", torch.randn(r, d, generator=g)))
            items.append((f"layers.{layer}.self_attn.{m}.lora_B.weight", torch.randn(d, r, generator=g)))
    if shuffle_insertion:
        random.Random(seed).shuffle(items)
    return AdapterState(dict(items))


@pytest.mark.parametrize("name", list(GEOMETRIES))
def test_phi_roundtrip_is_bitwise_exact(name):
    d, layers, r, shape = GEOMETRIES[name]
    state = make_state(d, layers, r)
    for layout in LAYOUTS.values():
        geom = infer_geometry(state)
        assert geom == LoRAGeometry(layers, QKVO, r, d)
        x = layout.forward(state, geom)
        assert tuple(x.shape) == shape == geom.phi_shape
        assert x.numel() == geom.num_elements == 2 * 4 * layers * r * d
        back = layout.inverse(x, geom)
        assert back.equal(state)


def test_paper_llama7b_element_count():
    # 4096 x 8 x 2 x 4 x 32 = 8,388,608 (paper, Sect. 3.2.3)
    assert LoRAGeometry(32, QKVO, 8, 4096).num_elements == 8_388_608


def test_default_layout_block_order_and_independence_from_dict_order():
    s1 = make_state(64, 2, 4, seed=5)
    s2 = make_state(64, 2, 4, seed=5, shuffle_insertion=True)
    lay = get_layout("layer_major_qkvo_AtB")
    x1, x2 = lay.forward(s1), lay.forward(s2)
    assert torch.equal(x1, x2)
    # first block = A(layer0, q)^T, second = B(layer0, q), third = A(layer0, k)^T ...
    assert torch.equal(x1[0, :, 0:4], s1["layers.0.self_attn.q_proj.lora_A.weight"].t())
    assert torch.equal(x1[0, :, 4:8], s1["layers.0.self_attn.q_proj.lora_B.weight"])
    assert torch.equal(x1[0, :, 8:12], s1["layers.0.self_attn.k_proj.lora_A.weight"].t())
    assert torch.equal(x1[0, :, 32:36], s1["layers.1.self_attn.q_proj.lora_A.weight"].t())
    meta = lay.metadata(infer_geometry(s1))
    assert meta["key_order"][:3] == [
        "layers.0.self_attn.q_proj.lora_A.weight",
        "layers.0.self_attn.q_proj.lora_B.weight",
        "layers.0.self_attn.k_proj.lora_A.weight",
    ]
    alt = get_layout("module_major_qkvo_AtB")
    assert not torch.equal(alt.forward(s1), x1)
    assert alt.metadata(infer_geometry(s1))["key_order_sha256"] != meta["key_order_sha256"]


def test_incompatible_geometries_fail_loudly():
    lay = get_layout("layer_major_qkvo_AtB")
    gqa = make_state(128, 1, 4)
    gqa.tensors["layers.0.self_attn.k_proj.lora_B.weight"] = torch.randn(32, 4)  # GQA: out_features != d
    gqa.tensors["layers.0.self_attn.v_proj.lora_B.weight"] = torch.randn(32, 4)
    with pytest.raises(LayoutGeometryError, match="non-uniform projection"):
        lay.forward(gqa)
    missing = make_state(64, 2, 4)
    del missing.tensors["layers.1.self_attn.o_proj.lora_B.weight"]
    with pytest.raises(LayoutGeometryError, match="expected factors"):
        lay.forward(missing)
    rank = make_state(64, 1, 4)
    rank.tensors["layers.0.self_attn.q_proj.lora_A.weight"] = torch.randn(8, 64)
    with pytest.raises(LayoutGeometryError, match="rank"):
        lay.forward(rank)
    mlp = make_state(64, 1, 4)
    mlp.tensors["layers.0.mlp.up_proj.lora_A.weight"] = torch.randn(4, 64)
    with pytest.raises(LayoutGeometryError, match="only self_attn"):
        lay.forward(mlp)
    gap = AdapterState(
        {k.replace("layers.1.", "layers.2."): v for k, v in make_state(64, 2, 4).tensors.items()}
    )
    with pytest.raises(LayoutGeometryError, match="contiguous"):
        lay.forward(gap)
    ok = make_state(64, 2, 4)
    with pytest.raises(LayoutGeometryError, match="expects shape"):
        lay.inverse(torch.zeros(1, 64, 60), infer_geometry(ok))
    with pytest.raises(ValueError, match="unknown layout"):
        get_layout("row_major")


def test_state_and_delta_representations():
    start = make_state(64, 2, 4, seed=1)
    end = make_state(64, 2, 4, seed=2)
    rep_state = to_representation(end, start, "adapter_state")
    assert rep_state.equal(end)
    assert recover_state(rep_state, start, "adapter_state").equal(end)  # bitwise
    rep_delta = to_representation(end, start, "adapter_delta")
    for k in end.keys():
        assert torch.equal(rep_delta.tensors[k], end.tensors[k] - start.tensors[k])
    rec = recover_state(rep_delta, start, "adapter_delta")
    assert rec.relative_l2_diff(end) < 1e-6  # exact only up to float32 rounding (documented)
    with pytest.raises(ValueError):
        to_representation(end, start, "gradients")


def test_adapter_state_hash_save_load(tmp_path):
    s = make_state(32, 1, 2, seed=9)
    p = s.save(tmp_path / "a.safetensors")
    back = AdapterState.load(p)
    assert back.equal(s) and back.sha256() == s.sha256()
    assert make_state(32, 1, 2, seed=9, shuffle_insertion=True).sha256() == s.sha256()
