"""Analytic multiply-accumulate (MAC) counting for convolutional AutoEncoders.

Only Conv2d / ConvTranspose2d MACs are counted (normalisation, activations and padding are ignored).
Two conventions exist for transposed convolutions and both are reported:

* ``input_grid``  -- the true arithmetic count ``C_in * H_in * W_in * C_out * k_h * k_w / groups``;
* ``output_grid`` -- ``C_out * H_out * W_out * C_in * k_h * k_w / groups`` (what several FLOP counters
  report; the paper's decoder figure of 1.80 "G FLOPS" matches this convention).

Shapes are propagated with forward hooks on a copy of the module placed on the ``meta`` device, so even
LLaMA-7B-sized inputs (1 x 4096 x 2048) cost no memory or compute.
"""

from __future__ import annotations

import copy

import torch
from torch import nn


def count_macs(module: nn.Module, input_shape: tuple[int, ...]) -> dict[str, float]:
    meta_module = copy.deepcopy(module).to("meta")
    totals = {"conv": 0, "convtranspose_input_grid": 0, "convtranspose_output_grid": 0}

    def hook(mod: nn.Module, inputs, output) -> None:
        x = inputs[0]
        batch = x.shape[0]
        kh, kw = mod.kernel_size
        if isinstance(mod, nn.ConvTranspose2d):
            c_in, h_in, w_in = x.shape[1], x.shape[2], x.shape[3]
            c_out, h_out, w_out = output.shape[1], output.shape[2], output.shape[3]
            totals["convtranspose_input_grid"] += batch * c_in * h_in * w_in * c_out * kh * kw // mod.groups
            totals["convtranspose_output_grid"] += batch * c_out * h_out * w_out * c_in * kh * kw // mod.groups
        elif isinstance(mod, nn.Conv2d):
            c_in = x.shape[1]
            c_out, h_out, w_out = output.shape[1], output.shape[2], output.shape[3]
            totals["conv"] += batch * c_out * h_out * w_out * (c_in // mod.groups) * kh * kw

    handles = [m.register_forward_hook(hook) for m in meta_module.modules() if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d))]
    try:
        with torch.no_grad():
            meta_module.eval()
            meta_module(torch.empty(input_shape, device="meta"))
    finally:
        for h in handles:
            h.remove()
    return {
        "conv_macs": float(totals["conv"]),
        "macs_input_grid": float(totals["conv"] + totals["convtranspose_input_grid"]),
        "macs_output_grid": float(totals["conv"] + totals["convtranspose_output_grid"]),
    }
