"""Reconstructed ResNet-3 AutoEncoder (Phase-2 primary compressor, reviewer decision R5).

Independent implementation from the paper's appendix (Table "Auto-Encoders with ResNet" and the ResNet
AutoEncoder diagram of arXiv v3); no CycleGAN code is copied. The architecture is *structurally*
comparable to the CycleGAN ``ResnetGenerator`` pattern, which the paper's diagram resembles.

Encoder:  ReflectionPad(k//2) -> Conv(k x k, 1 -> stem_ch, no bias) -> BN -> ReLU
          6 x [Conv3x3 stride 2 pad 1 (no bias) -> BN -> ReLU], channels stem_ch -> 2 -> 4 -> ... -> 64
          3 x ResidualBlock(64)
Decoder:  3 x ResidualBlock(64)
          6 x [ConvTranspose3x3 stride 2 pad 1 output_pad 1 (no bias) -> BN -> ReLU], 64 -> ... -> stem_ch
          ReflectionPad(3) -> Conv7x7(stem_ch -> 1, bias) -> Tanh
ResidualBlock(c): x + [ReflectionPad(1), Conv3x3, BN, ReLU, ReflectionPad(1), Conv3x3, BN](x)

Evidence / confidence (see docs/evidence_ledger.md):
* 6 stride-2 stages, 64 latent channels, 3+3 residual blocks, Tanh head -- appendix table + diagram (HIGH);
* 3x3 stem (table) vs CycleGAN's 7x7 -- table says 3x3; the paper's FLOPs (0.81 G) fit 3x3 (MEDIUM);
* the table's spurious 7th "deconv1 4->32" row is dropped: the text says six deconvolutions (MEDIUM);
* conv bias disabled before BatchNorm, reflection padding inside residual blocks -- from the diagram's
  layer list and standard practice; parameter count then reproduces the paper's 0.94 MB (MEDIUM);
* a single-channel stem followed by ReLU discards the negative half-plane of its output -- this is a
  consequence of the reconstructed architecture, recorded as a Phase-3 question, not "fixed" here.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from cg_fedllm.utils.io import load_tensors, save_tensors_atomic


@dataclass(frozen=True)
class ResNetAEConfig:
    in_channels: int = 1
    stem_kernel: int = 3
    stem_channels: int = 1
    down_channels: tuple[int, ...] = (2, 4, 8, 16, 32, 64)
    num_res_blocks: int = 3
    final_kernel: int = 7

    @property
    def num_down(self) -> int:
        return len(self.down_channels)

    @property
    def latent_channels(self) -> int:
        return self.down_channels[-1]


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=False),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResNetEncoder(nn.Module):
    def __init__(self, cfg: ResNetAEConfig) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.ReflectionPad2d(cfg.stem_kernel // 2),
            nn.Conv2d(cfg.in_channels, cfg.stem_channels, kernel_size=cfg.stem_kernel, bias=False),
            nn.BatchNorm2d(cfg.stem_channels),
            nn.ReLU(inplace=False),
        ]
        chans = [cfg.stem_channels, *cfg.down_channels]
        for c_in, c_out in zip(chans[:-1], chans[1:]):
            layers += [nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=False)]
        layers += [ResidualBlock(cfg.latent_channels) for _ in range(cfg.num_res_blocks)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResNetDecoder(nn.Module):
    def __init__(self, cfg: ResNetAEConfig) -> None:
        super().__init__()
        layers: list[nn.Module] = [ResidualBlock(cfg.latent_channels) for _ in range(cfg.num_res_blocks)]
        chans = [cfg.stem_channels, *cfg.down_channels]
        for c_in, c_out in zip(reversed(chans[1:]), reversed(chans[:-1])):
            layers += [
                nn.ConvTranspose2d(c_in, c_out, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=False),
            ]
        layers += [
            nn.ReflectionPad2d(cfg.final_kernel // 2),
            nn.Conv2d(cfg.stem_channels, cfg.in_channels, kernel_size=cfg.final_kernel, bias=True),
            nn.Tanh(),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class ResNetAutoEncoder(nn.Module):
    def __init__(self, cfg: ResNetAEConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ResNetAEConfig()
        self.encoder = ResNetEncoder(self.cfg)
        self.decoder = ResNetDecoder(self.cfg)

    # ---- geometry --------------------------------------------------------------------------------------
    def check_input_hw(self, height: int, width: int) -> None:
        f = 2**self.cfg.num_down
        if height % f or width % f:
            raise ValueError(f"AE input {height}x{width} must be divisible by {f}")
        if height // f < 2 or width // f < 2:
            raise ValueError(f"AE latent grid {height // f}x{width // f} too small (reflection padding needs >= 2)")

    def latent_shape(self, height: int, width: int) -> tuple[int, int, int]:
        self.check_input_hw(height, width)
        f = 2**self.cfg.num_down
        return (self.cfg.latent_channels, height // f, width // f)

    def compression_ratio(self, height: int, width: int) -> float:
        c, h, w = self.latent_shape(height, width)
        return (c * h * w) / (self.cfg.in_channels * height * width)

    # ---- forward -------------------------------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        self.check_input_hw(x.shape[-2], x.shape[-1])
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z

    def num_parameters(self) -> dict[str, int]:
        enc = sum(p.numel() for p in self.encoder.parameters())
        dec = sum(p.numel() for p in self.decoder.parameters())
        return {"encoder": enc, "decoder": dec, "total": enc + dec}


def config_from_section(section: Any) -> ResNetAEConfig:
    return ResNetAEConfig(
        in_channels=1,
        stem_kernel=section.stem_kernel,
        stem_channels=section.stem_channels,
        down_channels=tuple(section.down_channels),
        num_res_blocks=section.num_res_blocks,
        final_kernel=section.final_kernel,
    )


def save_autoencoder(path: str | Path, ae: ResNetAutoEncoder, metadata: dict[str, Any]) -> Path:
    meta = {"format": "cg_fedllm.resnet_ae/v1", "config": json.dumps(asdict(ae.cfg)), "metadata": json.dumps(metadata, sort_keys=True)}
    return save_tensors_atomic(path, dict(ae.state_dict()), meta)


def load_autoencoder(path: str | Path, device: str | torch.device = "cpu") -> tuple[ResNetAutoEncoder, dict[str, Any]]:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as fh:
        meta = fh.metadata() or {}
    if meta.get("format") != "cg_fedllm.resnet_ae/v1":
        raise ValueError(f"{path}: not a ResNet AE checkpoint")
    raw = json.loads(meta["config"])
    raw["down_channels"] = tuple(raw["down_channels"])
    ae = ResNetAutoEncoder(ResNetAEConfig(**raw))
    ae.load_state_dict(load_tensors(path), strict=True)
    ae.to(device).eval()
    return ae, json.loads(meta.get("metadata", "{}"))
