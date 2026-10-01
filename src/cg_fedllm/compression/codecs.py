"""Update codecs: client-side ``encode`` and server-side ``decode`` of the Phi-stacked tensor.

Every codec operates on ``X in R^{1 x d x W}`` produced by the configured Phi layout. ``encode`` returns
a :class:`Payload` (the tensors that would be transmitted); ``decode`` reconstructs ``X_hat``. The
federated simulator calls the same code path for every codec (reviewer decision R16):

* :class:`IdentityCodec`      -- transmits X unchanged (lossless control).
* :class:`AutoEncoderCodec`   -- CG-FedLLM: client encoder -> latent; server decoder -> X_hat.
* :class:`ConstantMeanCodec`  -- transmits nothing; the server "decodes" a fixed tensor (e.g. the TGAP
                                 training mean). Control for an input-independent decoder.
* :class:`GaussianNoiseCodec` -- transmits X + N(0, sigma^2) (seeded per round/client). Control for
                                 "compression acts as noise".

Logical payload size = sum over transmitted tensors of ``numel * element_size`` (metadata excluded);
the safetensors-serialised size is reported separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from safetensors.torch import save as safetensors_save

from cg_fedllm.utils.seeding import torch_generator

if TYPE_CHECKING:  # the codec only needs the AE's encode/decode methods at runtime
    from cg_fedllm.compression.autoencoder import ResNetAutoEncoder

LATENT_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


@dataclass(frozen=True)
class CodecContext:
    round_index: int
    client_id: int
    seed: int


@dataclass
class Payload:
    tensors: dict[str, torch.Tensor]
    meta: dict[str, Any] = field(default_factory=dict)

    def logical_nbytes(self) -> int:
        return int(sum(t.numel() * t.element_size() for t in self.tensors.values()))

    def numel(self) -> int:
        return int(sum(t.numel() for t in self.tensors.values()))

    def serialized_nbytes(self) -> int:
        if not self.tensors:
            return 0
        return len(safetensors_save({k: v.contiguous() for k, v in self.tensors.items()}))


class Codec:
    codec_id = "abstract"

    def encode(self, x: torch.Tensor, ctx: CodecContext) -> Payload:  # client side
        raise NotImplementedError

    def decode(self, payload: Payload, ctx: CodecContext) -> torch.Tensor:  # server side
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {"codec_id": self.codec_id}


class IdentityCodec(Codec):
    codec_id = "identity"

    def encode(self, x: torch.Tensor, ctx: CodecContext) -> Payload:
        return Payload({"x": x.detach().to("cpu").clone()})

    def decode(self, payload: Payload, ctx: CodecContext) -> torch.Tensor:
        return payload.tensors["x"].clone()


class AutoEncoderCodec(Codec):
    codec_id = "autoencoder"

    def __init__(self, ae: ResNetAutoEncoder, device: str | torch.device = "cpu", latent_dtype: str = "float32", info: dict | None = None):
        self.ae = ae.to(device).eval()
        self.device = torch.device(device)
        self.latent_dtype = LATENT_DTYPES[latent_dtype]
        self.info = info or {}

    @torch.no_grad()
    def encode(self, x: torch.Tensor, ctx: CodecContext) -> Payload:
        z = self.ae.encode(x.to(self.device, torch.float32).unsqueeze(0))  # [1, 1, d, W] -> [1, C, h, w]
        return Payload({"z": z[0].to("cpu", self.latent_dtype).contiguous()}, {"input_shape": list(x.shape)})

    @torch.no_grad()
    def decode(self, payload: Payload, ctx: CodecContext) -> torch.Tensor:
        z = payload.tensors["z"].to(self.device, torch.float32).unsqueeze(0)
        return self.ae.decode(z)[0].to("cpu", torch.float32)

    def describe(self) -> dict[str, Any]:
        return {"codec_id": self.codec_id, "latent_dtype": str(self.latent_dtype), **self.info}


class ConstantMeanCodec(Codec):
    codec_id = "constant_mean"

    def __init__(self, mean: torch.Tensor, info: dict | None = None):
        self.mean = mean.detach().to("cpu", torch.float32).clone()
        self.info = info or {}

    def encode(self, x: torch.Tensor, ctx: CodecContext) -> Payload:
        if tuple(x.shape) != tuple(self.mean.shape):
            raise ValueError(f"constant mean shape {tuple(self.mean.shape)} != input {tuple(x.shape)}")
        return Payload({}, {"input_shape": list(x.shape)})

    def decode(self, payload: Payload, ctx: CodecContext) -> torch.Tensor:
        return self.mean.clone()

    def describe(self) -> dict[str, Any]:
        return {"codec_id": self.codec_id, **self.info}


class GaussianNoiseCodec(Codec):
    codec_id = "gaussian_noise"

    def __init__(self, sigma: float):
        if sigma < 0:
            raise ValueError("sigma must be >= 0")
        self.sigma = float(sigma)

    def encode(self, x: torch.Tensor, ctx: CodecContext) -> Payload:
        g = torch_generator(ctx.seed, "gaussian_noise", ctx.round_index, ctx.client_id)
        noise = torch.randn(x.shape, generator=g, dtype=torch.float32) * self.sigma
        return Payload({"x": (x.detach().to("cpu", torch.float32) + noise).contiguous()}, {"sigma": self.sigma})

    def decode(self, payload: Payload, ctx: CodecContext) -> torch.Tensor:
        return payload.tensors["x"].clone()

    def describe(self) -> dict[str, Any]:
        return {"codec_id": self.codec_id, "sigma": self.sigma}
