from __future__ import annotations

import math
from functools import partial
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


DEFAULT_MAE_CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth"
)


def _build_2d_sincos_position_embedding(
    height: int,
    width: int,
    channels: int,
) -> Tensor:
    """Return CounTR/MAE-style fixed positions with shape ``[1,HW,C]``."""

    if channels % 4 != 0:
        raise ValueError("ViT embedding dimension must be divisible by four")
    quarter = channels // 4
    omega = torch.arange(quarter, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / max(quarter - 1, 1)))
    y = torch.arange(height, dtype=torch.float32)
    x = torch.arange(width, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    phase_x = grid_x[..., None] * omega
    phase_y = grid_y[..., None] * omega
    position = torch.cat(
        (phase_x.sin(), phase_x.cos(), phase_y.sin(), phase_y.cos()), dim=-1
    )
    return position.reshape(1, height * width, channels)


def _unwrap_checkpoint(checkpoint: object) -> Mapping[str, Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("MAE checkpoint must contain a state-dict mapping")
    for key in ("model", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            checkpoint = value
            break
    return {
        str(key): value
        for key, value in checkpoint.items()
        if isinstance(value, Tensor)
    }


def _strip_encoder_prefix(key: str) -> str:
    for prefix in ("module.", "model."):
        if key.startswith(prefix):
            key = key[len(prefix) :]
    for prefix in ("backbone.", "encoder."):
        if key.startswith(prefix):
            key = key[len(prefix) :]
    return key


class CounTRMAEViTBackbone(nn.Module):
    """Vanilla ViT encoder used by CounTR, with optional MAE initialization."""

    def __init__(
        self,
        *,
        img_size: int = 384,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        pretrained: bool = True,
        checkpoint: str | None = None,
    ) -> None:
        super().__init__()
        try:
            from timm.models.vision_transformer import Block, PatchEmbed
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise ImportError(
                "v6 requires timm. Install the pinned requirements.txt environment."
            ) from exc

        if img_size % patch_size:
            raise ValueError("CounTR ViT image size must be divisible by patch size")
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=embed_dim,
            strict_img_size=False,
            dynamic_img_pad=False,
        )
        self.pos_embed = nn.Parameter(
            _build_2d_sincos_position_embedding(*self.grid_size, embed_dim),
            requires_grad=False,
        )
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for _ in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self._initialize_weights()
        if pretrained:
            self.load_mae_pretrained(checkpoint or DEFAULT_MAE_CHECKPOINT_URL)

    def _initialize_weights(self) -> None:
        weight = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(weight.view(weight.shape[0], -1))
        self.apply(self._initialize_module)

    @staticmethod
    def _initialize_module(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def load_mae_pretrained(self, checkpoint: str) -> None:
        """Load an official MAE, CounTR, or previously wrapped encoder checkpoint."""

        if checkpoint.startswith(("http://", "https://")):
            payload = torch.hub.load_state_dict_from_url(
                checkpoint,
                map_location="cpu",
                check_hash=False,
            )
        else:
            path = Path(checkpoint).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"MAE checkpoint not found: {path}")
            payload = torch.load(path, map_location="cpu", weights_only=False)

        source = _unwrap_checkpoint(payload)
        current = self.state_dict()
        compatible: dict[str, Tensor] = {}
        for source_key, value in source.items():
            key = _strip_encoder_prefix(source_key)
            if key == "pos_embed":
                # CounTR uses a fixed 24x24 sine/cosine grid at 384x384.  An
                # ImageNet MAE checkpoint normally contains the fixed 14x14
                # grid for 224x224; interpolating it would no longer equal the
                # positions generated directly for the target resolution.
                if value.shape == current[key].shape:
                    compatible[key] = value
                continue
            if key in current and value.shape == current[key].shape:
                compatible[key] = value

        required_prefixes = ("patch_embed.proj.", "blocks.", "norm.")
        loaded_required = [
            key for key in compatible if key.startswith(required_prefixes)
        ]
        if not loaded_required:
            raise RuntimeError(
                f"checkpoint {checkpoint!r} contained no compatible ViT encoder weights"
            )
        message = self.load_state_dict(compatible, strict=False)
        print(
            f"Loaded {len(compatible)} MAE ViT encoder tensors from {checkpoint}; "
            f"missing={len(message.missing_keys)}, unexpected={len(message.unexpected_keys)}"
        )

    def _position_for_grid(
        self,
        height: int,
        width: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        if (height, width) == self.grid_size:
            return self.pos_embed.to(device=device, dtype=dtype)
        position = self.pos_embed.reshape(1, *self.grid_size, self.embed_dim)
        position = position.permute(0, 3, 1, 2)
        position = F.interpolate(
            position,
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        )
        return position.permute(0, 2, 3, 1).reshape(
            1, height * width, self.embed_dim
        ).to(device=device, dtype=dtype)

    def get_intermediate_layers(
        self,
        images: Tensor,
        n: int | Sequence[int] = 4,
        *,
        reshape: bool = True,
        return_class_token: bool = False,
        norm: bool = True,
    ) -> list[Tensor] | list[tuple[Tensor, Tensor]]:
        if return_class_token:
            raise ValueError("CounTR ViT has no class token")
        if images.shape[-2] % self.patch_size or images.shape[-1] % self.patch_size:
            raise ValueError("input dimensions must be divisible by patch size")
        requested = (
            tuple(range(self.depth - n, self.depth))
            if isinstance(n, int)
            else tuple(int(index) for index in n)
        )
        if not requested or min(requested) < 0 or max(requested) >= self.depth:
            raise ValueError(f"invalid ViT block indices: {requested}")

        height = images.shape[-2] // self.patch_size
        width = images.shape[-1] // self.patch_size
        tokens = self.patch_embed(images)
        tokens = tokens + self._position_for_grid(
            height,
            width,
            device=tokens.device,
            dtype=tokens.dtype,
        )
        requested_set = set(requested)
        outputs: dict[int, Tensor] = {}
        for index, block in enumerate(self.blocks):
            tokens = block(tokens)
            if index in requested_set:
                output = self.norm(tokens) if norm else tokens
                if reshape:
                    output = output.transpose(1, 2).reshape(
                        output.shape[0], output.shape[-1], height, width
                    )
                outputs[index] = output
        return [outputs[index] for index in requested]

    def forward(self, images: Tensor) -> Tensor:
        return self.get_intermediate_layers(images, n=(self.depth - 1,))[0]
