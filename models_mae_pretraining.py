"""CounTR-style masked-autoencoder pretraining for the v6 ViT encoder."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor, nn

from models_mae_vit import (
    _build_2d_sincos_position_embedding,
)


DEFAULT_MAE_FULL_CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/mae/pretrain/"
    "mae_pretrain_vit_base_full.pth"
)


class CounTRMaskedAutoencoder(nn.Module):
    """The class-token-free MAE used by CounTR before counting fine-tuning.

    CounTR differs from the canonical ImageNet MAE in two small but important
    ways: it uses a native 384x384 fixed positional grid without a class token,
    and its published implementation averages reconstruction loss over every
    patch rather than only over removed patches.
    """

    def __init__(
        self,
        *,
        img_size: int = 384,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        mlp_ratio: float = 4.0,
        norm_pix_loss: bool = False,
    ) -> None:
        super().__init__()
        try:
            from timm.models.vision_transformer import Block, PatchEmbed
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise ImportError("MAE pretraining requires timm") from exc

        if img_size % patch_size:
            raise ValueError("image size must be divisible by patch size")
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.norm_pix_loss = norm_pix_loss

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=embed_dim,
            strict_img_size=True,
        )
        self.pos_embed = nn.Parameter(
            _build_2d_sincos_position_embedding(*self.grid_size, embed_dim),
            requires_grad=False,
        )
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

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            _build_2d_sincos_position_embedding(
                *self.grid_size, decoder_embed_dim
            ),
            requires_grad=False,
        )
        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for _ in range(decoder_depth)
            ]
        )
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            patch_size * patch_size * in_channels,
            bias=True,
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        nn.init.xavier_uniform_(
            self.patch_embed.proj.weight.data.view(
                self.patch_embed.proj.weight.shape[0], -1
            )
        )
        nn.init.normal_(self.mask_token, std=0.02)
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

    def patchify(self, images: Tensor) -> Tensor:
        if images.shape[-2:] != (self.img_size, self.img_size):
            raise ValueError(
                f"expected {self.img_size}x{self.img_size} images, "
                f"received {tuple(images.shape[-2:])}"
            )
        p = self.patch_size
        height = width = self.img_size // p
        patches = images.reshape(
            images.shape[0], self.in_channels, height, p, width, p
        )
        patches = torch.einsum("nchpwq->nhwpqc", patches)
        return patches.reshape(
            images.shape[0], height * width, p * p * self.in_channels
        )

    def unpatchify(self, patches: Tensor) -> Tensor:
        p = self.patch_size
        height, width = self.grid_size
        if patches.shape[1] != height * width:
            raise ValueError("patch sequence does not match the configured grid")
        images = patches.reshape(
            patches.shape[0], height, width, p, p, self.in_channels
        )
        images = torch.einsum("nhwpqc->nchpwq", images)
        return images.reshape(
            patches.shape[0], self.in_channels, height * p, width * p
        )

    @staticmethod
    def random_masking(tokens: Tensor, mask_ratio: float):
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError("mask_ratio must be strictly between zero and one")
        batch, length, channels = tokens.shape
        keep = int(length * (1.0 - mask_ratio))
        noise = torch.rand(batch, length, device=tokens.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :keep]
        visible = torch.gather(
            tokens,
            dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, channels),
        )
        mask = torch.ones(batch, length, device=tokens.device)
        mask[:, :keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return visible, mask, ids_restore

    def forward_encoder(self, images: Tensor, mask_ratio: float):
        tokens = self.patch_embed(images) + self.pos_embed
        tokens, mask, ids_restore = self.random_masking(tokens, mask_ratio)
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens), mask, ids_restore

    def forward_decoder(self, visible: Tensor, ids_restore: Tensor) -> Tensor:
        visible = self.decoder_embed(visible)
        masked_count = ids_restore.shape[1] - visible.shape[1]
        mask_tokens = self.mask_token.expand(visible.shape[0], masked_count, -1)
        tokens = torch.cat((visible, mask_tokens), dim=1)
        tokens = torch.gather(
            tokens,
            dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]),
        )
        tokens = tokens + self.decoder_pos_embed
        for block in self.decoder_blocks:
            tokens = block(tokens)
        return self.decoder_pred(self.decoder_norm(tokens))

    def forward_loss(self, images: Tensor, prediction: Tensor) -> Tensor:
        target = self.patchify(images)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            variance = target.var(dim=-1, keepdim=True)
            target = (target - mean) / torch.sqrt(variance + 1e-6)
        # This intentionally matches CounTR's all-patch MAE objective.
        return (prediction - target).square().mean(dim=-1).mean()

    def forward(self, images: Tensor, mask_ratio: float = 0.5):
        visible, mask, ids_restore = self.forward_encoder(images, mask_ratio)
        prediction = self.forward_decoder(visible, ids_restore)
        return self.forward_loss(images, prediction), prediction, mask

    def load_initial_weights(
        self,
        checkpoint: str = DEFAULT_MAE_FULL_CHECKPOINT_URL,
    ) -> None:
        """Warm-start from an official MAE or a CounTR MAE checkpoint."""

        if checkpoint.startswith(("http://", "https://")):
            payload = torch.hub.load_state_dict_from_url(
                checkpoint, map_location="cpu", check_hash=False
            )
        else:
            path = Path(checkpoint).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"MAE checkpoint not found: {path}")
            payload = torch.load(path, map_location="cpu", weights_only=False)
        source: Mapping[str, Tensor] = payload.get("model", payload)
        current = self.state_dict()
        compatible = {}
        for source_key, value in source.items():
            key = source_key
            for prefix in ("module.", "backbone."):
                if key.startswith(prefix):
                    key = key[len(prefix) :]
            if key in {"cls_token", "pos_embed", "decoder_pos_embed"}:
                if key in current and value.shape == current[key].shape:
                    compatible[key] = value
                continue
            if key in current and value.shape == current[key].shape:
                compatible[key] = value
        if not any(key.startswith("blocks.") for key in compatible):
            raise RuntimeError("checkpoint contains no compatible ViT blocks")
        message = self.load_state_dict(compatible, strict=False)
        print(
            f"Loaded {len(compatible)} MAE tensors from {checkpoint}; "
            f"missing={len(message.missing_keys)}, "
            f"unexpected={len(message.unexpected_keys)}"
        )


def countr_mae_vit_base_patch16(**kwargs) -> CounTRMaskedAutoencoder:
    return CounTRMaskedAutoencoder(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        **kwargs,
    )
