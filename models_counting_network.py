"""Modern spatially-aware class-agnostic counting network.

The original implementation reduced DINOv2 to its last feature map, converted
GPU tensors through PIL inside ``forward`` and computed text features without
using them.  This module keeps the public ``CountingNetwork`` entry point while
updating the architecture around four ideas from recent counting literature:

* dense, multi-depth DINOv2 features;
* explicit and resolution-independent 2-D positional encoding;
* optional text/query conditioning that actually modulates spatial features;
* a differentiable proposal-and-verification density head.

Inputs are RGB tensors in ``[0, 1]``.  Device placement is inherited from the
input/model; there are no hard-coded CUDA transfers.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from models_mae_vit import CounTRMAEViTBackbone


# Pinning the hub repository prevents a future DINOv2 main-branch change from
# silently changing this model.  The value can still be overridden from CLI.
DEFAULT_DINOV2_REPO = (
    "facebookresearch/dinov2:7764ea0f912e53c92e82eb78a2a1631e92725fc8"
)

DINOV2_DIMS = {
    "dinov2_vits14": 384,
    "dinov2_vits14_reg": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitb14_reg": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitl14_reg": 1024,
}

IMAGE_ONLY_ARCHITECTURES = frozenset({"v4", "v5", "v6"})


def _group_count(channels: int, maximum: int = 8) -> int:
    """Return the largest useful GroupNorm group count for ``channels``."""

    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def build_2d_sincos_position_embedding(
    height: int,
    width: int,
    channels: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Create a dynamic 2-D sine/cosine encoding with shape ``[1,C,H,W]``."""

    if channels % 4 != 0:
        raise ValueError("decoder_dim must be divisible by four")

    quarter = channels // 4
    omega = torch.arange(quarter, device=device, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / max(quarter - 1, 1)))
    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=torch.float32)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")

    phase_x = grid_x[..., None] * omega
    phase_y = grid_y[..., None] * omega
    position = torch.cat(
        (phase_x.sin(), phase_x.cos(), phase_y.sin(), phase_y.cos()), dim=-1
    )
    return position.permute(2, 0, 1).unsqueeze(0).to(dtype=dtype)


class OpenCLIPTextEncoder(nn.Module):
    """Frozen OpenCLIP text tower with the unused visual tower removed."""

    def __init__(
        self,
        model_name: str = "ViT-B-16",
        pretrained: str = "laion2b_s34b_b88k",
    ) -> None:
        super().__init__()
        try:
            import open_clip
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise ImportError(
                "Text conditioning requires open_clip_torch. Install requirements.txt "
                "or construct CountingNetwork(enable_text_conditioning=False)."
            ) from exc

        model = open_clip.create_model(model_name, pretrained=pretrained)
        # encode_text does not use the visual tower. Removing it saves a large
        # amount of checkpoint and GPU memory while preserving the text API.
        model.visual = nn.Identity()
        for parameter in model.parameters():
            parameter.requires_grad = False
        self.model = model

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.ndim == 3:
            batch, shots, length = tokens.shape
            encoded = self.model.encode_text(tokens.reshape(batch * shots, length))
            return encoded.reshape(batch, shots, -1).mean(dim=1)
        if tokens.ndim != 2:
            raise ValueError("counting_queries must have shape [B,L] or [B,S,L]")
        return self.model.encode_text(tokens)


class MultiDepthFusion(nn.Module):
    """Project and fuse several DINO transformer depths at patch resolution."""

    def __init__(self, in_channels: int, out_channels: int, levels: int) -> None:
        super().__init__()
        self.levels = levels
        self.projections = nn.ModuleList(
            [nn.Conv2d(in_channels, out_channels, kernel_size=1) for _ in range(levels)]
        )
        self.level_logits = nn.Parameter(torch.zeros(levels))
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * levels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
        )

    def forward(self, features: Sequence[Tensor]) -> Tuple[Tensor, Tensor]:
        if len(features) != self.levels:
            raise ValueError(f"expected {self.levels} feature levels, got {len(features)}")
        projected = [projection(feature) for projection, feature in zip(self.projections, features)]
        weights = self.level_logits.softmax(dim=0)
        weighted = sum(weight * feature for weight, feature in zip(weights, projected))
        return weighted + self.fuse(torch.cat(projected, dim=1)), weights


class SpatialMultiDepthFusion(nn.Module):
    """Fuse DINO depths with content-dependent weights at every patch.

    A single global scalar per transformer depth cannot choose shallow detail in
    crowded regions and deep semantics in ambiguous regions.  This variant
    predicts a softmax over levels independently at every spatial location.
    """

    def __init__(self, in_channels: int, out_channels: int, levels: int) -> None:
        super().__init__()
        self.levels = levels
        self.projections = nn.ModuleList(
            [nn.Conv2d(in_channels, out_channels, kernel_size=1) for _ in range(levels)]
        )
        self.level_logits = nn.Parameter(torch.zeros(levels))
        self.spatial_logits = nn.Sequential(
            nn.Conv2d(out_channels * levels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, levels, kernel_size=1),
        )
        nn.init.zeros_(self.spatial_logits[-1].weight)
        nn.init.zeros_(self.spatial_logits[-1].bias)
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * levels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
        )

    def forward(self, features: Sequence[Tensor]) -> Tuple[Tensor, Tensor]:
        if len(features) != self.levels:
            raise ValueError(f"expected {self.levels} feature levels, got {len(features)}")
        projected = [projection(feature) for projection, feature in zip(self.projections, features)]
        concatenated = torch.cat(projected, dim=1)
        logits = self.spatial_logits(concatenated)
        logits = logits + self.level_logits.view(1, -1, 1, 1)
        weights = logits.softmax(dim=1)
        stacked = torch.stack(projected, dim=1)
        weighted = (stacked * weights.unsqueeze(2)).sum(dim=1)
        return weighted + self.fuse(concatenated), weights


class PyramidResidualUnit(nn.Module):
    """Residual convolution used by the DPT-style top-down pyramid."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(_group_count(channels), channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, features: Tensor) -> Tensor:
        return features + self.block(features)


class DinoPyramidFusion(nn.Module):
    """Reassemble spaced ViT layers into a DPT-style feature pyramid.

    Vanilla ViT and DINO transformer layers all have patch resolution. Following the DPT
    reassembly pattern, the four projected levels are mapped to relative
    resolutions x4, x2, x1, and x0.5 before top-down residual fusion.  The
    highest-resolution output is subsequently aligned with the stride-four RGB
    detail stem by :class:`CountingNetwork`.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        levels: int = 4,
    ) -> None:
        super().__init__()
        if levels != 4:
            raise ValueError("DinoPyramidFusion requires exactly four feature levels")
        self.levels = levels
        self.projections = nn.ModuleList(
            [nn.Conv2d(in_channels, hidden_channels, kernel_size=1) for _ in range(levels)]
        )
        self.reassemble = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    hidden_channels, hidden_channels, kernel_size=4, stride=4
                ),
                nn.ConvTranspose2d(
                    hidden_channels, hidden_channels, kernel_size=2, stride=2
                ),
                nn.Identity(),
                nn.Conv2d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            ]
        )
        self.level_logits = nn.Parameter(torch.zeros(levels))
        self.refine = nn.ModuleList(
            [PyramidResidualUnit(hidden_channels) for _ in range(levels)]
        )
        self.output = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
        )

    def forward(
        self, features: Sequence[Tensor]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if len(features) != self.levels:
            raise ValueError(f"expected {self.levels} feature levels, got {len(features)}")
        projected = [
            projection(feature)
            for projection, feature in zip(self.projections, features)
        ]
        semantic_anchor = projected[-1]
        weights = self.level_logits.softmax(dim=0)
        pyramid = [
            resize(weight * feature)
            for resize, weight, feature in zip(self.reassemble, weights, projected)
        ]

        path = self.refine[-1](pyramid[-1])
        for index in range(self.levels - 2, -1, -1):
            path = F.interpolate(
                path,
                size=pyramid[index].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            path = self.refine[index](path + pyramid[index])
        return self.output(path), weights, semantic_anchor


class FeatUpJBUAdapter(nn.Module):
    """Pure-PyTorch adaptation of FeatUp's learned joint bilateral upsampler.

    The official FeatUp checkpoint targets DINOv2-S/14 features and relies on a
    custom adaptive-convolution extension.  This task-specific variant learns
    range and spatial kernels for the ViT-B semantic feature and applies them
    with ``unfold``, keeping the repository portable and end-to-end trainable.
    """

    def __init__(
        self,
        source_channels: int,
        output_channels: int,
        guidance_channels: int,
        *,
        key_channels: int = 16,
        radius: int = 2,
    ) -> None:
        super().__init__()
        if radius <= 0:
            raise ValueError("FeatUp JBU radius must be positive")
        self.radius = radius
        self.kernel_size = 2 * radius + 1
        neighbors = self.kernel_size**2
        self.source_projection = nn.Conv2d(
            source_channels, output_channels, kernel_size=1, bias=False
        )
        self.range_projection = nn.Sequential(
            nn.Conv2d(guidance_channels, key_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(key_channels, key_channels, kernel_size=1),
        )
        self.kernel_correction = nn.Sequential(
            nn.Conv2d(guidance_channels + neighbors, neighbors, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(neighbors, neighbors, kernel_size=1),
        )
        self.log_range_temperature = nn.Parameter(torch.tensor(0.0))
        self.log_spatial_sigma = nn.Parameter(torch.tensor(0.0))
        self.output_norm = nn.GroupNorm(
            _group_count(output_channels), output_channels
        )

    def _spatial_logits(self, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        coordinates = torch.arange(
            -self.radius, self.radius + 1, device=device, dtype=torch.float32
        )
        y, x = torch.meshgrid(coordinates, coordinates, indexing="ij")
        squared_distance = x.square() + y.square()
        sigma = self.log_spatial_sigma.exp().clamp(0.25, 4.0)
        logits = -squared_distance / (2.0 * sigma.square())
        return logits.flatten().view(1, -1, 1, 1).to(dtype=dtype)

    def forward(self, source: Tensor, guidance: Tensor) -> Tensor:
        batch, _, height, width = guidance.shape
        source = self.source_projection(source)
        source = F.interpolate(
            source,
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        )
        keys = self.range_projection(guidance)
        key_neighbors = F.unfold(
            F.pad(keys, [self.radius] * 4, mode="reflect"),
            kernel_size=self.kernel_size,
        ).view(batch, keys.shape[1], self.kernel_size**2, height, width)
        range_logits = (key_neighbors * keys.unsqueeze(2)).sum(dim=1)
        range_temperature = self.log_range_temperature.exp().clamp(1e-2, 100.0)
        logits = range_temperature * range_logits
        logits = logits + self._spatial_logits(device=source.device, dtype=source.dtype)
        preliminary_weights = logits.softmax(dim=1)
        logits = logits + 0.1 * self.kernel_correction(
            torch.cat((preliminary_weights, guidance), dim=1)
        )
        weights = logits.softmax(dim=1)

        source_neighbors = F.unfold(
            F.pad(source, [self.radius] * 4, mode="reflect"),
            kernel_size=self.kernel_size,
        ).view(
            batch,
            source.shape[1],
            self.kernel_size**2,
            height,
            width,
        )
        upsampled = (source_neighbors * weights.unsqueeze(1)).sum(dim=2)
        return self.output_norm(upsampled)


class RepetitionPrototypeGate(nn.Module):
    """Build an image-derived prototype from non-local DINO self-similarity."""

    def __init__(
        self,
        semantic_channels: int,
        output_channels: int,
        *,
        topk: int = 16,
        local_radius: int = 1,
        prototype_temperature: float = 0.1,
    ) -> None:
        super().__init__()
        if topk <= 0:
            raise ValueError("repetition topk must be positive")
        if local_radius < 0:
            raise ValueError("repetition local_radius must be non-negative")
        if prototype_temperature <= 0:
            raise ValueError("prototype_temperature must be positive")
        self.topk = topk
        self.local_radius = local_radius
        self.prototype_temperature = prototype_temperature
        self.key = nn.Conv2d(
            semantic_channels, semantic_channels, kernel_size=1, bias=False
        )
        self.value = nn.Conv2d(
            semantic_channels, output_channels, kernel_size=1, bias=False
        )
        gate_channels = max(output_channels // 2, 16)
        self.signal_fusion = nn.Sequential(
            nn.Conv2d(2, gate_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(gate_channels, 1, kernel_size=1),
        )
        self.feature_gate = nn.Conv2d(1, output_channels, kernel_size=1)
        # Adding repetition gating to an image-only DPT checkpoint starts as an
        # identity feature transform while still exposing the raw signal to the
        # verification head.
        nn.init.zeros_(self.feature_gate.weight)
        nn.init.zeros_(self.feature_gate.bias)

    def _non_local_mask(
        self, height: int, width: int, *, device: torch.device
    ) -> Tensor:
        y, x = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing="ij",
        )
        coordinates = torch.stack((y.flatten(), x.flatten()), dim=1)
        distance = (coordinates[:, None] - coordinates[None, :]).abs()
        return distance.amax(dim=-1) <= self.local_radius

    def forward(
        self, semantic: Tensor, decoded: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        batch, _, height, width = semantic.shape
        keys = F.normalize(self.key(semantic), dim=1)
        tokens = keys.flatten(2).transpose(1, 2)
        affinity = torch.bmm(tokens, tokens.transpose(1, 2))
        non_local_mask = self._non_local_mask(height, width, device=semantic.device)
        affinity = affinity.masked_fill(non_local_mask.unsqueeze(0), -torch.inf)
        available = (~non_local_mask).sum(dim=1).amin().item()
        k = min(self.topk, int(available))
        if k <= 0:
            raise ValueError("feature map is too small for non-local repetition gating")
        repetition = affinity.topk(k, dim=-1).values.mean(dim=-1)
        repetition_mean = repetition.mean(dim=1, keepdim=True)
        repetition_std = repetition.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        repetition_normalized = ((repetition - repetition_mean) / repetition_std).sigmoid()

        prototype_weights = F.softmax(
            repetition / self.prototype_temperature, dim=1
        )
        values = self.value(semantic).flatten(2).transpose(1, 2)
        prototype = torch.sum(prototype_weights.unsqueeze(-1) * values, dim=1)
        normalized_values = F.normalize(values, dim=-1)
        normalized_prototype = F.normalize(prototype, dim=-1)
        prototype_similarity = torch.einsum(
            "bnc,bc->bn", normalized_values, normalized_prototype
        )

        repetition_map = repetition_normalized.view(batch, 1, height, width)
        similarity_map = prototype_similarity.view(batch, 1, height, width)
        signal = self.signal_fusion(torch.cat((repetition_map, similarity_map), dim=1))
        signal = F.interpolate(
            signal, size=decoded.shape[-2:], mode="bilinear", align_corners=False
        )
        conditioned = decoded * (1.0 + 0.25 * self.feature_gate(signal).tanh())
        return conditioned, signal, prototype, repetition_map


class RGBDetailStem(nn.Module):
    """Retain stride-four image evidence lost by a patch-14 transformer."""

    def __init__(self, out_channels: int) -> None:
        super().__init__()
        mid_channels = max(out_channels // 2, 16)
        self.stem = nn.Sequential(
            nn.Conv2d(3, mid_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(_group_count(mid_channels), mid_channels),
            nn.GELU(),
            nn.Conv2d(
                mid_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
            SpatialContextBlock(out_channels),
        )

    def forward(self, images: Tensor) -> Tensor:
        return self.stem(images)


class SpatialContextBlock(nn.Module):
    """Local multi-scale context on top of DINO's global transformer features."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.local = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels, bias=False
        )
        self.dilated = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=2,
            dilation=2,
            groups=channels,
            bias=False,
        )
        self.project = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels, kernel_size=1),
        )

    def forward(self, features: Tensor) -> Tensor:
        normalized = self.norm(features)
        context = torch.cat((self.local(normalized), self.dilated(normalized)), dim=1)
        return features + self.project(context)


class QuerySpatialFusion(nn.Module):
    """Fuse an image-derived or text-derived query into dense spatial features."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.key = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.query = nn.Linear(channels, channels, bias=False)
        self.film = nn.Linear(channels, channels * 2)
        self.output = nn.Sequential(
            nn.Conv2d(channels + 1, channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
            nn.GELU(),
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))

    def forward(self, features: Tensor, query: Tensor) -> Tuple[Tensor, Tensor]:
        keys = F.normalize(self.key(features), dim=1)
        query_key = F.normalize(self.query(query), dim=-1)
        similarity = torch.einsum("bchw,bc->bhw", keys, query_key).unsqueeze(1)
        similarity = similarity * self.logit_scale.exp().clamp(max=100.0)

        gamma, beta = self.film(query).chunk(2, dim=-1)
        gamma = 0.1 * gamma.tanh().unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        conditioned = features * (1.0 + gamma) + beta
        fused = features + self.output(torch.cat((conditioned, similarity), dim=1))
        return fused, similarity


class ResidualUpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
        )
        self.activation = nn.GELU()

    def forward(self, features: Tensor) -> Tensor:
        features = F.interpolate(
            features, scale_factor=2.0, mode="bilinear", align_corners=False
        )
        return self.activation(self.main(features) + self.skip(features))


class CountingNetwork(nn.Module):
    """Spatially-aware, optionally text-guided class-agnostic counter.

    ``counting_queries`` preserves the original token-input interface.  For
    experiments or unit tests, ``query_features`` can supply already-encoded
    features directly.  With neither input, an image-derived query enables the
    original reference-less majority-object setting.
    """

    def __init__(
        self,
        *,
        backbone_name: str = "dinov2_vitb14_reg",
        backbone_repo: str = DEFAULT_DINOV2_REPO,
        backbone: Optional[nn.Module] = None,
        backbone_dim: Optional[int] = None,
        patch_size: int = 14,
        num_feature_levels: int = 4,
        decoder_dim: int = 192,
        decoder_channels: Sequence[int] = (128, 96, 64),
        freeze_backbone: bool = True,
        trainable_backbone_blocks: int = 0,
        enable_text_conditioning: bool = True,
        query_dim: int = 512,
        text_model_name: str = "ViT-B-16",
        text_pretrained: str = "laion2b_s34b_b88k",
        text_dropout_p: float = 0.25,
        architecture_version: str = "v3",
        dino_layer_indices: Optional[Sequence[int]] = None,
        v4_query_mode: str = "repetition",
        repetition_topk: int = 16,
        mae_checkpoint: Optional[str] = None,
        mae_pretrained: bool = True,
    ) -> None:
        super().__init__()
        if decoder_dim % 4 != 0:
            raise ValueError("decoder_dim must be divisible by four")
        if not 0.0 <= text_dropout_p <= 1.0:
            raise ValueError("text_dropout_p must be between zero and one")
        if architecture_version not in {"v1", "v2", "v3", "v4", "v5", "v6"}:
            raise ValueError(
                "architecture_version must be 'v1', 'v2', 'v3', 'v4', 'v5', or 'v6'"
            )
        if trainable_backbone_blocks < 0:
            raise ValueError("trainable_backbone_blocks must be non-negative")
        if v4_query_mode not in {"image_film", "none", "repetition"}:
            raise ValueError(
                "v4_query_mode must be 'image_film', 'none', or 'repetition'"
            )
        if architecture_version in IMAGE_ONLY_ARCHITECTURES and enable_text_conditioning:
            raise ValueError("v4/v5/v6 are image-only; disable text conditioning")
        if dino_layer_indices is not None:
            dino_layer_indices = tuple(int(index) for index in dino_layer_indices)
            if len(dino_layer_indices) != num_feature_levels:
                raise ValueError(
                    "dino_layer_indices must contain num_feature_levels entries"
                )
            if tuple(sorted(set(dino_layer_indices))) != dino_layer_indices:
                raise ValueError("dino_layer_indices must be unique and increasing")

        if architecture_version == "v6":
            patch_size = 16
            backbone_name = "countr_mae_vitb16"
        self.backbone_name = backbone_name
        self.backbone_repo = backbone_repo
        self.patch_size = patch_size
        self.num_feature_levels = num_feature_levels
        self.text_dropout_p = text_dropout_p
        self.architecture_version = architecture_version
        self.dino_layer_indices = (
            tuple(dino_layer_indices)
            if dino_layer_indices is not None
            else (
                (2, 5, 8, 11)
                if architecture_version in IMAGE_ONLY_ARCHITECTURES
                else None
            )
        )
        self.v4_query_mode = v4_query_mode

        if backbone is None:
            if architecture_version == "v6":
                backbone = CounTRMAEViTBackbone(
                    img_size=384,
                    patch_size=16,
                    pretrained=mae_pretrained,
                    checkpoint=mae_checkpoint,
                )
                backbone_dim = 768
            else:
                if backbone_name not in DINOV2_DIMS and backbone_dim is None:
                    raise ValueError(
                        f"unknown backbone {backbone_name!r}; provide backbone_dim explicitly"
                    )
                backbone = torch.hub.load(
                    backbone_repo,
                    backbone_name,
                    pretrained=True,
                    trust_repo=True,
                )
        self.backbone = backbone
        resolved_backbone_dim = (
            backbone_dim
            or getattr(backbone, "embed_dim", None)
            or DINOV2_DIMS[backbone_name]
        )

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
            if trainable_backbone_blocks:
                blocks = getattr(self.backbone, "blocks", None)
                if blocks is None:
                    raise ValueError(
                        "partial backbone fine-tuning requires a backbone.blocks sequence"
                    )
                if trainable_backbone_blocks > len(blocks):
                    raise ValueError(
                        f"requested {trainable_backbone_blocks} trainable blocks, "
                        f"but backbone only has {len(blocks)}"
                    )
                for block in blocks[-trainable_backbone_blocks:]:
                    for parameter in block.parameters():
                        parameter.requires_grad = True
                backbone_norm = getattr(self.backbone, "norm", None)
                if backbone_norm is not None:
                    for parameter in backbone_norm.parameters():
                        parameter.requires_grad = True
        self.freeze_backbone = not any(
            parameter.requires_grad for parameter in self.backbone.parameters()
        )
        self.trainable_backbone_blocks = trainable_backbone_blocks

        v4_final_channels = decoder_channels[-1]
        if architecture_version in IMAGE_ONLY_ARCHITECTURES:
            self.feature_fusion = DinoPyramidFusion(
                resolved_backbone_dim,
                decoder_dim,
                v4_final_channels,
                num_feature_levels,
            )
        else:
            fusion_class = (
                MultiDepthFusion
                if architecture_version == "v1"
                else SpatialMultiDepthFusion
            )
            self.feature_fusion = fusion_class(
                resolved_backbone_dim, decoder_dim, num_feature_levels
            )
        self.position_gain = nn.Parameter(torch.tensor(1.0))
        self.spatial_context = nn.Sequential(
            SpatialContextBlock(decoder_dim),
            SpatialContextBlock(decoder_dim),
        )

        self.text_encoder: Optional[nn.Module]
        if enable_text_conditioning:
            self.text_encoder = OpenCLIPTextEncoder(text_model_name, text_pretrained)
        else:
            self.text_encoder = None
        if architecture_version in IMAGE_ONLY_ARCHITECTURES:
            self.query_projection = None
            self.image_query = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(v4_final_channels, v4_final_channels),
                nn.GELU(),
            )
            self.query_fusion = QuerySpatialFusion(v4_final_channels)
            self.repetition_gate = RepetitionPrototypeGate(
                decoder_dim, v4_final_channels, topk=repetition_topk
            )
        else:
            self.query_projection = nn.Linear(query_dim, decoder_dim)
            self.image_query = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(decoder_dim, decoder_dim),
                nn.GELU(),
            )
            self.query_fusion = QuerySpatialFusion(decoder_dim)

        if architecture_version in {"v1", "v3"}:
            channels = [decoder_dim, *decoder_channels]
            self.decoder = nn.Sequential(
                *[
                    ResidualUpsampleBlock(in_channels, out_channels)
                    for in_channels, out_channels in zip(channels[:-1], channels[1:])
                ]
            )
            final_channels = channels[-1]
            if architecture_version == "v3":
                detail_channels = max(decoder_dim // 3, 16)
                refinement_channels = (detail_channels, max(detail_channels * 3 // 4, 16))
                self.detail_stem = RGBDetailStem(detail_channels)
                self.detail_fusion = nn.Sequential(
                    nn.Conv2d(
                        final_channels + detail_channels,
                        final_channels,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    nn.GroupNorm(_group_count(final_channels), final_channels),
                    nn.GELU(),
                    SpatialContextBlock(final_channels),
                )
                refinement_path = [final_channels, *refinement_channels]
                self.refinement_decoder = nn.Sequential(
                    *[
                        ResidualUpsampleBlock(in_channels, out_channels)
                        for in_channels, out_channels in zip(
                            refinement_path[:-1], refinement_path[1:]
                        )
                    ]
                )
                self.density_refinement_head = nn.Sequential(
                    nn.Conv2d(
                        refinement_channels[-1],
                        refinement_channels[-1],
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.GELU(),
                    nn.Conv2d(refinement_channels[-1], 1, kernel_size=1),
                )
                nn.init.zeros_(self.density_refinement_head[-1].weight)
                nn.init.zeros_(self.density_refinement_head[-1].bias)
        elif architecture_version == "v2":
            semantic_channels = (max(decoder_dim * 2 // 3, 32), max(decoder_dim // 2, 24))
            detail_channels = max(decoder_dim // 3, 16)
            refinement_channels = (detail_channels, max(detail_channels * 3 // 4, 16))
            semantic_path = [decoder_dim, *semantic_channels]
            self.semantic_decoder = nn.Sequential(
                *[
                    ResidualUpsampleBlock(in_channels, out_channels)
                    for in_channels, out_channels in zip(semantic_path[:-1], semantic_path[1:])
                ]
            )
            self.detail_stem = RGBDetailStem(detail_channels)
            self.detail_fusion = nn.Sequential(
                nn.Conv2d(
                    semantic_channels[-1] + detail_channels,
                    semantic_channels[-1],
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(_group_count(semantic_channels[-1]), semantic_channels[-1]),
                nn.GELU(),
                SpatialContextBlock(semantic_channels[-1]),
            )
            refinement_path = [semantic_channels[-1], *refinement_channels]
            self.refinement_decoder = nn.Sequential(
                *[
                    ResidualUpsampleBlock(in_channels, out_channels)
                    for in_channels, out_channels in zip(
                        refinement_path[:-1], refinement_path[1:]
                    )
                ]
            )
            final_channels = refinement_channels[-1]
        else:
            # v4/v5/v6 receive an already reassembled stride-four ViT pyramid.
            # Fuse it with RGB detail without repeatedly interpolating a single
            # patch-resolution tensor through the legacy decoder.
            final_channels = v4_final_channels
            detail_channels = max(decoder_dim // 3, 16)
            self.detail_stem = RGBDetailStem(detail_channels)
            if architecture_version in {"v5", "v6"}:
                self.featup_adapter = FeatUpJBUAdapter(
                    decoder_dim,
                    final_channels,
                    detail_channels,
                )
                self.featup_fusion = nn.Sequential(
                    nn.Conv2d(
                        final_channels * 2,
                        final_channels,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    nn.GroupNorm(_group_count(final_channels), final_channels),
                    nn.GELU(),
                    nn.Conv2d(final_channels, final_channels, kernel_size=1),
                )
                nn.init.zeros_(self.featup_fusion[-1].weight)
                nn.init.zeros_(self.featup_fusion[-1].bias)
            self.detail_fusion = nn.Sequential(
                nn.Conv2d(
                    final_channels + detail_channels,
                    final_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(_group_count(final_channels), final_channels),
                nn.GELU(),
                SpatialContextBlock(final_channels),
            )
            self.density_refinement_head = nn.Sequential(
                nn.Conv2d(
                    final_channels,
                    final_channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.GELU(),
                nn.Conv2d(final_channels, 1, kernel_size=1),
            )
            nn.init.zeros_(self.density_refinement_head[-1].weight)
            nn.init.zeros_(self.density_refinement_head[-1].bias)
        self.proposal_head = nn.Sequential(
            nn.Conv2d(final_channels, final_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(final_channels, 1, kernel_size=1),
        )
        self.verification_head = nn.Sequential(
            nn.Conv2d(final_channels + 1, final_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(final_channels, 1, kernel_size=1),
        )

        # FSC-147 density maps are sparse. These priors avoid an enormous
        # uniform initial count while keeping gradients well behaved.
        nn.init.constant_(self.proposal_head[-1].bias, -4.0)
        nn.init.constant_(
            self.verification_head[-1].bias,
            2.0 if architecture_version in {"v1", "v3", "v4", "v5", "v6"} else 0.0,
        )

        if architecture_version in {"v2", "v3", "v4", "v5", "v6"}:
            # The spatial decoder determines *where* objects are.  This small
            # head only calibrates total mass, which is essential when many
            # small objects occupy a single patch token.  Zero initialization
            # makes the initial correction exactly one.
            self.max_log_count_scale = math.log(
                12.0 if architecture_version == "v2" else 4.0
            )
            count_feature_channels = (
                final_channels
                if architecture_version in IMAGE_ONLY_ARCHITECTURES
                else decoder_dim
            )
            self.count_scale_head = nn.Sequential(
                nn.Linear(count_feature_channels * 2, decoder_dim),
                nn.GELU(),
                nn.Linear(decoder_dim, 1),
            )
            nn.init.zeros_(self.count_scale_head[-1].weight)
            nn.init.zeros_(self.count_scale_head[-1].bias)

        self.register_buffer(
            "image_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "image_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        )

    def train(self, mode: bool = True) -> "CountingNetwork":
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        if self.text_encoder is not None:
            self.text_encoder.eval()
        return self

    def _prepare_images(self, images: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape [B,3,H,W]")
        original_size = (images.shape[-2], images.shape[-1])
        images = images.float()
        pad_h = (-images.shape[-2]) % self.patch_size
        pad_w = (-images.shape[-1]) % self.patch_size
        if pad_h or pad_w:
            mode = "reflect" if images.shape[-2] > pad_h and images.shape[-1] > pad_w else "replicate"
            images = F.pad(images, (0, pad_w, 0, pad_h), mode=mode)
        return (images - self.image_mean) / self.image_std, original_size

    def _extract_features(self, images: Tensor) -> Sequence[Tensor]:
        requested_layers: int | Sequence[int] = (
            self.dino_layer_indices
            if self.dino_layer_indices is not None
            else self.num_feature_levels
        )

        def extract() -> Sequence[Tensor]:
            try:
                outputs = self.backbone.get_intermediate_layers(
                    images,
                    n=requested_layers,
                    reshape=True,
                    return_class_token=False,
                    norm=True,
                )
            except TypeError:
                outputs = self.backbone.get_intermediate_layers(
                    images,
                    n=requested_layers,
                    reshape=True,
                    return_class_token=False,
                )
            return outputs

        if self.freeze_backbone:
            with torch.no_grad():
                outputs = extract()
        else:
            outputs = extract()

        features = []
        for output in outputs:
            if isinstance(output, tuple):
                output = output[0]
            if output.ndim == 3:
                height = images.shape[-2] // self.patch_size
                width = images.shape[-1] // self.patch_size
                output = output.transpose(1, 2).reshape(
                    output.shape[0], output.shape[-1], height, width
                )
            features.append(output)
        return features

    def _resolve_query(
        self,
        features: Tensor,
        counting_queries: Optional[Tensor],
        query_features: Optional[Tensor],
    ) -> Tensor:
        image_query = self.image_query(features)
        if query_features is not None:
            if query_features.ndim == 3:
                query_features = query_features.mean(dim=1)
            encoded_query = query_features
        elif counting_queries is not None:
            if self.text_encoder is None:
                raise RuntimeError(
                    "counting_queries were supplied but text conditioning is disabled"
                )
            with torch.no_grad():
                encoded_query = self.text_encoder(counting_queries)
        else:
            return image_query

        query = self.query_projection(
            encoded_query.to(device=features.device, dtype=features.dtype)
        )
        if self.training and self.text_dropout_p > 0.0:
            # Random modality switching makes the model robust in both the
            # prompt-guided and original reference-less settings.
            use_image_query = torch.rand(
                query.shape[0], 1, device=query.device
            ) < self.text_dropout_p
            query = torch.where(use_image_query, image_query, query)
        return query

    def forward(
        self,
        images: Tensor,
        counting_queries: Optional[Tensor] = None,
        *,
        query_features: Optional[Tensor] = None,
        return_aux: bool = False,
    ) -> Tensor | Dict[str, Tensor]:
        prepared, original_size = self._prepare_images(images)
        backbone_features = self._extract_features(prepared)
        repetition_map: Optional[Tensor] = None
        density_refinement_logits: Optional[Tensor] = None
        if self.architecture_version in IMAGE_ONLY_ARCHITECTURES:
            if counting_queries is not None or query_features is not None:
                raise ValueError(
                    "v4/v5/v6 are image-only and do not accept text/query features"
                )
            decoded, level_weights, semantic_anchor = self.feature_fusion(
                backbone_features
            )
            raw_images = prepared * self.image_std + self.image_mean
            detail = self.detail_stem(raw_images)
            decoded = F.interpolate(
                decoded,
                size=detail.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            if self.architecture_version in {"v5", "v6"}:
                featup = self.featup_adapter(semantic_anchor, detail)
                decoded = decoded + self.featup_fusion(
                    torch.cat((decoded, featup), dim=1)
                )
            decoded = self.detail_fusion(torch.cat((decoded, detail), dim=1))
            if self.v4_query_mode == "image_film":
                query = self.image_query(decoded)
                decoded, similarity = self.query_fusion(decoded, query)
            elif self.v4_query_mode == "none":
                query = self.image_query(decoded)
                similarity = decoded.new_zeros(
                    decoded.shape[0], 1, *decoded.shape[-2:]
                )
            else:
                decoded, similarity, _, repetition_map = self.repetition_gate(
                    semantic_anchor, decoded
                )
                # Keep global count calibration comparable across v4 ablations;
                # the repetition prototype replaces spatial FiLM, not the
                # image-derived scalar count context.
                query = self.image_query(decoded)
            density_refinement_logits = self.density_refinement_head(decoded)
            count_features = decoded
        else:
            features, level_weights = self.feature_fusion(backbone_features)
            position = build_2d_sincos_position_embedding(
                features.shape[-2],
                features.shape[-1],
                features.shape[1],
                device=features.device,
                dtype=features.dtype,
            )
            features = features + self.position_gain * position
            features = self.spatial_context(features)
            query = self._resolve_query(features, counting_queries, query_features)
            features, similarity = self.query_fusion(features, query)
            count_features = features
            if self.architecture_version in {"v1", "v3"}:
                decoded = self.decoder(features)
                if self.architecture_version == "v3":
                    raw_images = prepared * self.image_std + self.image_mean
                    detail = self.detail_stem(raw_images)
                    decoded_for_refinement = F.interpolate(
                        decoded,
                        size=detail.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    refinement = self.detail_fusion(
                        torch.cat((decoded_for_refinement, detail), dim=1)
                    )
                    refinement = self.refinement_decoder(refinement)
                    density_refinement_logits = self.density_refinement_head(refinement)
            else:
                semantic = self.semantic_decoder(features)
                raw_images = prepared * self.image_std + self.image_mean
                detail = self.detail_stem(raw_images)
                semantic = F.interpolate(
                    semantic,
                    size=detail.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                decoded = self.detail_fusion(torch.cat((semantic, detail), dim=1))
                decoded = self.refinement_decoder(decoded)

        similarity_decoded = F.interpolate(
            similarity, size=decoded.shape[-2:], mode="bilinear", align_corners=False
        )
        proposal_logits = self.proposal_head(decoded)
        verification_logits = self.verification_head(
            torch.cat((decoded, similarity_decoded), dim=1)
        )

        padded_size = prepared.shape[-2:]
        proposal_logits = F.interpolate(
            proposal_logits, size=padded_size, mode="bilinear", align_corners=False
        )
        verification_logits = F.interpolate(
            verification_logits, size=padded_size, mode="bilinear", align_corners=False
        )
        similarity = F.interpolate(
            similarity, size=padded_size, mode="bilinear", align_corners=False
        )
        if repetition_map is not None:
            repetition_map = F.interpolate(
                repetition_map,
                size=padded_size,
                mode="bilinear",
                align_corners=False,
            )
        if density_refinement_logits is not None:
            density_refinement_logits = F.interpolate(
                density_refinement_logits,
                size=padded_size,
                mode="bilinear",
                align_corners=False,
            )

        proposal_density = F.softplus(proposal_logits)
        verification = verification_logits.sigmoid()
        if self.architecture_version == "v1":
            verification_factor = verification
            count_correction = proposal_density.new_ones(
                proposal_density.shape[0], 1, 1, 1
            )
            spatial_correction = proposal_density.new_ones(proposal_density.shape)
        else:
            if self.architecture_version == "v2":
                # Unlike a sigmoid gate, this residual factor cannot erase
                # density mass; it can suppress or reinforce a response.
                verification_factor = 0.5 + verification
                spatial_correction = proposal_density.new_ones(proposal_density.shape)
            else:
                # v3 starts from the exact v1 density and learns bounded,
                # high-resolution residual corrections around that baseline.
                verification_factor = verification
                assert density_refinement_logits is not None
                spatial_correction = (
                    math.log(4.0) * density_refinement_logits.tanh()
                ).exp()
            pooled_features = F.adaptive_avg_pool2d(count_features, 1).flatten(1)
            log_count_correction = self.max_log_count_scale * torch.tanh(
                self.count_scale_head(torch.cat((pooled_features, query), dim=1))
            )
            count_correction = log_count_correction.exp().view(-1, 1, 1, 1)
        base_density = proposal_density * verification_factor
        density = base_density * spatial_correction * count_correction

        height, width = original_size
        density = density[..., :height, :width].squeeze(1)
        if not return_aux:
            return density

        return {
            "density": density,
            "proposal_density": proposal_density[..., :height, :width].squeeze(1),
            "verification": verification[..., :height, :width].squeeze(1),
            "verification_logits": verification_logits[..., :height, :width].squeeze(1),
            "verification_factor": verification_factor[..., :height, :width].squeeze(1),
            "base_density": base_density[..., :height, :width].squeeze(1),
            "count_correction": count_correction.flatten(),
            "spatial_correction": spatial_correction[..., :height, :width].squeeze(1),
            "similarity": similarity[..., :height, :width].squeeze(1),
            "repetition": (
                repetition_map[..., :height, :width].squeeze(1)
                if repetition_map is not None
                else similarity.new_zeros(similarity.shape[0], height, width)
            ),
            "level_weights": level_weights,
        }
