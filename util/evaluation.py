"""Shared FSC-147 evaluation helpers."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn


def sliding_window_starts(length: int, window_size: int, stride: int) -> list[int]:
    """Return deterministic starts that cover an axis, including its far edge."""

    if length <= 0:
        raise ValueError("length must be positive")
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")
    if length <= window_size:
        return [0]

    starts = list(range(0, length - window_size + 1, stride))
    final_start = length - window_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


@torch.inference_mode()
def predict_density_sliding_window(
    model: nn.Module,
    image: Tensor,
    query: Optional[Tensor] = None,
    *,
    window_size: int = 384,
    stride: int = 128,
) -> Tensor:
    """Predict one full density map using overlap-averaged 2-D tiles.

    The model was trained on 384x384 crops. Evaluation images may be wider or
    taller, so every pixel is covered by at least one tile. Uniform overlap
    averaging prevents density from being counted multiple times.
    """

    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("evaluation expects one image with shape [1,3,H,W]")

    device = image.device
    height, width = image.shape[-2:]
    top_starts = sliding_window_starts(height, window_size, stride)
    left_starts = sliding_window_starts(width, window_size, stride)
    density_sum = torch.zeros((height, width), device=device, dtype=torch.float32)
    weights = torch.zeros_like(density_sum)

    for top in top_starts:
        bottom = min(top + window_size, height)
        for left in left_starts:
            right = min(left + window_size, width)
            crop = image[..., top:bottom, left:right]
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                prediction = model(crop, query).squeeze(0)
            density_sum[top:bottom, left:right] += prediction.float()
            weights[top:bottom, left:right] += 1.0

    if torch.any(weights == 0):
        raise RuntimeError("sliding-window inference left uncovered pixels")
    return density_sum / weights

