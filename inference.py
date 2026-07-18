"""Simple reference-free inference for the UPCount v6 model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from models_counting_network import CountingNetwork
from util.evaluation import predict_density_sliding_window


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Count objects in one RGB image with UPCount")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output_dir", default=Path("outputs/demo"), type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--window_size", default=384, type=int)
    parser.add_argument("--window_stride", default=128, type=int)
    parser.add_argument("--density_scale", default=60.0, type=float)
    parser.add_argument("--overlay_alpha", default=0.55, type=float)
    return parser


def _saved_arg(checkpoint: dict[str, Any], name: str, default: Any) -> Any:
    saved = checkpoint.get("args")
    if isinstance(saved, dict):
        return saved.get(name, default)
    return getattr(saved, name, default) if saved is not None else default


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[CountingNetwork, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must be a state dictionary or contain a 'model' key")

    architecture = _saved_arg(checkpoint, "architecture_version", "v6")
    if architecture != "v6":
        raise ValueError(f"expected a v6 checkpoint, found {architecture!r}")

    decoder_dim = int(_saved_arg(checkpoint, "decoder_dim", 192))
    layer_indices = _saved_arg(checkpoint, "dino_layer_indices", None)
    query_mode = _saved_arg(checkpoint, "v4_query_mode", "repetition")
    repetition_topk = int(_saved_arg(checkpoint, "repetition_topk", 16))
    model = CountingNetwork(
        decoder_dim=decoder_dim,
        architecture_version="v6",
        dino_layer_indices=layer_indices,
        v4_query_mode=query_mode,
        repetition_topk=repetition_topk,
        enable_text_conditioning=False,
        freeze_backbone=True,
        mae_pretrained=False,
    )
    state = checkpoint.get("model", checkpoint)
    if state and all(str(key).startswith("module.") for key in state):
        state = {str(key)[7:]: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    metadata = {
        "architecture_version": architecture,
        "decoder_dim": decoder_dim,
        "layer_indices": list(layer_indices or (2, 5, 8, 11)),
        "query_mode": query_mode,
        "repetition_topk": repetition_topk,
        "checkpoint_epoch": checkpoint.get("epoch"),
    }
    return model, metadata


def _save_density(path: Path, density: np.ndarray, vmax: float) -> None:
    rgba = plt.get_cmap("magma")(np.clip(density / vmax, 0.0, 1.0))
    Image.fromarray((rgba[..., :3] * 255.0).round().astype(np.uint8)).save(path)


@torch.inference_mode()
def main(args: argparse.Namespace) -> None:
    if args.window_size <= 0 or args.window_stride <= 0:
        raise ValueError("window_size and window_stride must be positive")
    if args.density_scale <= 0:
        raise ValueError("density_scale must be positive")
    if not 0.0 <= args.overlay_alpha <= 1.0:
        raise ValueError("overlay_alpha must be between zero and one")

    device = torch.device(args.device)
    model, metadata = load_model(args.checkpoint, device)
    source = Image.open(args.image).convert("RGB")
    source_array = np.asarray(source, dtype=np.float32) / 255.0
    image = torch.from_numpy(source_array).permute(2, 0, 1).unsqueeze(0).to(device)
    density_tensor = predict_density_sliding_window(
        model,
        image,
        None,
        window_size=args.window_size,
        stride=args.window_stride,
    )
    density = density_tensor.cpu().numpy()
    predicted_count = float(density.sum() / args.density_scale)
    vmax = max(float(np.quantile(density, 0.995)), 1e-8)
    heatmap = plt.get_cmap("magma")(np.clip(density / vmax, 0.0, 1.0))[..., :3]
    overlay = (1.0 - args.overlay_alpha) * source_array + args.overlay_alpha * heatmap

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source.save(args.output_dir / "source.png")
    _save_density(args.output_dir / "predicted_density.png", density, vmax)
    Image.fromarray((np.clip(overlay, 0.0, 1.0) * 255.0).round().astype(np.uint8)).save(
        args.output_dir / "density_overlay.png"
    )
    result = {
        "image": str(args.image.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "predicted_count": predicted_count,
        "density_sum": float(density.sum()),
        "density_scale": args.density_scale,
        "height": source.height,
        "width": source.width,
        "window_size": args.window_size,
        "window_stride": args.window_stride,
        **metadata,
    }
    (args.output_dir / "prediction.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(get_args_parser().parse_args())

