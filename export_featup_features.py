"""Export interpretable intermediate features from a trained v6 FeatUp model.

The exporter uses forward hooks, so it does not change the model's inference
path or checkpoint format.  For every input image it saves the projected
24x24 ViT feature, a bicubic baseline, the learned 96x96 JBU feature, the JBU
detail gain, DPT/refined features, and the final density prediction.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from models_counting_network import CountingNetwork


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "Export learned FeatUp/JBU feature visualizations"
    )
    parser.add_argument(
        "images",
        nargs="+",
        type=Path,
        help="one or more FSC-147/CARPK image files",
    )
    parser.add_argument("--resume", required=True, type=Path)
    parser.add_argument("--output_dir", default="./evaluation/featup-features", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--image_size",
        default=384,
        type=int,
        help="square inference size used by v6; zero preserves the source size",
    )
    parser.add_argument("--density_scale", default=60.0, type=float)
    parser.add_argument(
        "--architecture_version",
        choices=("v6",),
        default="v6",
        help="the current FeatUp architecture",
    )
    parser.add_argument("--decoder_dim", default=192, type=int)
    parser.add_argument("--dino_layer_indices", nargs=4, type=int, default=None)
    parser.add_argument(
        "--v4_query_mode",
        choices=("image_film", "none", "repetition"),
        default="repetition",
    )
    parser.add_argument("--repetition_topk", default=16, type=int)
    return parser


def _checkpoint_argument(checkpoint: dict[str, Any], name: str, default: Any) -> Any:
    saved = checkpoint.get("args")
    if saved is None:
        return default
    if isinstance(saved, dict):
        return saved.get(name, default)
    return getattr(saved, name, default)


def _load_model(args: argparse.Namespace, device: torch.device) -> tuple[CountingNetwork, dict]:
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must contain a state dictionary")

    architecture_version = _checkpoint_argument(
        checkpoint, "architecture_version", args.architecture_version
    )
    if architecture_version != "v6":
        raise ValueError(
            f"expected a v6 checkpoint with FeatUp, found {architecture_version!r}"
        )
    decoder_dim = int(_checkpoint_argument(checkpoint, "decoder_dim", args.decoder_dim))
    layer_indices = _checkpoint_argument(
        checkpoint, "dino_layer_indices", args.dino_layer_indices
    )
    query_mode = _checkpoint_argument(checkpoint, "v4_query_mode", args.v4_query_mode)
    repetition_topk = int(
        _checkpoint_argument(checkpoint, "repetition_topk", args.repetition_topk)
    )

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
        "checkpoint": str(args.resume.resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "architecture_version": architecture_version,
        "decoder_dim": decoder_dim,
        "layer_indices": list(layer_indices or (2, 5, 8, 11)),
        "v4_query_mode": query_mode,
        "repetition_topk": repetition_topk,
    }
    return model, metadata


def _image_tensor(path: Path, image_size: int) -> tuple[Image.Image, torch.Tensor]:
    image = Image.open(path).convert("RGB")
    if image_size > 0:
        image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return image, tensor


def _fit_shared_pca(features: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    channels = {int(feature.shape[0]) for feature in features}
    if len(channels) != 1:
        raise ValueError("shared PCA features must have the same channel dimension")
    samples = torch.cat(
        [feature.float().flatten(1).transpose(0, 1) for feature in features], dim=0
    )
    mean = samples.mean(dim=0, keepdim=True)
    centered = samples - mean
    covariance = centered.transpose(0, 1) @ centered
    covariance = covariance / max(centered.shape[0] - 1, 1)
    _, eigenvectors = torch.linalg.eigh(covariance)
    basis = eigenvectors[:, -3:].flip(dims=(1,))
    # Resolve the arbitrary eigenvector sign for reproducible colors.
    for component in range(basis.shape[1]):
        vector = basis[:, component]
        pivot = vector.abs().argmax()
        if vector[pivot] < 0:
            basis[:, component] = -vector
    return mean, basis


def _project_shared_pca(
    features: Sequence[torch.Tensor], mean: torch.Tensor, basis: torch.Tensor
) -> list[np.ndarray]:
    projected = []
    flat_outputs = []
    for feature in features:
        height, width = feature.shape[-2:]
        samples = feature.float().flatten(1).transpose(0, 1)
        values = ((samples - mean) @ basis).reshape(height, width, 3)
        flat_outputs.append(values)
    all_values = torch.cat([values.reshape(-1, 3) for values in flat_outputs], dim=0)
    lower = torch.quantile(all_values, 0.01, dim=0)
    upper = torch.quantile(all_values, 0.99, dim=0)
    scale = (upper - lower).clamp_min(1e-6)
    for values in flat_outputs:
        rgb = ((values - lower) / scale).clamp(0, 1)
        projected.append(rgb.numpy())
    return projected


def _heatmap(values: torch.Tensor, cmap: str = "magma") -> np.ndarray:
    values = values.float().squeeze().cpu()
    lower = torch.quantile(values, 0.01)
    upper = torch.quantile(values, 0.99)
    normalized = ((values - lower) / (upper - lower).clamp_min(1e-6)).clamp(0, 1)
    return plt.get_cmap(cmap)(normalized.numpy())[..., :3]


def _save_rgb(path: Path, values: np.ndarray) -> None:
    encoded = (np.clip(values, 0, 1) * 255.0).round().astype(np.uint8)
    Image.fromarray(encoded).save(path)


def _safe_stem(path: Path, index: int) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("._") or "image"
    return f"{index:02d}_{stem}"


@torch.inference_mode()
def _export_image(
    model: CountingNetwork,
    image_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
    model_metadata: dict[str, Any],
) -> dict[str, Any]:
    pil_image, image = _image_tensor(image_path, args.image_size)
    image = image.to(device=device, dtype=torch.float32)
    captures: dict[str, torch.Tensor] = {}

    def capture_source_projection(_module, _inputs, output):
        captures["projected_vit"] = output.detach().cpu()[0]

    def capture_feature_fusion(_module, _inputs, output):
        dpt, level_weights, semantic_anchor = output
        captures["dpt"] = dpt.detach().cpu()[0]
        captures["level_weights"] = level_weights.detach().cpu()
        captures["semantic_anchor"] = semantic_anchor.detach().cpu()[0]

    def capture_jbu(_module, _inputs, output):
        captures["jbu"] = output.detach().cpu()[0]

    def capture_featup_residual(_module, _inputs, output):
        captures["featup_residual"] = output.detach().cpu()[0]

    def capture_detail(_module, _inputs, output):
        captures["rgb_guidance"] = output.detach().cpu()[0]

    def capture_refined(_module, _inputs, output):
        captures["refined"] = output.detach().cpu()[0]

    def capture_proposal_feature(_module, _inputs, output):
        captures["proposal_feature"] = output.detach().cpu()[0]

    def capture_verifier_feature(_module, _inputs, output):
        captures["verifier_feature"] = output.detach().cpu()[0]

    hooks = [
        model.feature_fusion.register_forward_hook(capture_feature_fusion),
        model.featup_adapter.source_projection.register_forward_hook(
            capture_source_projection
        ),
        model.featup_adapter.register_forward_hook(capture_jbu),
        model.featup_fusion.register_forward_hook(capture_featup_residual),
        model.detail_stem.register_forward_hook(capture_detail),
        model.detail_fusion.register_forward_hook(capture_refined),
        model.proposal_head[1].register_forward_hook(capture_proposal_feature),
        model.verification_head[1].register_forward_hook(capture_verifier_feature),
    ]
    try:
        output = model(image, return_aux=True)
    finally:
        for hook in hooks:
            hook.remove()

    required = {
        "projected_vit",
        "dpt",
        "jbu",
        "featup_residual",
        "rgb_guidance",
        "refined",
        "proposal_feature",
        "verifier_feature",
    }
    missing = sorted(required - captures.keys())
    if missing:
        raise RuntimeError(f"failed to capture intermediate features: {missing}")

    projected = captures["projected_vit"]
    jbu = captures["jbu"]
    bicubic = F.interpolate(
        projected.unsqueeze(0),
        size=jbu.shape[-2:],
        mode="bicubic",
        align_corners=False,
    )[0]
    pca_mean, pca_basis = _fit_shared_pca((projected, bicubic, jbu))
    projected_rgb, bicubic_rgb, jbu_rgb = _project_shared_pca(
        (projected, bicubic, jbu), pca_mean, pca_basis
    )

    dpt = captures["dpt"]
    residual = captures["featup_residual"]
    if dpt.shape[-2:] != residual.shape[-2:]:
        dpt = F.interpolate(
            dpt.unsqueeze(0),
            size=residual.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[0]
    featup_fused = dpt + residual
    refined = captures["refined"]
    refined_mean, refined_basis = _fit_shared_pca((dpt, featup_fused, refined))
    dpt_rgb, fused_rgb, refined_rgb = _project_shared_pca(
        (dpt, featup_fused, refined), refined_mean, refined_basis
    )

    gain = (jbu - bicubic).square().mean(dim=0).sqrt()
    gain_rgb = _heatmap(gain)
    density = output["density"].detach().cpu()[0]
    density_rgb = _heatmap(density, "inferno")
    proposal_density = output["proposal_density"].detach().cpu()[0]
    proposal_density_rgb = _heatmap(proposal_density, "inferno")
    verification = output["verification"].detach().cpu()[0]
    verification_rgb = _heatmap(verification, "viridis")
    verified_density = output["base_density"].detach().cpu()[0]
    verified_density_rgb = _heatmap(verified_density, "inferno")
    repetition = output["repetition"].detach().cpu()[0]
    repetition_rgb = _heatmap(repetition, "magma")
    proposal_feature = captures["proposal_feature"]
    proposal_mean, proposal_basis = _fit_shared_pca((proposal_feature,))
    proposal_feature_rgb = _project_shared_pca(
        (proposal_feature,), proposal_mean, proposal_basis
    )[0]
    verifier_feature = captures["verifier_feature"]
    verifier_mean, verifier_basis = _fit_shared_pca((verifier_feature,))
    verifier_feature_rgb = _project_shared_pca(
        (verifier_feature,), verifier_mean, verifier_basis
    )[0]
    source_rgb = np.asarray(pil_image, dtype=np.float32) / 255.0
    density_for_overlay = np.asarray(
        Image.fromarray((density_rgb * 255).astype(np.uint8)).resize(
            pil_image.size, Image.Resampling.BICUBIC
        ),
        dtype=np.float32,
    ) / 255.0
    overlay = 0.58 * source_rgb + 0.42 * density_for_overlay
    predicted_count = float(density.sum().item() / args.density_scale)

    output_dir.mkdir(parents=True, exist_ok=True)
    image_files = {
        "source": source_rgb,
        "projected_vit_24x24_pca": projected_rgb,
        "bicubic_96x96_pca": bicubic_rgb,
        "featup_jbu_96x96_pca": jbu_rgb,
        "featup_detail_gain": gain_rgb,
        "dpt_96x96_pca": dpt_rgb,
        "dpt_featup_fused_96x96_pca": fused_rgb,
        "refined_96x96_pca": refined_rgb,
        "predicted_density": density_rgb,
        "density_overlay": overlay,
        "proposal_feature_96x96_pca": proposal_feature_rgb,
        "proposal_density": proposal_density_rgb,
        "verifier_feature_96x96_pca": verifier_feature_rgb,
        "verification_confidence": verification_rgb,
        "proposal_verified_density": verified_density_rgb,
        "repetition_similarity": repetition_rgb,
    }
    for name, values in image_files.items():
        _save_rgb(output_dir / f"{name}.png", values)

    titles = [
        "Input image",
        f"Projected ViT\n{projected.shape[-2]}x{projected.shape[-1]}",
        f"Bicubic baseline\n{bicubic.shape[-2]}x{bicubic.shape[-1]}",
        f"FeatUp JBU\n{jbu.shape[-2]}x{jbu.shape[-1]}",
        "JBU detail gain",
        "DPT pyramid",
        "DPT + FeatUp",
        "RGB-refined feature",
        "Density response",
        "Density overlay",
    ]
    panels = [
        source_rgb,
        projected_rgb,
        bicubic_rgb,
        jbu_rgb,
        gain_rgb,
        dpt_rgb,
        fused_rgb,
        refined_rgb,
        density_rgb,
        overlay,
    ]
    figure, axes = plt.subplots(2, 5, figsize=(18, 7.2))
    for axis, panel, title in zip(axes.flat, panels, titles):
        interpolation = "nearest" if "Projected ViT" in title else "bilinear"
        axis.imshow(panel, interpolation=interpolation)
        axis.set_title(title, fontsize=11)
        axis.axis("off")
    figure.suptitle(
        f"FeatUp feature extraction: {image_path.name}", fontsize=14, fontweight="bold"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(output_dir / "featup_feature_panel.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    proposal_verifier_titles = [
        "Input image",
        "Refined counting feature\n64x96x96",
        "Proposal-head feature\n64x96x96",
        "Proposal density\n1x384x384",
        "Repetition similarity",
        "Verifier-head feature\n64x96x96",
        "Verification confidence\n1x384x384",
        "Proposal x verification\n1x384x384",
    ]
    proposal_verifier_panels = [
        source_rgb,
        refined_rgb,
        proposal_feature_rgb,
        proposal_density_rgb,
        repetition_rgb,
        verifier_feature_rgb,
        verification_rgb,
        verified_density_rgb,
    ]
    figure, axes = plt.subplots(2, 4, figsize=(15, 7.4))
    for axis, panel, title in zip(
        axes.flat, proposal_verifier_panels, proposal_verifier_titles
    ):
        axis.imshow(panel, interpolation="bilinear")
        axis.set_title(title, fontsize=11)
        axis.axis("off")
    figure.suptitle(
        f"Proposal-verification features: {image_path.name}",
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(
        output_dir / "proposal_verifier_feature_panel.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)

    metadata = {
        **model_metadata,
        "image": str(image_path.resolve()),
        "resized_image_size": list(image.shape[-2:]),
        "predicted_count": predicted_count,
        "density_scale": args.density_scale,
        "level_weights": captures["level_weights"].tolist(),
        "tensor_shapes": {
            key: list(value.shape)
            for key, value in captures.items()
            if key != "level_weights"
        },
        "files": {name: f"{name}.png" for name in image_files},
        "panel": "featup_feature_panel.png",
        "proposal_verifier_panel": "proposal_verifier_feature_panel.png",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def main(args: argparse.Namespace) -> None:
    if args.image_size < 0:
        raise ValueError("--image_size must be non-negative")
    if args.density_scale <= 0:
        raise ValueError("--density_scale must be positive")
    missing = [path for path in args.images if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"input images not found: {missing}")
    device = torch.device(args.device)
    model, model_metadata = _load_model(args, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for index, image_path in enumerate(args.images):
        sample_dir = args.output_dir / _safe_stem(image_path, index)
        metadata = _export_image(
            model,
            image_path,
            sample_dir,
            args,
            device,
            model_metadata,
        )
        manifest.append(
            {
                "image": metadata["image"],
                "output_dir": str(sample_dir.resolve()),
                "predicted_count": metadata["predicted_count"],
            }
        )
        print(
            f"[{index + 1}/{len(args.images)}] {image_path.name}: "
            f"count={metadata['predicted_count']:.2f} -> {sample_dir}",
            flush=True,
        )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main(get_args_parser().parse_args())
