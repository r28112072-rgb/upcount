"""Complete FSC-147 evaluation with quantitative and qualitative outputs."""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import os
import textwrap
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open_clip
from PIL import Image
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
from torchvision import transforms

from models_counting_network import (
    CountingNetwork,
    DEFAULT_DINOV2_REPO,
    IMAGE_ONLY_ARCHITECTURES,
)
from util.evaluation import predict_density_sliding_window
from util.FSC147 import TTensor
import util.misc as misc


def get_args_parser():
    parser = argparse.ArgumentParser(
        "Evaluate spatially-aware class-agnostic counting on FSC-147"
    )
    parser.add_argument(
        "--data_split",
        choices=("val", "test"),
        default="val",
        help="official FSC-147 split to evaluate",
    )
    parser.add_argument("--output_dir", default="./test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backbone_name", default="dinov2_vitb14_reg")
    parser.add_argument("--backbone_repo", default=DEFAULT_DINOV2_REPO)
    parser.add_argument("--decoder_dim", default=192, type=int)
    parser.add_argument(
        "--architecture_version",
        choices=("v1", "v2", "v3", "v4", "v5", "v6"),
        default="v6",
        help="must match the architecture used to create the checkpoint",
    )
    parser.add_argument(
        "--dino_layer_indices",
        nargs=4,
        type=int,
        default=None,
        metavar=("L1", "L2", "L3", "L4"),
        help="explicit zero-based DINO blocks; v4 defaults to 2 5 8 11 for ViT-B",
    )
    parser.add_argument(
        "--v4_query_mode",
        choices=("image_film", "none", "repetition"),
        default="repetition",
        help="must match the v4 ablation used to train the checkpoint",
    )
    parser.add_argument("--repetition_topk", default=16, type=int)
    parser.add_argument(
        "--mae_checkpoint",
        default="",
        help="optional MAE/CounTR encoder checkpoint when evaluating untrained v6",
    )
    parser.add_argument(
        "--disable_text_conditioning",
        action="store_true",
        help="evaluate the reference-less image-query mode",
    )
    parser.add_argument(
        "--checkpoint_without_text_encoder",
        action="store_true",
        help="construct no text tower for a strict reference-only checkpoint load",
    )
    parser.add_argument(
        "--resume",
        default="",
        help="trained modern-architecture checkpoint (required by default)",
    )
    parser.add_argument(
        "--allow_untrained",
        action="store_true",
        help="explicitly permit evaluation without a checkpoint for pipeline debugging",
    )
    parser.add_argument(
        "--allow_partial_checkpoint",
        action="store_true",
        help="permit missing or shape-mismatched checkpoint tensors",
    )
    parser.add_argument("--window_size", default=384, type=int)
    parser.add_argument("--window_stride", default=128, type=int)
    parser.add_argument(
        "--density_scale",
        default=60.0,
        type=float,
        help="density integral divisor used by FSC-147 preprocessing",
    )
    parser.add_argument(
        "--num_visualizations",
        default=6,
        type=int,
        help="number of error-percentile qualitative examples to render",
    )
    parser.add_argument(
        "--save_visualizations",
        action="store_true",
        help="deprecated alias that ensures six qualitative examples are rendered",
    )
    parser.add_argument(
        "--max_images",
        default=0,
        type=int,
        help="stop after N images for debugging; zero evaluates the full split",
    )
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--pin_mem", action="store_false")
    parser.add_argument(
        "--img_dir",
        default="./data/FSC147/images_384_VarV2",
    )
    parser.add_argument(
        "--FSC147_anno_file",
        default="./data/FSC147/annotation_FSC147_384.json",
    )
    parser.add_argument("--FSC147_D_anno_file", default="./FSC-147-D.json")
    parser.add_argument(
        "--data_split_file",
        default="./data/FSC147/Train_Test_Val_FSC_147.json",
    )
    return parser


class TestData(Dataset):
    def __init__(self, args):
        self.img_dir = args.img_dir
        with open(args.data_split_file, encoding="utf-8") as handle:
            self.image_ids = json.load(handle)[args.data_split]
        with open(args.FSC147_anno_file, encoding="utf-8") as handle:
            self.annotations = json.load(handle)
        with open(args.FSC147_D_anno_file, encoding="utf-8") as handle:
            self.text_annotations = json.load(handle)
        self.clip_tokenizer = open_clip.get_tokenizer("ViT-B-16")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        description = self.text_annotations[image_id]["text_description"]
        tokens = self.clip_tokenizer(description).squeeze(0)
        points = np.asarray(self.annotations[image_id]["points"], dtype=np.float32)

        image = Image.open(os.path.join(self.img_dir, image_id)).convert("RGB")
        width, height = image.size
        new_height = max(16, 16 * int(height / 16))
        new_width = max(16, 16 * int(width / 16))
        image = transforms.Resize((new_height, new_width))(image)
        image = TTensor(image)

        points[:, 0] *= new_width / width
        points[:, 1] *= new_height / height
        return image, torch.from_numpy(points), tokens, image_id, description


def _write_predictions(output_dir: Path, records: list[dict]) -> None:
    fieldnames = [
        "dataset_index",
        "split",
        "image_id",
        "description",
        "predicted_count",
        "target_count",
        "absolute_error",
        "squared_error",
        "height",
        "width",
    ]
    with (output_dir / "predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _select_error_percentiles(records: list[dict], count: int) -> list[tuple[str, dict]]:
    if count <= 0 or not records:
        return []
    ordered = sorted(records, key=lambda item: item["absolute_error"])
    count = min(count, len(ordered))
    ranks = np.linspace(0, len(ordered) - 1, count).round().astype(int)
    selected = []
    for rank in dict.fromkeys(ranks.tolist()):
        percentile = 100.0 * rank / max(len(ordered) - 1, 1)
        selected.append((f"error p{percentile:.0f}", ordered[rank]))
    return selected


def _save_qualitative_grid(
    model,
    dataset,
    selected,
    args,
    device,
    output_dir: Path,
) -> None:
    if not selected:
        return

    rows = len(selected)
    figure, axes = plt.subplots(rows, 3, figsize=(13, 3.6 * rows), squeeze=False)
    selected_metadata = []
    for row, (selection_label, record) in enumerate(selected):
        image, points, tokens, image_id, description = dataset[
            record["dataset_index"]
        ]
        image_batch = image.unsqueeze(0).to(device, dtype=torch.float32)
        tokens = tokens.unsqueeze(0).to(device)
        query = (
            None
            if args.disable_text_conditioning
            or args.architecture_version in IMAGE_ONLY_ARCHITECTURES
            else tokens
        )
        density = predict_density_sliding_window(
            model,
            image_batch,
            query,
            window_size=args.window_size,
            stride=args.window_stride,
        ).cpu().numpy()
        source = image.permute(1, 2, 0).numpy().clip(0.0, 1.0)
        points_np = points.numpy()
        vmax = max(float(np.quantile(density, 0.995)), 1e-8)

        axes[row, 0].imshow(source)
        axes[row, 0].scatter(
            points_np[:, 0],
            points_np[:, 1],
            s=8,
            facecolors="none",
            edgecolors="#00ffff",
            linewidths=0.7,
        )
        axes[row, 0].set_title("Source + ground-truth points")
        axes[row, 1].imshow(density, cmap="magma", vmin=0.0, vmax=vmax)
        axes[row, 1].set_title("Predicted density")
        axes[row, 2].imshow(source)
        axes[row, 2].imshow(
            density, cmap="magma", alpha=0.55, vmin=0.0, vmax=vmax
        )
        axes[row, 2].set_title("Density overlay")
        for axis in axes[row]:
            axis.axis("off")

        caption = (
            f"{selection_label} · {image_id} · GT {record['target_count']:.0f} · "
            f"Pred {record['predicted_count']:.2f} · AE {record['absolute_error']:.2f}\n"
            + textwrap.shorten(description, width=100, placeholder="…")
        )
        axes[row, 0].text(
            0.0,
            -0.12,
            caption,
            transform=axes[row, 0].transAxes,
            ha="left",
            va="top",
            fontsize=9,
        )
        selected_metadata.append({"selection": selection_label, **record})

    figure.tight_layout(h_pad=2.8)
    figure.savefig(output_dir / "qualitative_examples.png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    with (output_dir / "qualitative_examples.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(selected_metadata, handle, indent=2)


def main(args):
    if not args.resume and not args.allow_untrained:
        raise ValueError(
            "--resume is required for a reportable evaluation; use --allow_untrained "
            "only for pipeline debugging"
        )
    if args.density_scale <= 0:
        raise ValueError("--density_scale must be positive")
    if args.checkpoint_without_text_encoder and not args.disable_text_conditioning:
        raise ValueError(
            "--checkpoint_without_text_encoder requires --disable_text_conditioning"
        )
    if args.save_visualizations:
        args.num_visualizations = max(args.num_visualizations, 6)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    cudnn.benchmark = False
    torch.manual_seed(0)
    np.random.seed(0)

    dataset = TestData(args)
    sampler = torch.utils.data.SequentialSampler(dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    model = CountingNetwork(
        backbone_name=args.backbone_name,
        backbone_repo=args.backbone_repo,
        decoder_dim=args.decoder_dim,
        freeze_backbone=True,
        # Reference-less checkpoints may have been trained either with query
        # dropout (and therefore contain a text tower) or as image-only models.
        enable_text_conditioning=(
            args.architecture_version not in IMAGE_ONLY_ARCHITECTURES
            and not args.checkpoint_without_text_encoder
        ),
        text_dropout_p=0.0,
        architecture_version=args.architecture_version,
        dino_layer_indices=args.dino_layer_indices,
        v4_query_mode=args.v4_query_mode,
        repetition_topk=args.repetition_topk,
        mae_checkpoint=args.mae_checkpoint or None,
        # A reportable evaluation checkpoint contains the complete encoder.
        mae_pretrained=not bool(args.resume),
    ).to(device)
    if args.resume:
        misc.load_model_FSC(
            args,
            model,
            strict=not args.allow_partial_checkpoint,
        )
    model.eval()

    records = []
    start_time = time.time()
    limit = min(args.max_images or len(dataset), len(dataset))
    for index, (image, points, tokens, image_id, description) in enumerate(data_loader):
        if index >= limit:
            break
        image = image.to(device, non_blocking=True, dtype=torch.float32)
        tokens = tokens.to(device, non_blocking=True)
        query = None if args.disable_text_conditioning else tokens
        density = predict_density_sliding_window(
            model,
            image,
            query,
            window_size=args.window_size,
            stride=args.window_stride,
        )
        predicted_count = float(density.sum().item() / args.density_scale)
        target_count = int(points.shape[1])
        absolute_error = abs(predicted_count - target_count)
        records.append(
            {
                "dataset_index": index,
                "split": args.data_split,
                "image_id": image_id[0],
                "description": description[0],
                "predicted_count": predicted_count,
                "target_count": target_count,
                "absolute_error": absolute_error,
                "squared_error": absolute_error**2,
                "height": image.shape[-2],
                "width": image.shape[-1],
            }
        )
        if index % 50 == 0 or index + 1 == limit:
            elapsed = time.time() - start_time
            print(
                f"[{index + 1}/{limit}] {image_id[0]}: pred={predicted_count:.2f} "
                f"gt={target_count} ae={absolute_error:.2f} "
                f"({(index + 1) / max(elapsed, 1e-6):.2f} images/s)",
                flush=True,
            )

    if not records:
        raise RuntimeError("no images were evaluated")
    elapsed = time.time() - start_time
    absolute_errors = np.asarray([row["absolute_error"] for row in records])
    squared_errors = np.asarray([row["squared_error"] for row in records])
    predicted_counts = np.asarray([row["predicted_count"] for row in records])
    target_counts = np.asarray([row["target_count"] for row in records])
    metrics = {
        "split": args.data_split,
        "mode": "reference-less" if args.disable_text_conditioning else "text-guided",
        "architecture_version": args.architecture_version,
        "dino_layer_indices": (
            args.dino_layer_indices
            if args.dino_layer_indices is not None
            else (
                [2, 5, 8, 11]
                if args.architecture_version in IMAGE_ONLY_ARCHITECTURES
                else None
            )
        ),
        "v4_query_mode": (
            args.v4_query_mode
            if args.architecture_version in IMAGE_ONLY_ARCHITECTURES
            else None
        ),
        "checkpoint": args.resume or None,
        "images": len(records),
        "complete_split": len(records) == len(dataset),
        "MAE": float(absolute_errors.mean()),
        "RMSE": float(math.sqrt(squared_errors.mean())),
        "median_absolute_error": float(np.median(absolute_errors)),
        "p90_absolute_error": float(np.quantile(absolute_errors, 0.9)),
        "mean_predicted_count": float(predicted_counts.mean()),
        "mean_target_count": float(target_counts.mean()),
        "count_bias": float((predicted_counts - target_counts).mean()),
        "elapsed_seconds": elapsed,
        "images_per_second": len(records) / elapsed,
        "window_size": args.window_size,
        "window_stride": args.window_stride,
        "density_scale": args.density_scale,
    }
    _write_predictions(output_dir, records)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    selected = _select_error_percentiles(records, args.num_visualizations)
    _save_qualitative_grid(model, dataset, selected, args, device, output_dir)

    print(json.dumps(metrics, indent=2))
    print("Evaluation time {}".format(str(datetime.timedelta(seconds=int(elapsed)))))


if __name__ == "__main__":
    parsed_args = get_args_parser().parse_args()
    main(parsed_args)
