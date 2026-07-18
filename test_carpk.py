"""Evaluate reference-free UPCount on CARPK."""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from models_counting_network import CountingNetwork, DEFAULT_DINOV2_REPO
from util.CARPK import CARPKTestData
from util.evaluation import predict_density_sliding_window
import util.misc as misc


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Evaluate reference-free counting on CARPK")
    parser.add_argument("--data_dir", default="./data/CARPK/test")
    parser.add_argument("--output_dir", default="./evaluation/carpk-test")
    parser.add_argument("--resume", required=True)
    parser.add_argument(
        "--protocol",
        default="CARPK fine-tuned",
        help="human-readable training/evaluation protocol stored with the metrics",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backbone_name", default="dinov2_vitb14_reg")
    parser.add_argument("--backbone_repo", default=DEFAULT_DINOV2_REPO)
    parser.add_argument("--decoder_dim", default=192, type=int)
    parser.add_argument(
        "--architecture_version",
        choices=("v1", "v2", "v3", "v4", "v5", "v6"),
        default="v6",
    )
    parser.add_argument("--dino_layer_indices", nargs=4, type=int, default=None)
    parser.add_argument(
        "--v4_query_mode",
        choices=("image_film", "none", "repetition"),
        default="repetition",
    )
    parser.add_argument("--repetition_topk", default=16, type=int)
    parser.add_argument("--window_size", default=384, type=int)
    parser.add_argument("--window_stride", default=128, type=int)
    parser.add_argument("--density_scale", default=60.0, type=float)
    parser.add_argument(
        "--resize_height",
        default=0,
        type=int,
        help="resize while preserving aspect ratio; zero evaluates native resolution",
    )
    parser.add_argument("--num_visualizations", default=6, type=int)
    parser.add_argument("--max_images", default=0, type=int)
    return parser


def _write_predictions(output_dir: Path, records: list[dict]) -> None:
    with (output_dir / "predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _selected_percentiles(records: list[dict], count: int) -> list[tuple[str, dict]]:
    ordered = sorted(records, key=lambda item: item["absolute_error"])
    if count <= 0 or not ordered:
        return []
    ranks = np.linspace(0, len(ordered) - 1, min(count, len(ordered))).round()
    selected = []
    for index in dict.fromkeys(ranks.astype(int).tolist()):
        percentile = 100.0 * index / max(len(ordered) - 1, 1)
        selected.append((f"error p{percentile:.0f}", ordered[index]))
    return selected


def _save_qualitative_grid(model, dataset, records, args, device, output_dir):
    if not records:
        return
    figure, axes = plt.subplots(
        len(records), 3, figsize=(13, 3.6 * len(records)), squeeze=False
    )
    metadata = []
    for row, (selection, record) in enumerate(records):
        image, points, image_id = dataset[record["dataset_index"]]
        density = predict_density_sliding_window(
            model,
            image.unsqueeze(0).to(device),
            None,
            window_size=args.window_size,
            stride=args.window_stride,
        ).cpu().numpy()
        source = image.permute(1, 2, 0).numpy().clip(0.0, 1.0)
        points_array = points.numpy()
        vmax = max(float(np.quantile(density, 0.995)), 1e-8)
        axes[row, 0].imshow(source)
        axes[row, 0].scatter(
            points_array[:, 0],
            points_array[:, 1],
            s=7,
            facecolors="none",
            edgecolors="#00ffff",
            linewidths=0.6,
        )
        axes[row, 0].set_title("Source + CARPK box centers")
        axes[row, 1].imshow(density, cmap="magma", vmin=0.0, vmax=vmax)
        axes[row, 1].set_title("Predicted density")
        axes[row, 2].imshow(source)
        axes[row, 2].imshow(density, cmap="magma", alpha=0.55, vmin=0.0, vmax=vmax)
        axes[row, 2].set_title("Density overlay")
        for axis in axes[row]:
            axis.axis("off")
        axes[row, 0].text(
            0.0,
            -0.12,
            f"{selection} · {image_id} · GT {record['target_count']} · "
            f"Pred {record['predicted_count']:.2f} · "
            f"AE {record['absolute_error']:.2f}",
            transform=axes[row, 0].transAxes,
            ha="left",
            va="top",
            fontsize=9,
        )
        metadata.append({"selection": selection, **record})
    figure.tight_layout(h_pad=2.8)
    figure.savefig(output_dir / "qualitative_examples.png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    (output_dir / "qualitative_examples.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def main(args) -> None:
    if args.density_scale <= 0:
        raise ValueError("density_scale must be positive")
    if args.resize_height < 0:
        raise ValueError("resize_height must be non-negative")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    cudnn.benchmark = False
    torch.manual_seed(0)
    np.random.seed(0)

    dataset = CARPKTestData(args.data_dir, resize_height=args.resize_height)
    model = CountingNetwork(
        backbone_name=args.backbone_name,
        backbone_repo=args.backbone_repo,
        decoder_dim=args.decoder_dim,
        freeze_backbone=True,
        enable_text_conditioning=False,
        text_dropout_p=0.0,
        architecture_version=args.architecture_version,
        dino_layer_indices=args.dino_layer_indices,
        v4_query_mode=args.v4_query_mode,
        repetition_topk=args.repetition_topk,
        mae_pretrained=False,
    ).to(device)
    misc.load_model_FSC(args, model, strict=True)
    model.eval()

    records = []
    started = time.time()
    limit = min(args.max_images or len(dataset), len(dataset))
    for index in range(limit):
        image, points, image_id = dataset[index]
        image_batch = image.unsqueeze(0).to(device, dtype=torch.float32)
        density = predict_density_sliding_window(
            model,
            image_batch,
            None,
            window_size=args.window_size,
            stride=args.window_stride,
        )
        predicted_count = float(density.sum().item() / args.density_scale)
        target_count = len(points)
        absolute_error = abs(predicted_count - target_count)
        records.append(
            {
                "dataset_index": index,
                "image_id": image_id,
                "predicted_count": predicted_count,
                "target_count": target_count,
                "absolute_error": absolute_error,
                "squared_error": absolute_error**2,
                "height": image.shape[-2],
                "width": image.shape[-1],
            }
        )
        if index % 25 == 0 or index + 1 == limit:
            elapsed = time.time() - started
            print(
                f"[{index + 1}/{limit}] {image_id}: pred={predicted_count:.2f} "
                f"gt={target_count} ae={absolute_error:.2f} "
                f"({(index + 1) / max(elapsed, 1e-6):.2f} images/s)",
                flush=True,
            )

    if not records:
        raise RuntimeError("no CARPK images were evaluated")
    elapsed = time.time() - started
    errors = np.asarray([record["absolute_error"] for record in records])
    predictions = np.asarray([record["predicted_count"] for record in records])
    targets = np.asarray([record["target_count"] for record in records])
    metrics = {
        "dataset": "CARPK",
        "split": "test",
        "protocol": args.protocol,
        "mode": "reference-less",
        "architecture_version": args.architecture_version,
        "dino_layer_indices": (
            args.dino_layer_indices
            if args.dino_layer_indices is not None
            else (
                [2, 5, 8, 11]
                if args.architecture_version in {"v4", "v5", "v6"}
                else None
            )
        ),
        "v4_query_mode": (
            args.v4_query_mode
            if args.architecture_version in {"v4", "v5", "v6"}
            else None
        ),
        "checkpoint": args.resume,
        "data_source": dataset.source,
        "images": len(records),
        "complete_split": len(records) == len(dataset),
        "MAE": float(errors.mean()),
        "RMSE": float(math.sqrt(np.mean(errors**2))),
        "median_absolute_error": float(np.median(errors)),
        "p90_absolute_error": float(np.quantile(errors, 0.9)),
        "mean_predicted_count": float(predictions.mean()),
        "mean_target_count": float(targets.mean()),
        "count_bias": float((predictions - targets).mean()),
        "elapsed_seconds": elapsed,
        "images_per_second": len(records) / elapsed,
        "resize_height": args.resize_height,
        "window_size": args.window_size,
        "window_stride": args.window_stride,
        "density_scale": args.density_scale,
    }
    _write_predictions(output_dir, records)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    _save_qualitative_grid(
        model,
        dataset,
        _selected_percentiles(records, args.num_visualizations),
        args,
        device,
        output_dir,
    )
    print(json.dumps(metrics, indent=2))
    print(f"Evaluation time {datetime.timedelta(seconds=int(elapsed))}")


if __name__ == "__main__":
    main(get_args_parser().parse_args())
