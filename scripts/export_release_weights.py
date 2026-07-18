"""Strip optimizer state and create model-only UPCount release checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


SPECS = {
    "mae": {
        "filename": "upcount_mae_fsc147_pretrain_epoch500.pth",
        "epoch": 500,
        "selection": "final FSC-147 MAE pretraining epoch",
    },
    "fsc147": {
        "filename": "upcount_fsc147_best_epoch432.pth",
        "epoch": 432,
        "selection": "lowest FSC-147 validation MAE",
    },
    "carpk": {
        "filename": "upcount_carpk_best_epoch816.pth",
        "epoch": 816,
        "selection": "lowest held-out CARPK train-split validation MAE",
    },
}

PUBLIC_ARGUMENTS = (
    "architecture_version",
    "decoder_dim",
    "dino_layer_indices",
    "v4_query_mode",
    "repetition_topk",
    "epochs",
    "batch_size",
    "accum_iter",
    "mask_ratio",
    "norm_pix_loss",
    "lr",
    "blr",
    "min_lr",
    "warmup_epochs",
    "weight_decay",
    "count_loss_weight",
    "verification_loss_weight",
    "log_count_loss_weight",
    "positive_density_weight",
    "density_scale",
    "crop_size",
    "window_size",
    "window_stride",
    "seed",
)


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Export model-only UPCount checkpoints")
    parser.add_argument("--mae", required=True, type=Path)
    parser.add_argument("--fsc147", required=True, type=Path)
    parser.add_argument("--carpk", required=True, type=Path)
    parser.add_argument("--output_dir", default=Path("weights"), type=Path)
    return parser


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, argparse.Namespace):
        return {key: _json_safe(item) for key, item in vars(value).items()}
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def export(source: Path, destination: Path, spec: dict[str, Any]) -> dict[str, Any]:
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise TypeError(f"{source} does not contain a model state dictionary")
    saved_args = _json_safe(checkpoint.get("args", {}))
    public_args = {
        name: saved_args[name]
        for name in PUBLIC_ARGUMENTS
        if isinstance(saved_args, dict) and name in saved_args
    }
    release = {
        "model": checkpoint["model"],
        "epoch": checkpoint.get("epoch"),
        "display_epoch": spec["epoch"],
        "args": public_args,
        "validation": _json_safe(checkpoint.get("validation")),
        "selection": spec["selection"],
        "format_version": 1,
    }
    torch.save(release, destination)
    return {
        "filename": destination.name,
        "size_bytes": destination.stat().st_size,
        "sha256": sha256(destination),
        "source": str(source),
    }


def main(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for key in ("mae", "fsc147", "carpk"):
        source = getattr(args, key)
        destination = args.output_dir / SPECS[key]["filename"]
        report[key] = export(source, destination, SPECS[key])
        print(json.dumps({key: report[key]}, indent=2))
    (args.output_dir / "export_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main(get_args_parser().parse_args())
