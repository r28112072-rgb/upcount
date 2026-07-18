"""Fine-tune UPCount v6 on CARPK without touching its Test split."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from models_counting_network import CountingNetwork, DEFAULT_DINOV2_REPO
from util.CARPK import CARPKCropData, CARPKTestData
from util.evaluation import predict_density_sliding_window
import util.misc as misc


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Fine-tune UPCount v6 on the CARPK training split")
    parser.add_argument("--data_dir", default="./data/CARPK/train")
    parser.add_argument("--output_dir", default="./outputs/carpk-finetuned")
    parser.add_argument(
        "--init_checkpoint",
        default="",
        help="strict full-model checkpoint used for ordinary CARPK fine-tuning",
    )
    parser.add_argument(
        "--mae_checkpoint",
        default="",
        help="v6 FSC-pretrained MAE checkpoint used to initialize the encoder",
    )
    parser.add_argument(
        "--head_resume",
        default="",
        help="initialize compatible non-backbone counting-head tensors",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backbone_name", default="dinov2_vitb14_reg")
    parser.add_argument("--backbone_repo", default=DEFAULT_DINOV2_REPO)
    parser.add_argument("--decoder_dim", default=192, type=int)
    parser.add_argument(
        "--architecture_version", choices=("v3", "v4", "v5", "v6"), default="v6"
    )
    parser.add_argument("--dino_layer_indices", nargs=4, type=int, default=None)
    parser.add_argument(
        "--v4_query_mode",
        choices=("image_film", "none", "repetition"),
        default="repetition",
    )
    parser.add_argument("--repetition_topk", default=16, type=int)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--accum_iter", default=1, type=int)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument(
        "--blr",
        default=None,
        type=float,
        help="CounTR base LR, scaled by effective batch size / 256",
    )
    parser.add_argument("--min_lr", default=2e-6, type=float)
    parser.add_argument("--weight_decay", default=0.05, type=float)
    parser.add_argument("--warmup_epochs", default=1, type=int)
    parser.add_argument("--val_interval", default=1, type=int)
    parser.add_argument(
        "--patience",
        default=5,
        type=int,
        help="stop after this many validation checks without an MAE improvement",
    )
    parser.add_argument("--val_fraction", default=0.1, type=float)
    parser.add_argument("--crop_size", default=384, type=int)
    parser.add_argument("--density_scale", default=60.0, type=float)
    parser.add_argument("--focus_probability", default=0.75, type=float)
    parser.add_argument("--window_size", default=384, type=int)
    parser.add_argument("--window_stride", default=128, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--positive_density_weight", default=4.0, type=float)
    parser.add_argument("--count_loss_weight", default=5e-3, type=float)
    parser.add_argument("--log_count_loss_weight", default=0.5, type=float)
    parser.add_argument("--verification_loss_weight", default=5e-2, type=float)
    parser.add_argument("--max_train_batches", default=0, type=int)
    parser.add_argument("--max_val_images", default=0, type=int)
    parser.add_argument("--train_backbone", action="store_true")
    parser.add_argument("--trainable_backbone_blocks", default=0, type=int)
    parser.add_argument("--backbone_lr_multiplier", default=1.0, type=float)
    return parser


def _split_indices(length: int, fraction: float, seed: int):
    if not 0.0 < fraction < 1.0:
        raise ValueError("val_fraction must be between zero and one")
    shuffled = np.random.default_rng(seed).permutation(length)
    val_count = max(1, round(length * fraction))
    return sorted(shuffled[val_count:].tolist()), sorted(shuffled[:val_count].tolist())


def _parameter_groups(model, weight_decay: float, backbone_lr_multiplier: float):
    groups = {
        (False, True): [],
        (False, False): [],
        (True, True): [],
        (True, False): [],
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_backbone = name.startswith("backbone.")
        use_decay = parameter.ndim > 1 and not name.endswith(".bias")
        groups[(is_backbone, use_decay)].append(parameter)
    return [
        {
            "params": parameters,
            "weight_decay": weight_decay if use_decay else 0.0,
            "lr_scale": backbone_lr_multiplier if is_backbone else 1.0,
        }
        for (is_backbone, use_decay), parameters in groups.items()
        if parameters
    ]


def _learning_rate(args, step: int, total_steps: int, steps_per_epoch: int) -> float:
    warmup_steps = args.warmup_epochs * steps_per_epoch
    if warmup_steps and step < warmup_steps:
        return args.lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps - 1, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return args.min_lr + (args.lr - args.min_lr) * cosine


@torch.inference_mode()
def _validate(model, dataset, indices, args, device):
    model.eval()
    limit = min(args.max_val_images or len(indices), len(indices))
    absolute_error = 0.0
    squared_error = 0.0
    bias = 0.0
    for position, index in enumerate(indices[:limit]):
        image, points, _ = dataset[index]
        density = predict_density_sliding_window(
            model,
            image.unsqueeze(0).to(device),
            None,
            window_size=args.window_size,
            stride=args.window_stride,
        )
        predicted = density.sum().item() / args.density_scale
        target = len(points)
        error = predicted - target
        absolute_error += abs(error)
        squared_error += error**2
        bias += error
        if position % 25 == 0 or position + 1 == limit:
            print(
                f"  val [{position + 1}/{limit}] pred={predicted:.2f} "
                f"gt={target} ae={abs(error):.2f}",
                flush=True,
            )
    return {
        "images": limit,
        "MAE": absolute_error / limit,
        "RMSE": math.sqrt(squared_error / limit),
        "count_bias": bias / limit,
    }


def main(args) -> None:
    if (
        args.epochs <= 0
        or args.batch_size <= 0
        or args.val_interval <= 0
        or args.patience < 0
        or args.accum_iter <= 0
    ):
        raise ValueError(
            "epochs, batch_size, accum_iter, and val_interval must be positive; "
            "patience may be zero to disable early stopping"
        )
    if args.backbone_lr_multiplier <= 0:
        raise ValueError("backbone_lr_multiplier must be positive")
    if args.init_checkpoint and (args.mae_checkpoint or args.head_resume):
        raise ValueError(
            "init_checkpoint is mutually exclusive with mae_checkpoint/head_resume"
        )
    if args.mae_checkpoint and args.architecture_version != "v6":
        raise ValueError("mae_checkpoint is only valid for v6")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    full_dataset = CARPKTestData(args.data_dir, resize_height=0)
    train_indices, val_indices = _split_indices(
        len(full_dataset), args.val_fraction, args.seed
    )
    split = {
        "seed": args.seed,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "source": full_dataset.source,
    }
    (output_dir / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    train_dataset = CARPKCropData(
        args.data_dir,
        train_indices,
        crop_size=args.crop_size,
        density_scale=args.density_scale,
        focus_probability=args.focus_probability,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        generator=torch.Generator().manual_seed(args.seed),
        drop_last=False,
    )

    model = CountingNetwork(
        backbone_name=args.backbone_name,
        backbone_repo=args.backbone_repo,
        decoder_dim=args.decoder_dim,
        freeze_backbone=not args.train_backbone,
        trainable_backbone_blocks=(
            0 if args.train_backbone else args.trainable_backbone_blocks
        ),
        enable_text_conditioning=False,
        text_dropout_p=0.0,
        architecture_version=args.architecture_version,
        dino_layer_indices=args.dino_layer_indices,
        v4_query_mode=args.v4_query_mode,
        repetition_topk=args.repetition_topk,
        mae_checkpoint=args.mae_checkpoint or None,
        mae_pretrained=bool(args.mae_checkpoint),
    ).to(device)
    if args.init_checkpoint:
        misc.load_model_FSC(
            argparse.Namespace(resume=args.init_checkpoint), model, strict=True
        )
    elif args.head_resume:
        misc.load_model_components(
            args.head_resume,
            model,
            exclude_prefixes=("backbone.",),
        )
    effective_batch_size = args.batch_size * args.accum_iter
    if args.blr is not None:
        args.lr = args.blr * effective_batch_size / 256.0
        print(
            f"CounTR LR scaling: base={args.blr:.3e}, "
            f"effective_batch={effective_batch_size}, actual={args.lr:.6e}"
        )
    optimizer = torch.optim.AdamW(
        _parameter_groups(
            model, args.weight_decay, args.backbone_lr_multiplier
        ),
        lr=args.lr,
        betas=(0.9, 0.95),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    log_path = output_dir / "log.jsonl"
    started = time.time()

    initial_validation = _validate(model, full_dataset, val_indices, args, device)
    initial_checkpoint = {
        "model": model.state_dict(),
        "epoch": -1,
        "validation": initial_validation,
        "args": vars(args),
    }
    torch.save(initial_checkpoint, output_dir / "checkpoint-best.pth")
    best_mae = initial_validation["MAE"]
    best_epoch = 0
    checks_without_improvement = 0
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps({"epoch": 0, "validation": initial_validation}) + "\n"
        )
    print(f"epoch 0 validation {json.dumps(initial_validation)}", flush=True)

    for epoch in range(args.epochs):
        model.train(True)
        running_loss = 0.0
        running_ae = 0.0
        processed = 0
        optimizer.zero_grad(set_to_none=True)
        epoch_steps = min(
            args.max_train_batches or steps_per_epoch,
            steps_per_epoch,
        )
        for batch_index, (images, targets, crop_counts) in enumerate(train_loader):
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
            step = epoch * steps_per_epoch + batch_index
            lr = _learning_rate(args, step, total_steps, steps_per_epoch)
            for group in optimizer.param_groups:
                group["lr"] = lr * group.get("lr_scale", 1.0)
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            crop_counts = crop_counts.to(device, non_blocking=True, dtype=torch.float32)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                auxiliary = model(images, None, return_aux=True)
                output = auxiliary["density"]
                target_peak = targets.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
                normalized_target = (targets / target_peak).clamp(0.0, 1.0)
                weights = 1.0 + args.positive_density_weight * normalized_target
                density_loss = ((output - targets).square() * weights).mean()
                predicted_counts = output.flatten(1).sum(1) / args.density_scale
                count_loss = torch.nn.functional.smooth_l1_loss(
                    predicted_counts, crop_counts
                )
                log_count_loss = torch.nn.functional.smooth_l1_loss(
                    torch.log1p(predicted_counts.clamp_min(0.0)),
                    torch.log1p(crop_counts),
                )
                verification_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    auxiliary["verification_logits"], normalized_target
                )
                loss = (
                    density_loss
                    + args.count_loss_weight * count_loss
                    + args.log_count_loss_weight * log_count_loss
                    + args.verification_loss_weight * verification_loss
                )
            scaler.scale(loss / args.accum_iter).backward()
            should_update = (
                (batch_index + 1) % args.accum_iter == 0
                or batch_index + 1 == epoch_steps
            )
            if should_update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            batch_size = len(images)
            running_loss += loss.item() * batch_size
            running_ae += (predicted_counts.detach() - crop_counts).abs().sum().item()
            processed += batch_size
            if batch_index % 20 == 0:
                print(
                    f"epoch {epoch + 1}/{args.epochs} batch {batch_index + 1}/"
                    f"{steps_per_epoch} loss={loss.item():.5f} "
                    f"crop_MAE={running_ae / processed:.3f} lr={lr:.2e}",
                    flush=True,
                )

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": running_loss / processed,
            "train_crop_MAE": running_ae / processed,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        if (epoch + 1) % args.val_interval == 0 or epoch + 1 == args.epochs:
            validation = _validate(model, full_dataset, val_indices, args, device)
            epoch_record["validation"] = validation
            print(f"epoch {epoch + 1} validation {json.dumps(validation)}", flush=True)
            checkpoint = {
                "model": model.state_dict(),
                "epoch": epoch,
                "validation": validation,
                "args": vars(args),
            }
            torch.save(checkpoint, output_dir / "checkpoint-last.pth")
            if validation["MAE"] < best_mae:
                best_mae = validation["MAE"]
                best_epoch = epoch + 1
                checks_without_improvement = 0
                torch.save(checkpoint, output_dir / "checkpoint-best.pth")
                print(f"new best checkpoint at epoch {best_epoch}", flush=True)
            else:
                checks_without_improvement += 1
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(epoch_record) + "\n")
        if args.patience and checks_without_improvement >= args.patience:
            print(
                f"early stopping after {checks_without_improvement} validation "
                "checks without improvement",
                flush=True,
            )
            break

    summary = {
        "best_epoch": best_epoch,
        "best_validation_MAE": best_mae,
        "initial_validation": initial_validation,
        "train_images": len(train_indices),
        "validation_images": len(val_indices),
        "elapsed_seconds": time.time() - started,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(get_args_parser().parse_args())
