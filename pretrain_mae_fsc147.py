"""CounTR-compatible 500-epoch MAE pretraining on the FSC-147 train split."""

from __future__ import annotations

import argparse
import datetime
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from models_mae_pretraining import (
    DEFAULT_MAE_FULL_CHECKPOINT_URL,
    countr_mae_vit_base_patch16,
)


class FSC147MAEData(Dataset):
    def __init__(self, image_dir: str, split_file: str) -> None:
        with open(split_file, encoding="utf-8") as handle:
            self.image_names = json.load(handle)["train"]
        self.image_dir = Path(image_dir)
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    384,
                    scale=(0.2, 1.0),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.image_dir / self.image_names[index]) as image:
            return self.transform(image.convert("RGB"))


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("CounTR-style FSC-147 MAE pretraining")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--data_split_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", default=500, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--accum_iter", default=1, type=int)
    parser.add_argument("--mask_ratio", default=0.5, type=float)
    parser.add_argument("--norm_pix_loss", action="store_true")
    parser.add_argument("--blr", default=1.5e-4, type=float)
    parser.add_argument("--lr", default=None, type=float)
    parser.add_argument("--min_lr", default=0.0, type=float)
    parser.add_argument("--warmup_epochs", default=10, type=int)
    parser.add_argument("--weight_decay", default=0.05, type=float)
    parser.add_argument("--init_checkpoint", default=DEFAULT_MAE_FULL_CHECKPOINT_URL)
    parser.add_argument("--resume", default="")
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--save_interval", default=100, type=int)
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_train_batches", default=0, type=int)
    return parser


def _parameter_groups(model, weight_decay: float):
    decay, no_decay = [], []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        (decay if parameter.ndim > 1 else no_decay).append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _learning_rate(args, progress: float) -> float:
    if progress < args.warmup_epochs:
        return args.lr * progress / max(args.warmup_epochs, 1)
    cosine_progress = (progress - args.warmup_epochs) / max(
        args.epochs - args.warmup_epochs, 1
    )
    cosine_progress = min(max(cosine_progress, 0.0), 1.0)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (
        1.0 + math.cos(math.pi * cosine_progress)
    )


def _save_checkpoint(path: Path, model, optimizer, scaler, epoch: int, args) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "args": vars(args),
        },
        path,
    )


def main(args) -> None:
    if args.epochs <= 0 or args.batch_size <= 0 or args.accum_iter <= 0:
        raise ValueError("epochs, batch_size, and accum_iter must be positive")
    if args.save_interval <= 0:
        raise ValueError("save_interval must be positive")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)

    dataset = FSC147MAEData(args.image_dir, args.data_split_file)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=False,
        generator=torch.Generator().manual_seed(args.seed),
    )
    model = countr_mae_vit_base_patch16(
        img_size=384,
        norm_pix_loss=args.norm_pix_loss,
    ).to(device)
    effective_batch_size = args.batch_size * args.accum_iter
    if args.lr is None:
        args.lr = args.blr * effective_batch_size / 256.0
    optimizer = torch.optim.AdamW(
        _parameter_groups(model, args.weight_decay),
        lr=args.lr,
        betas=(0.9, 0.95),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        args.start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed pretraining from {args.resume} at epoch {args.start_epoch}")
    elif args.init_checkpoint:
        model.load_initial_weights(args.init_checkpoint)

    print(
        f"FSC-147 MAE: images={len(dataset)}, epochs={args.epochs}, "
        f"effective_batch={effective_batch_size}, base_lr={args.blr:.3e}, "
        f"actual_lr={args.lr:.6e}, mask_ratio={args.mask_ratio}"
    )
    log_path = output_dir / "log.jsonl"
    started = time.time()
    steps_per_epoch = len(loader)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]

    for epoch in range(args.start_epoch, args.epochs):
        model.train(True)
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        processed_batches = 0
        for step, images in enumerate(loader):
            if args.max_train_batches and step >= args.max_train_batches:
                break
            progress = epoch + step / max(steps_per_epoch, 1)
            learning_rate = _learning_rate(args, progress)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            images = images.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                loss, _, _ = model(images, mask_ratio=args.mask_ratio)
                scaled_loss = loss / args.accum_iter
            scaler.scale(scaled_loss).backward()
            should_update = (
                (step + 1) % args.accum_iter == 0
                or step + 1 == steps_per_epoch
                or (
                    args.max_train_batches
                    and step + 1 == args.max_train_batches
                )
            )
            if should_update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running_loss += loss.detach().item()
            processed_batches += 1
            if step % 20 == 0:
                print(
                    f"epoch {epoch + 1}/{args.epochs} batch {step + 1}/"
                    f"{steps_per_epoch} loss={running_loss / processed_batches:.6f} "
                    f"lr={learning_rate:.3e}",
                    flush=True,
                )

        record = {
            "epoch": epoch + 1,
            "loss": running_loss / processed_batches,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": time.time() - started,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if (epoch + 1) % args.save_interval == 0 or epoch + 1 == args.epochs:
            numbered = output_dir / f"checkpoint-{epoch + 1}.pth"
            _save_checkpoint(numbered, model, optimizer, scaler, epoch, args)
            _save_checkpoint(
                output_dir / "checkpoint-last.pth",
                model,
                optimizer,
                scaler,
                epoch,
                args,
            )

    elapsed = str(datetime.timedelta(seconds=int(time.time() - started)))
    print(f"MAE pretraining time {elapsed}")


if __name__ == "__main__":
    main(get_args_parser().parse_args())
