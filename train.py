import argparse
import datetime
import json
import numpy as np
import os
import time
import random
from pathlib import Path
import math
from PIL import Image

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset

from torchvision.transforms import Resize

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.FSC147 import ProcessTrainImage, TTensor
from util.evaluation import predict_density_sliding_window
from models_counting_network import (
    CountingNetwork,
    DEFAULT_DINOV2_REPO,
    IMAGE_ONLY_ARCHITECTURES,
)
import open_clip


def get_args_parser():
    parser = argparse.ArgumentParser(
        "Train UPCount for reference-free class-agnostic object counting"
    )

    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
    )
    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="gradient accumulation steps",
    )

    parser.add_argument("--epochs", default=50, type=int)

    parser.add_argument("--weight_decay", type=float, default=0.05)

    parser.add_argument(
        "--count_loss_weight",
        type=float,
        default=5e-3,
        help="weight of count-integral consistency loss",
    )
    parser.add_argument(
        "--verification_loss_weight",
        type=float,
        default=5e-2,
        help="weight of the differentiable verification-gate loss",
    )
    parser.add_argument(
        "--log_count_loss_weight",
        type=float,
        default=0.5,
        help="weight of scale-balanced log-count consistency loss",
    )
    parser.add_argument(
        "--positive_density_weight",
        type=float,
        default=4.0,
        help="extra density-map weight near annotated objects",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="learning rate",
    )
    parser.add_argument(
        "--blr",
        type=float,
        default=None,
        help=(
            "CounTR-style base learning rate; when set, actual lr is "
            "blr * batch_size * accum_iter / 256"
        ),
    )

    parser.add_argument(
        "--min_lr",
        type=float,
        default=0.0,
        help="lower lr bound for cyclic schedulers that hit 0",
    )

    parser.add_argument(
        "--warmup_epochs", type=int, default=2, help="epochs to warmup lr"
    )

    parser.add_argument(
        "--output_dir",
        default="./results",
        help="path where to save model and log",
    )

    parser.add_argument("--device", default="cuda", help="device to use for training")

    parser.add_argument("--backbone_name", default="dinov2_vitb14_reg")
    parser.add_argument("--backbone_repo", default=DEFAULT_DINOV2_REPO)
    parser.add_argument("--decoder_dim", default=192, type=int)
    parser.add_argument(
        "--architecture_version",
        choices=("v1", "v2", "v3", "v4", "v5", "v6"),
        default="v6",
        help=(
            "v4 adds DPT/repetition, v5 adds FeatUp, and v6 replaces DINO "
            "with a CounTR-style MAE-pretrained vanilla ViT-B/16"
        ),
    )
    parser.add_argument(
        "--dino_layer_indices",
        nargs=4,
        type=int,
        default=None,
        metavar=("L1", "L2", "L3", "L4"),
        help="explicit zero-based ViT blocks; v4-v6 default to 2 5 8 11 for ViT-B",
    )
    parser.add_argument(
        "--v4_query_mode",
        choices=("image_film", "none", "repetition"),
        default="repetition",
        help="v4 ablation mode: legacy image FiLM, no query, or repetition prototype",
    )
    parser.add_argument(
        "--repetition_topk",
        default=16,
        type=int,
        help="number of strongest non-local token matches used by the v4 prototype",
    )
    parser.add_argument(
        "--mae_checkpoint",
        default="",
        help=(
            "optional MAE/CounTR encoder checkpoint for v6; defaults to the "
            "official ImageNet MAE ViT-B/16 checkpoint"
        ),
    )
    parser.add_argument(
        "--no_mae_pretrained",
        action="store_true",
        help="construct the v6 ViT without loading MAE weights (debugging only)",
    )
    parser.add_argument(
        "--refinement_only",
        action="store_true",
        help="train only v3 residual-refinement modules (normally with a v1 checkpoint)",
    )
    parser.add_argument(
        "--head_resume",
        default="",
        help=(
            "initialize compatible non-backbone tensors from a counting checkpoint; "
            "keeps the v6 MAE encoder intact"
        ),
    )
    parser.add_argument(
        "--train_backbone",
        action="store_true",
        help="fine-tune the image backbone instead of training only the counting head",
    )
    parser.add_argument(
        "--trainable_backbone_blocks",
        default=0,
        type=int,
        help="fine-tune only the last N ViT blocks; ignored with --train_backbone",
    )
    parser.add_argument(
        "--backbone_lr_multiplier",
        default=0.1,
        type=float,
        help="learning-rate multiplier for trainable backbone parameters",
    )
    parser.add_argument(
        "--disable_text_conditioning",
        action="store_true",
        help="use the reference-less image-derived query for every sample",
    )
    parser.add_argument(
        "--text_dropout_p",
        default=0.25,
        type=float,
        help="probability of switching from text to the image-derived query while training",
    )
    parser.add_argument(
        "--validation_interval",
        default=5,
        type=int,
        help="run the validation split every N epochs and on the final epoch",
    )
    parser.add_argument(
        "--save_interval",
        default=50,
        type=int,
        help="save a resumable numbered checkpoint every N epochs; zero disables",
    )
    parser.add_argument(
        "--max_validation_images",
        default=0,
        type=int,
        help="limit validation images for debugging; zero uses the complete split",
    )
    parser.add_argument("--window_size", default=384, type=int)
    parser.add_argument("--window_stride", default=128, type=int)
    parser.add_argument("--print_freq", default=20, type=int)
    parser.add_argument(
        "--max_train_batches",
        default=0,
        type=int,
        help="limit batches per epoch for smoke tests; zero uses the full training set",
    )

    parser.add_argument("--seed", default=0, type=int)

    parser.add_argument(
        "--resume",
        default="",
        help="file name for model checkpoint to resume from (leave empty to not use a checkpoint)",
    )

    parser.add_argument("--start_epoch", default=0, type=int)

    parser.add_argument("--num_workers", default=8, type=int)

    parser.add_argument(
        "--pin_mem",
        action="store_false",
        help="pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU",
    )

    parser.add_argument(
        "--img_dir",
        default="./data/FSC147/images_384_VarV2",
        help="directory containing images from FSC-147",
    )

    parser.add_argument(
        "--gt_dir",
        default="./data/FSC147/gt_density_map_adaptive_384_VarV2",
        help="directory containing ground truth binary dot annotation maps",
    )

    parser.add_argument(
        "--class_file",
        default="./data/FSC147/ImageClasses_FSC147.txt",
        help="name of file with FSC-147 image class names",
    )

    parser.add_argument(
        "--FSC147_anno_file",
        default="./data/FSC147/annotation_FSC147_384.json",
        help="name of file with FSC-147 annotations",
    )

    parser.add_argument(
        "--FSC147_D_anno_file",
        default="./FSC-147-D.json",
        help="name of file with FSC-147-D",
    )

    parser.add_argument(
        "--data_split_file",
        default="./data/FSC147/Train_Test_Val_FSC_147.json",
        help="name of file with train, val, test splits of FSC-147",
    )

    return parser


def parameter_groups_weight_decay(model, weight_decay, backbone_lr_multiplier=0.1):
    """AdamW groups without a dependency on the legacy timm optimizer API."""

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
    parameter_groups = []
    for (is_backbone, use_decay), parameters in groups.items():
        if parameters:
            parameter_groups.append(
                {
                    "params": parameters,
                    "weight_decay": weight_decay if use_decay else 0.0,
                    "lr_scale": backbone_lr_multiplier if is_backbone else 1.0,
                }
            )
    return parameter_groups


class TrainData(Dataset):
    def __init__(self, args):

        self.img_dir = args.img_dir
        self.gt_dir = args.gt_dir

        with open(args.data_split_file) as f:
            data_split = json.load(f)
        self.img = data_split["train"]

        with open(args.FSC147_anno_file) as f:
            fsc147_annotations = json.load(f)
        self.fsc147_annotations = fsc147_annotations

        with open(args.FSC147_D_anno_file) as f:
            fsc147_d_annotations = json.load(f)
        self.fsc147_d_annotations = fsc147_d_annotations

        self.class_dict = {}
        with open(args.class_file) as f:
            for line in f:
                key = line.split()[0]
                val = line.split()[1:]
                self.class_dict[key] = val

        self.transform_train = ProcessTrainImage(
            self.img_dir,
            self.fsc147_annotations,
            self.fsc147_d_annotations,
            self.class_dict,
            self.img,
        )

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        im_id = self.img[idx]
        fsc147_anno = self.fsc147_annotations[im_id]
        fsc147_d_anno = self.fsc147_d_annotations[im_id]
        text = fsc147_d_anno["text_description"]

        dots = np.array(fsc147_anno["points"])

        image = Image.open("{}/{}".format(self.img_dir, im_id))
        image.load()
        density_path = self.gt_dir + "/" + im_id.split(".jpg")[0] + ".npy"
        density = np.load(density_path).astype("float32")

        sample = {
            "image": image,
            "text": text,
            "gt_density": density,
            "dots": dots,
            "id": im_id,
        }
        sample = self.transform_train(sample)
        return (
            # open_clip_vit_b_16_preprocess(
                sample["image"]
            # )
            ,
            sample["gt_density"],
            sample["text"],
        )


class ValData(Dataset):
    def __init__(self, args):

        self.img_dir = args.img_dir

        with open(args.data_split_file) as f:
            data_split = json.load(f)
        self.img = data_split["val"]

        with open(args.FSC147_anno_file) as f:
            fsc147_annotations = json.load(f)
        self.fsc147_annotations = fsc147_annotations

        with open(args.FSC147_D_anno_file) as f:
            fsc147_d_annotations = json.load(f)
        self.fsc147_d_annotations = fsc147_d_annotations

        self.clip_tokenizer = open_clip.get_tokenizer("ViT-B-16")

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        im_id = self.img[idx]
        fsc147_anno = self.fsc147_annotations[im_id]
        fsc147_d_anno = self.fsc147_d_annotations[im_id]
        text = self.clip_tokenizer(fsc147_d_anno["text_description"]).squeeze(-2)

        dots = np.array(fsc147_anno["points"])

        image = Image.open("{}/{}".format(self.img_dir, im_id))
        image.load()
        W, H = image.size

        # This resizing step exists for consistency with CounTR's data resizing step.
        new_H = 16 * int(H / 16)
        new_W = 16 * int(W / 16)
        image = Resize((new_H, new_W))(image)
        image = TTensor(image)

        return image, dots, text


def cosine_learning_rate(args, step, total_steps, warmup_steps):
    """Per-step linear warm-up followed by cosine decay."""

    if warmup_steps > 0 and step < warmup_steps:
        return args.lr * float(step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps - 1, 1)
    progress = min(max(progress, 0.0), 1.0)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (
        1.0 + math.cos(math.pi * progress)
    )


@torch.inference_mode()
def evaluate_validation(model, data_loader, args, device):
    """Evaluate the complete validation split with the test-time tiling path."""

    model.eval()
    absolute_error = 0.0
    squared_error = 0.0
    processed = 0
    limit = min(
        args.max_validation_images or len(data_loader.dataset),
        len(data_loader.dataset),
    )
    for samples, gt_dots, text_description in data_loader:
        if processed >= limit:
            break
        samples = samples.to(device, non_blocking=True, dtype=torch.float32)
        text_description = text_description.to(device, non_blocking=True)
        model_query = (
            None
            if args.disable_text_conditioning
            or args.architecture_version in IMAGE_ONLY_ARCHITECTURES
            else text_description
        )
        density_map = predict_density_sliding_window(
            model,
            samples,
            model_query,
            window_size=args.window_size,
            stride=args.window_stride,
        )
        predicted_count = density_map.sum().item() / 60.0
        target_count = gt_dots.shape[1]
        error = abs(predicted_count - target_count)
        absolute_error += error
        squared_error += error**2
        processed += 1
        if processed % 100 == 0 or processed == limit:
            print(
                f"Validation [{processed}/{limit}] pred={predicted_count:.2f} "
                f"gt={target_count} running_MAE={absolute_error / processed:.2f}",
                flush=True,
            )
    if processed == 0:
        raise RuntimeError("validation processed no images")
    return {
        "MAE": absolute_error / processed,
        "RMSE": (squared_error / processed) ** 0.5,
        "images": processed,
        "complete_split": processed == len(data_loader.dataset),
    }


def main(args):

    if args.validation_interval <= 0:
        raise ValueError("--validation_interval must be positive")
    if args.epochs <= 0 or args.batch_size <= 0 or args.accum_iter <= 0:
        raise ValueError("epochs, batch_size, and accum_iter must be positive")
    if args.blr is not None and args.blr <= 0:
        raise ValueError("--blr must be positive")
    if args.save_interval < 0:
        raise ValueError("--save_interval cannot be negative")
    if args.trainable_backbone_blocks < 0:
        raise ValueError("--trainable_backbone_blocks must be non-negative")
    if args.backbone_lr_multiplier <= 0:
        raise ValueError("--backbone_lr_multiplier must be positive")
    if args.resume and args.head_resume:
        raise ValueError("--resume and --head_resume are mutually exclusive")
    if args.mae_checkpoint and args.architecture_version != "v6":
        raise ValueError("--mae_checkpoint is only valid with --architecture_version v6")

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)

    # Fix a random seed, and force PyTorch to be deterministic for reproducibility. See https://pytorch.org/docs/stable/notes/randomness.html.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    cudnn.benchmark = True
    # NOTE: some operations during training do not have deterministic alternatives (such as [upsample_bilinear2d_backward_out_cuda]). Therefore, the line below is not executed.
    # torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    dataset_train = TrainData(args)
    dataset_val = ValData(args)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    effective_batch_size = args.batch_size * args.accum_iter
    if args.blr is not None:
        args.lr = args.blr * effective_batch_size / 256.0
        print(
            f"CounTR LR scaling: base={args.blr:.3e}, "
            f"effective_batch={effective_batch_size}, actual={args.lr:.6e}"
        )

    # Initialize the modern multi-depth, query-conditioned counting model.
    model = CountingNetwork(
        backbone_name=args.backbone_name,
        backbone_repo=args.backbone_repo,
        decoder_dim=args.decoder_dim,
        freeze_backbone=not args.train_backbone,
        trainable_backbone_blocks=(
            0 if args.train_backbone else args.trainable_backbone_blocks
        ),
        enable_text_conditioning=(
            args.architecture_version not in IMAGE_ONLY_ARCHITECTURES
            and not args.disable_text_conditioning
        ),
        text_dropout_p=args.text_dropout_p,
        architecture_version=args.architecture_version,
        dino_layer_indices=args.dino_layer_indices,
        v4_query_mode=args.v4_query_mode,
        repetition_topk=args.repetition_topk,
        mae_checkpoint=args.mae_checkpoint or None,
        mae_pretrained=not args.no_mae_pretrained,
    )

    model.to(device)

    if args.refinement_only:
        if args.architecture_version != "v3":
            raise ValueError("--refinement_only requires --architecture_version v3")
        refinement_prefixes = (
            "feature_fusion.spatial_logits",
            "detail_stem",
            "detail_fusion",
            "refinement_decoder",
            "density_refinement_head",
            "count_scale_head",
        )
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith(refinement_prefixes)

    print("Model = %s" % str(model))

    print("lr: %.2e" % args.lr)

    param_groups = parameter_groups_weight_decay(
        model, args.weight_decay, args.backbone_lr_multiplier
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)

    loss_scaler = NativeScaler()

    if args.head_resume:
        misc.load_model_components(
            args.head_resume,
            model,
            exclude_prefixes=("backbone.",),
        )
    if args.resume:
        misc.load_model(
            args=args,
            model_without_ddp=model,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
        )

    print(f"Start training for {args.epochs} epochs")

    best_val_mae = math.inf
    best_epoch = None
    start_time = time.time()
    total_steps = args.epochs * len(data_loader_train)
    warmup_steps = args.warmup_epochs * len(data_loader_train)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]

    for epoch in range(args.start_epoch, args.epochs):
        model.train(True)
        metric_logger = misc.MetricLogger(delimiter="  ")
        metric_logger.add_meter(
            "lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
        )
        header = "Epoch: [{}]".format(epoch)
        train_absolute_error = 0.0
        train_squared_error = 0.0
        train_images = 0
        optimizer.zero_grad(set_to_none=True)
        epoch_steps = min(
            args.max_train_batches or len(data_loader_train),
            len(data_loader_train),
        )

        for data_iter_step, (samples, gt_density, text_descriptions) in enumerate(
            metric_logger.log_every(data_loader_train, args.print_freq, header)
        ):
            if args.max_train_batches and data_iter_step >= args.max_train_batches:
                break
            global_step = epoch * len(data_loader_train) + data_iter_step
            learning_rate = cosine_learning_rate(
                args, global_step, total_steps, warmup_steps
            )
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = learning_rate * parameter_group["lr_scale"]

            samples = samples.to(device, non_blocking=True, dtype=torch.float32)
            gt_density = gt_density.to(device, non_blocking=True, dtype=torch.float32)
            text_descriptions = text_descriptions.to(device, non_blocking=True)
            model_queries = (
                None
                if args.disable_text_conditioning
                or args.architecture_version in IMAGE_ONLY_ARCHITECTURES
                else text_descriptions
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                auxiliary = model(samples, model_queries, return_aux=True)
                output = auxiliary["density"]
                target_scale = gt_density.amax(
                    dim=(-2, -1), keepdim=True
                ).clamp_min(1e-6)
                normalized_target = (gt_density / target_scale).clamp(0.0, 1.0)
                density_weights = 1.0 + args.positive_density_weight * normalized_target
                density_loss = ((output - gt_density).square() * density_weights).mean()
                predicted_counts = output.flatten(1).sum(dim=1) / 60.0
                target_counts = gt_density.flatten(1).sum(dim=1) / 60.0
                count_loss = F.smooth_l1_loss(predicted_counts, target_counts)
                log_count_loss = F.smooth_l1_loss(
                    torch.log1p(predicted_counts), torch.log1p(target_counts)
                )
                verification_target = normalized_target
                verification_weights = 1.0 + 4.0 * verification_target
                verification_loss = (
                    F.binary_cross_entropy_with_logits(
                        auxiliary["verification_logits"],
                        verification_target,
                        reduction="none",
                    )
                    * verification_weights
                ).mean()
                loss = (
                    density_loss
                    + args.count_loss_weight * count_loss
                    + args.log_count_loss_weight * log_count_loss
                    + args.verification_loss_weight * verification_loss
                )

            loss_scaler(
                loss / args.accum_iter,
                optimizer,
                parameters=trainable_parameters,
                update_grad=(
                    (data_iter_step + 1) % args.accum_iter == 0
                    or data_iter_step + 1 == epoch_steps
                ),
            )
            if (
                (data_iter_step + 1) % args.accum_iter == 0
                or data_iter_step + 1 == epoch_steps
            ):
                optimizer.zero_grad(set_to_none=True)
            count_errors = (predicted_counts.detach() - target_counts).abs()
            train_absolute_error += count_errors.sum().item()
            train_squared_error += count_errors.square().sum().item()
            train_images += output.shape[0]
            metric_logger.update(
                loss=loss.item(),
                density_loss=density_loss.item(),
                count_loss=count_loss.item(),
                log_count_loss=log_count_loss.item(),
                verification_loss=verification_loss.item(),
                lr=learning_rate,
            )

        train_stats = {
            key: meter.global_avg for key, meter in metric_logger.meters.items()
        }
        current_train_mae = train_absolute_error / train_images
        current_train_rmse = (train_squared_error / train_images) ** 0.5
        should_validate = (
            (epoch + 1) % args.validation_interval == 0
            or epoch + 1 == args.epochs
        )
        validation = None
        saved_checkpoint = False
        if should_validate:
            validation = evaluate_validation(model, data_loader_val, args, device)
            if validation["MAE"] < best_val_mae:
                best_val_mae = validation["MAE"]
                best_epoch = epoch
                misc.save_model(
                    args=args,
                    model=model,
                    model_without_ddp=model,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                    epoch=epoch,
                )
                saved_checkpoint = True
                with open(
                    os.path.join(args.output_dir, "best_checkpoint.json"),
                    mode="w",
                    encoding="utf-8",
                ) as handle:
                    json.dump(
                        {
                            "checkpoint": f"checkpoint-{epoch}.pth",
                            "epoch": epoch,
                            **validation,
                        },
                        handle,
                        indent=2,
                    )
        should_save_periodically = args.save_interval and (
            (epoch + 1) % args.save_interval == 0
            or epoch + 1 == args.epochs
        )
        if should_save_periodically and not saved_checkpoint:
            misc.save_model(
                args=args,
                model=model,
                model_without_ddp=model,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
            )

        log_stats = {
            **{f"train_{key}": value for key, value in train_stats.items()},
            "train_MAE": current_train_mae,
            "train_RMSE": current_train_rmse,
            "val_MAE": None if validation is None else validation["MAE"],
            "val_RMSE": None if validation is None else validation["RMSE"],
            "val_images": None if validation is None else validation["images"],
            "best_val_MAE": None if best_epoch is None else best_val_mae,
            "best_epoch": best_epoch,
            "epoch": epoch,
        }
        print(json.dumps(log_stats, indent=2), flush=True)
        with open(
            os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
        ) as handle:
            handle.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
