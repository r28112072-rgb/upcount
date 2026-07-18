"""CARPK test-set loading for official files or an exported manifest."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor


_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def _load_image_and_points(root: Path, record: dict) -> tuple[Image.Image, np.ndarray]:
    image = Image.open(root / record["image_path"]).convert("RGB")
    boxes = np.asarray(record["boxes"], dtype=np.float32).reshape(-1, 4)
    box_format = record.get("box_format", "xywh")
    if box_format == "xywh":
        points = boxes[:, :2] + boxes[:, 2:] / 2.0
    elif box_format == "xyxy":
        points = (boxes[:, :2] + boxes[:, 2:]) / 2.0
    else:
        raise ValueError(f"unsupported CARPK box format {box_format!r}")
    return image, points


def _find_directory(root: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        candidate = root / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"none of {names!r} exists below CARPK data directory {root}"
    )


def _find_image(images_dir: Path, image_id: str) -> Path:
    supplied = images_dir / image_id
    if supplied.is_file():
        return supplied
    stem = Path(image_id).stem
    for extension in _IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{extension}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no image found for CARPK id {image_id!r}")


def _parse_xyxy_annotations(path: Path) -> np.ndarray:
    boxes = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = [item for item in re.split(r"[\s,]+", line.strip()) if item]
            if not fields:
                continue
            if len(fields) < 4:
                raise ValueError(
                    f"{path}:{line_number} has fewer than four box coordinates"
                )
            boxes.append([float(value) for value in fields[:4]])
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def _official_records(root: Path) -> list[dict]:
    images_dir = _find_directory(root, ("Images", "images"))
    annotations_dir = _find_directory(root, ("Annotations", "annotations"))
    split_candidates = (
        root / "ImageSets" / "test.txt",
        root / "ImageSets" / "Main" / "test.txt",
        root / "test.txt",
    )
    split_file = next((path for path in split_candidates if path.is_file()), None)
    if split_file is not None:
        image_ids = [
            line.strip().split()[0]
            for line in split_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        image_ids = sorted(
            path.name
            for path in images_dir.iterdir()
            if path.suffix.lower() in _IMAGE_EXTENSIONS
        )
    records = []
    for image_id in image_ids:
        image_path = _find_image(images_dir, image_id)
        annotation_path = annotations_dir / f"{image_path.stem}.txt"
        if not annotation_path.is_file():
            raise FileNotFoundError(f"missing CARPK annotation {annotation_path}")
        boxes_xyxy = _parse_xyxy_annotations(annotation_path)
        records.append(
            {
                "image_id": image_path.name,
                "image_path": str(image_path.relative_to(root)),
                "boxes": boxes_xyxy.tolist(),
                "box_format": "xyxy",
            }
        )
    return records


class CARPKTestData(Dataset):
    """Return resized images and box-center points for the CARPK test split."""

    def __init__(self, data_dir: str | Path, *, resize_height: int = 384):
        self.root = Path(data_dir)
        if resize_height < 0:
            raise ValueError("resize_height must be non-negative")
        self.resize_height = resize_height
        manifest_path = self.root / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.records = manifest["records"]
            self.source = manifest.get("source", "manifest")
        else:
            self.records = _official_records(self.root)
            self.source = "official-files"
        if not self.records:
            raise RuntimeError(f"CARPK data directory contains no records: {self.root}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image, points = _load_image_and_points(self.root, record)
        original_width, original_height = image.size

        if self.resize_height:
            new_height = self.resize_height
            new_width = max(1, round(original_width * new_height / original_height))
            image = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
            points[:, 0] *= new_width / original_width
            points[:, 1] *= new_height / original_height

        image_tensor = pil_to_tensor(image).float().div_(255.0)
        return image_tensor, torch.from_numpy(points), record["image_id"]


class CARPKCropData(Dataset):
    """Random native-scale crops with Gaussian point-density supervision."""

    def __init__(
        self,
        data_dir: str | Path,
        indices: list[int],
        *,
        crop_size: int = 384,
        density_scale: float = 60.0,
        focus_probability: float = 0.75,
    ):
        base = CARPKTestData(data_dir, resize_height=0)
        if crop_size <= 0:
            raise ValueError("crop_size must be positive")
        if density_scale <= 0:
            raise ValueError("density_scale must be positive")
        if not 0.0 <= focus_probability <= 1.0:
            raise ValueError("focus_probability must be between zero and one")
        self.root = base.root
        self.records = base.records
        self.indices = list(indices)
        self.crop_size = crop_size
        self.density_scale = density_scale
        self.focus_probability = focus_probability

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        record = self.records[self.indices[index]]
        image, points = _load_image_and_points(self.root, record)
        width, height = image.size
        if width < self.crop_size or height < self.crop_size:
            raise ValueError(
                f"CARPK image {record['image_id']} is smaller than crop size "
                f"{self.crop_size}: {width}x{height}"
            )

        max_left = width - self.crop_size
        max_top = height - self.crop_size
        if len(points) and random.random() < self.focus_probability:
            focus_x, focus_y = points[random.randrange(len(points))]
            jitter = self.crop_size // 3
            left = round(focus_x - self.crop_size / 2 + random.randint(-jitter, jitter))
            top = round(focus_y - self.crop_size / 2 + random.randint(-jitter, jitter))
            left = min(max(left, 0), max_left)
            top = min(max(top, 0), max_top)
        else:
            left = random.randint(0, max_left)
            top = random.randint(0, max_top)

        image = image.crop((left, top, left + self.crop_size, top + self.crop_size))
        crop_points = points.copy()
        crop_points[:, 0] -= left
        crop_points[:, 1] -= top
        inside = (
            (crop_points[:, 0] >= 0)
            & (crop_points[:, 0] < self.crop_size)
            & (crop_points[:, 1] >= 0)
            & (crop_points[:, 1] < self.crop_size)
        )
        crop_points = crop_points[inside]

        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            crop_points[:, 0] = self.crop_size - 1 - crop_points[:, 0]

        impulse = np.zeros((self.crop_size, self.crop_size), dtype=np.float32)
        if len(crop_points):
            x = np.clip(np.rint(crop_points[:, 0]).astype(int), 0, self.crop_size - 1)
            y = np.clip(np.rint(crop_points[:, 1]).astype(int), 0, self.crop_size - 1)
            np.add.at(impulse, (y, x), 1.0)
        density = gaussian_filter(impulse, sigma=1.0) * self.density_scale
        image_tensor = pil_to_tensor(image).float().div_(255.0)
        return image_tensor, torch.from_numpy(density), len(crop_points)
