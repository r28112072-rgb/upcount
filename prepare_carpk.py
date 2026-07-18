"""Export the public Deep Lake CARPK test split to ordinary image files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Export CARPK from its public Deep Lake mirror")
    parser.add_argument("--uri", default="hub://activeloop/carpk-test")
    parser.add_argument(
        "--split",
        choices=("auto", "train", "test"),
        default="auto",
        help="manifest split name; auto infers it from the URI",
    )
    parser.add_argument("--output_dir", default="./data/CARPK/test")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(args) -> None:
    try:
        import deeplake
    except ImportError as error:
        raise RuntimeError(
            "prepare_carpk.py requires the optional dependency 'deeplake<4'"
        ) from error

    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    dataset = deeplake.load(args.uri, read_only=True)
    required = {"images", "boxes", "labels"}
    missing = required.difference(dataset.tensors)
    if missing:
        raise RuntimeError(f"Deep Lake CARPK dataset is missing tensors: {missing}")

    records = []
    for index in range(len(dataset)):
        image_id = f"{index:06d}.png"
        relative_path = Path("images") / image_id
        destination = output_dir / relative_path
        image = dataset.images[index].numpy()
        boxes = dataset.boxes[index].numpy().astype(float)
        labels = dataset.labels[index].numpy()
        if len(boxes) != len(labels):
            raise RuntimeError(
                f"CARPK sample {index} has {len(boxes)} boxes but {len(labels)} labels"
            )
        if args.overwrite or not destination.is_file():
            Image.fromarray(image).save(destination)
        records.append(
            {
                "dataset_index": index,
                "image_id": image_id,
                "image_path": relative_path.as_posix(),
                "boxes": boxes.tolist(),
                "box_format": "xywh",
                "target_count": len(labels),
            }
        )
        if index % 50 == 0 or index + 1 == len(dataset):
            print(f"[{index + 1}/{len(dataset)}] exported {image_id}", flush=True)

    split = args.split
    if split == "auto":
        split = "train" if "train" in args.uri.rsplit("/", 1)[-1].lower() else "test"
    manifest = {
        "dataset": "CARPK",
        "split": split,
        "source": args.uri,
        "images": len(records),
        "records": records,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(records)} CARPK records to {output_dir}")


if __name__ == "__main__":
    main(get_args_parser().parse_args())
