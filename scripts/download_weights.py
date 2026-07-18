"""Download and SHA-256 verify UPCount release weights."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Download UPCount weights")
    parser.add_argument("ids", nargs="*", help="weight ids from weights/manifest.json")
    parser.add_argument("--all", action="store_true", help="download every weight")
    parser.add_argument("--manifest", default=Path("weights/manifest.json"), type=Path)
    parser.add_argument("--output_dir", default=Path("weights"), type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(args: argparse.Namespace) -> None:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    entries = {entry["id"]: entry for entry in manifest["weights"]}
    selected = list(entries) if args.all else args.ids
    if not selected:
        raise SystemExit("provide one or more weight ids, or pass --all")
    unknown = sorted(set(selected) - set(entries))
    if unknown:
        raise SystemExit(f"unknown weight ids: {', '.join(unknown)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for weight_id in selected:
        entry = entries[weight_id]
        url = entry.get("url")
        expected = entry.get("sha256")
        if not url or not expected:
            raise SystemExit(
                f"{weight_id} has no published URL/checksum yet; see weights/README.md"
            )
        destination = args.output_dir / entry["filename"]
        if destination.exists() and not args.overwrite:
            if sha256(destination) == expected:
                print(f"verified existing {destination}")
                continue
            raise SystemExit(f"existing file has the wrong checksum: {destination}")
        temporary = destination.with_suffix(destination.suffix + ".part")
        print(f"downloading {url} -> {destination}")
        with urlopen(url) as response, temporary.open("wb") as handle:
            while chunk := response.read(8 * 1024 * 1024):
                handle.write(chunk)
        actual = sha256(temporary)
        if actual != expected:
            temporary.unlink(missing_ok=True)
            raise SystemExit(
                f"checksum mismatch for {weight_id}: expected {expected}, got {actual}"
            )
        temporary.replace(destination)
        print(f"verified {destination}")


if __name__ == "__main__":
    main(get_args_parser().parse_args())

