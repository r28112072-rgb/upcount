# Model weights

The `.pth` files in this directory are ignored by Git. `manifest.json` is the
source of truth for release filenames, lineage, metrics, hashes, and download
URLs.

To generate model-only release files from full training checkpoints:

```bash
python scripts/export_release_weights.py \
  --mae /path/to/checkpoint-500.pth \
  --fsc147 /path/to/checkpoint-432.pth \
  --carpk /path/to/checkpoint-best.pth \
  --output_dir weights
```

To download published files after URLs have been added to the manifest:

```bash
python scripts/download_weights.py --all
```

