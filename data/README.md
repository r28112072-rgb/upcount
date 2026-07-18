# Data preparation

Dataset files are not redistributed by this repository. Please follow each
dataset's license and access terms.

## FSC-147

The training and evaluation scripts use the CounTR-compatible 384-pixel FSC-147
layout documented in the main README:

```text
data/FSC147/
├── images_384_VarV2/
├── gt_density_map_adaptive_384_VarV2/
├── annotation_FSC147_384.json
├── Train_Test_Val_FSC_147.json
└── ImageClasses_FSC147.txt
```

The density maps are NumPy arrays whose integral is 60 times the object count.
The split JSON maps `train`, `val`, and `test` to image filenames.

## CARPK

Each CARPK split contains `images/` and `manifest.json`. The manifest has this
minimal structure:

```json
{
  "dataset": "CARPK",
  "split": "test",
  "records": [
    {
      "image_id": "000000.png",
      "image_path": "images/000000.png",
      "boxes": [[120.0, 80.0, 20.0, 40.0]],
      "box_format": "xywh",
      "target_count": 1
    }
  ]
}
```

`prepare_carpk.py` generates this layout from the public Deep Lake mirror.

