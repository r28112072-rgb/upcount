import json

import numpy as np
from PIL import Image

from util.CARPK import CARPKCropData, CARPKTestData


def test_manifest_xywh_boxes_become_resized_centers(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.fromarray(np.zeros((100, 200, 3), dtype=np.uint8)).save(
        images_dir / "000000.png"
    )
    manifest = {
        "source": "unit-test",
        "records": [
            {
                "image_id": "000000.png",
                "image_path": "images/000000.png",
                "boxes": [[10, 20, 30, 40], [100, 50, 20, 10]],
                "box_format": "xywh",
            }
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    image, points, image_id = CARPKTestData(tmp_path, resize_height=50)[0]

    assert image_id == "000000.png"
    assert image.shape == (3, 50, 100)
    np.testing.assert_allclose(points.numpy(), [[12.5, 20.0], [55.0, 27.5]])


def test_official_xyxy_annotations_and_split(tmp_path):
    images_dir = tmp_path / "Images"
    annotations_dir = tmp_path / "Annotations"
    image_sets_dir = tmp_path / "ImageSets"
    images_dir.mkdir()
    annotations_dir.mkdir()
    image_sets_dir.mkdir()
    Image.fromarray(np.zeros((40, 80, 3), dtype=np.uint8)).save(
        images_dir / "carpark.png"
    )
    (annotations_dir / "carpark.txt").write_text(
        "10 10 30 20\n40, 5, 60, 25\n", encoding="utf-8"
    )
    (image_sets_dir / "test.txt").write_text("carpark\n", encoding="utf-8")

    image, points, image_id = CARPKTestData(tmp_path, resize_height=0)[0]

    assert image_id == "carpark.png"
    assert image.shape == (3, 40, 80)
    np.testing.assert_allclose(points.numpy(), [[20, 15], [50, 15]])


def test_training_crop_preserves_scaled_density_integral(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.fromarray(np.zeros((80, 80, 3), dtype=np.uint8)).save(
        images_dir / "000000.png"
    )
    manifest = {
        "records": [
            {
                "image_id": "000000.png",
                "image_path": "images/000000.png",
                "boxes": [[30, 30, 10, 10]],
                "box_format": "xywh",
            }
        ]
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    image, density, count = CARPKCropData(
        tmp_path, [0], crop_size=64, focus_probability=1.0
    )[0]

    assert image.shape == (3, 64, 64)
    assert density.shape == (64, 64)
    assert count == 1
    np.testing.assert_allclose(density.sum().item(), 60.0, rtol=1e-5)
