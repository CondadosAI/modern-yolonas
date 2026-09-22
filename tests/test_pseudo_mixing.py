"""Training on real and pseudo-labelled COCO together.

Two failures here are silent. A pseudo-label file that lacks one rare class
derives a category mapping shifted by one from train2017's, so every later class
trains under the wrong label. And a mixed dataset that hides its children from
Mosaic would train on mosaics of one source only. Neither raises; both train.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from modern_yolonas import CROWD_CLASS
from modern_yolonas.data import COCODetectionDataset, ConcatDetectionDataset
from modern_yolonas.data.transforms import TrainTransformPipeline
from modern_yolonas.training.recipes import COCO_RECIPE
from modern_yolonas.training.run import build_transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from pseudo_label import filter_annotations, select_images  # noqa: E402

CATEGORIES = [{"id": 1, "name": "person"}, {"id": 3, "name": "car"}, {"id": 90, "name": "toothbrush"}]


def write_coco(tmp_path: Path, name: str, anns: list[tuple], n_images: int = 2, size: int = 64) -> tuple[Path, Path]:
    """Images of one flat colour per file, and ``(image_id, category_id, score, iscrowd)`` boxes."""
    root = tmp_path / name
    root.mkdir()
    images = []
    for i in range(1, n_images + 1):
        cv2.imwrite(str(root / f"{i}.jpg"), np.full((size, size, 3), i * 20, np.uint8))
        images.append({"id": i, "file_name": f"{i}.jpg", "height": size, "width": size})
    annotations = [
        {"id": k, "image_id": image_id, "category_id": cat, "bbox": [8, 8, 20, 20],
         "area": 400, "iscrowd": crowd, "score": score}
        for k, (image_id, cat, score, crowd) in enumerate(anns, start=1)
    ]
    ann_file = tmp_path / f"{name}.json"
    ann_file.write_text(json.dumps({"images": images, "annotations": annotations, "categories": CATEGORIES}))
    return root, ann_file


@pytest.fixture
def real(tmp_path):
    root, ann = write_coco(tmp_path, "real", [(1, 1, 1.0, 0), (1, 3, 1.0, 0), (2, 90, 1.0, 0)])
    return COCODetectionDataset(root, ann)


class TestCategoryMapping:
    def test_a_file_missing_a_class_derives_a_shifted_mapping(self, tmp_path, real):
        """The trap, shown directly: no 'car' in the file moves 'toothbrush' down one."""
        root, ann = write_coco(tmp_path, "pseudo", [(1, 1, 0.9, 0), (2, 90, 0.9, 0)])
        derived = COCODetectionDataset(root, ann)
        assert derived.cat_id_to_label[90] != real.cat_id_to_label[90]

    def test_an_explicit_mapping_keeps_the_real_labels(self, tmp_path, real):
        root, ann = write_coco(tmp_path, "pseudo", [(1, 1, 0.9, 0), (2, 90, 0.9, 0)])
        pseudo = COCODetectionDataset(root, ann, cat_id_to_label=real.cat_id_to_label)
        assert pseudo.cat_id_to_label == real.cat_id_to_label
        assert pseudo.class_names == real.class_names
        _, targets = pseudo.load_raw(1)
        assert targets[0, 0] == real.cat_id_to_label[90] == 2

    def test_a_category_outside_the_mapping_is_rejected(self, tmp_path):
        root, ann = write_coco(tmp_path, "pseudo", [(1, 3, 0.9, 0)])
        with pytest.raises(ValueError, match=r"\[3\]"):
            COCODetectionDataset(root, ann, cat_id_to_label={1: 0})


class TestConcat:
    @pytest.fixture
    def pair(self, tmp_path, real):
        root, ann = write_coco(tmp_path, "pseudo", [(1, 1, 0.9, 0), (2, 3, 0.9, 0), (3, 90, 0.9, 0)], n_images=3)
        pseudo = COCODetectionDataset(root, ann, cat_id_to_label=real.cat_id_to_label)
        return real, pseudo, ConcatDetectionDataset([real, pseudo])

    def test_indices_run_through_both_children_in_order(self, pair):
        real, pseudo, mixed = pair
        assert len(mixed) == len(real) + len(pseudo) == 5
        for index in range(len(mixed)):
            child, local = (real, index) if index < len(real) else (pseudo, index - len(real))
            image, targets = mixed.load_raw(index)
            want_image, want_targets = child.load_raw(local)
            np.testing.assert_array_equal(image, want_image)
            np.testing.assert_array_equal(targets, want_targets)

    def test_out_of_range_raises(self, pair):
        _, _, mixed = pair
        with pytest.raises(IndexError):
            mixed.load_raw(len(mixed))

    def test_a_different_mapping_is_rejected(self, tmp_path, real):
        root, ann = write_coco(tmp_path, "pseudo", [(1, 1, 0.9, 0), (2, 90, 0.9, 0)])
        with pytest.raises(ValueError, match="maps categories differently"):
            ConcatDetectionDataset([real, COCODetectionDataset(root, ann)])

    def test_mosaic_draws_from_both_children(self, pair):
        """The pipeline sits on the wrapper, so its extra tiles span the whole pool."""
        real, pseudo, mixed = pair
        mixed.transforms = build_transforms({**COCO_RECIPE, "input_size": 64}, train=True, dataset=mixed)
        assert isinstance(mixed.transforms, TrainTransformPipeline)
        calls = {"real": 0, "pseudo": 0}
        for name, child in (("real", real), ("pseudo", pseudo)):
            original = child.load_raw

            def counting(i, _name=name, _original=original):
                calls[_name] += 1
                return _original(i)

            child.load_raw = counting
        random.seed(0)
        image, targets = mixed[0]  # index 0 is a real image
        assert image.shape[-2:] == (64, 64)
        assert calls["pseudo"] > 0, "Mosaic never drew a pseudo-labelled tile"

    def test_close_mosaic_reaches_the_wrapper(self, pair):
        """CloseMosaicCallback finds the pipeline at ``dataloader.dataset.transforms``."""
        _, pseudo, mixed = pair
        mixed.transforms = build_transforms({**COCO_RECIPE, "input_size": 64}, train=True, dataset=mixed)
        mixed.transforms.disable_mosaic_mixup()
        assert not mixed.transforms.mosaic.enabled and not mixed.transforms.mixup.enabled

        def refuse(i):
            raise AssertionError("a closed mosaic still drew another image")

        pseudo.load_raw = refuse
        image, _ = mixed[0]
        assert image.shape[-2:] == (64, 64)


def labelled(scores: dict[int, list[float]]) -> dict:
    """Pseudo-label output with the given scores per image, all category 1."""
    annotations, k = [], 0
    for image_id, values in scores.items():
        for score in values:
            k += 1
            annotations.append({"id": k, "image_id": image_id, "category_id": 1,
                                "bbox": [8, 8, 20, 20], "area": 400, "iscrowd": 0, "score": score})
    images = [{"id": i, "file_name": f"{i}.jpg", "height": 64, "width": 64} for i in scores]
    return {"images": images, "annotations": annotations, "categories": CATEGORIES}


class TestFilter:
    def test_threshold_keeps_positives_and_drops_the_rest(self):
        out = filter_annotations(labelled({1: [0.9, 0.6, 0.35]}), positive=0.5)
        assert [a["score"] for a in out["annotations"]] == [0.9, 0.6]
        assert all(a["iscrowd"] == 0 for a in out["annotations"])

    def test_the_band_is_kept_as_ignore_regions(self):
        out = filter_annotations(labelled({1: [0.9, 0.6, 0.35, 0.2]}), positive=0.7, ignore_below=0.3)
        assert [(a["score"], a["iscrowd"]) for a in out["annotations"]] == [(0.9, 0), (0.6, 1), (0.35, 1)]

    def test_the_input_is_not_modified(self):
        data = labelled({1: [0.5]})
        filter_annotations(data, positive=0.7, ignore_below=0.3)
        assert data["annotations"][0]["iscrowd"] == 0

    def test_an_inverted_band_is_rejected(self):
        with pytest.raises(SystemExit, match="must be below"):
            filter_annotations(labelled({1: [0.9]}), positive=0.5, ignore_below=0.5)

    def test_the_boundary_is_a_positive(self):
        out = filter_annotations(labelled({1: [0.7]}), positive=0.7, ignore_below=0.3)
        assert out["annotations"][0]["iscrowd"] == 0

    def test_a_band_box_reaches_training_as_a_crowd_region(self, tmp_path):
        """End to end through the dataset: the band must arrive as CROWD_CLASS, not a class."""
        root = tmp_path / "img"
        root.mkdir()
        cv2.imwrite(str(root / "1.jpg"), np.zeros((64, 64, 3), np.uint8))
        out = filter_annotations(labelled({1: [0.9, 0.5]}), positive=0.7, ignore_below=0.3)
        ann = tmp_path / "f.json"
        ann.write_text(json.dumps(out))
        _, targets = COCODetectionDataset(root, ann, cat_id_to_label={1: 0}).load_raw(0)
        assert sorted(targets[:, 0].tolist()) == [CROWD_CLASS, 0]


class TestSubset:
    DATA = labelled({1: [0.9], 2: [0.6], 3: [0.75, 0.2], 4: [0.4], 5: [0.95], 6: [0.71]})

    def test_same_seed_same_images(self):
        assert select_images(self.DATA, 3, 0.7, seed=1) == select_images(self.DATA, 3, 0.7, seed=1)

    def test_only_images_with_a_strict_positive_are_eligible(self):
        chosen = select_images(self.DATA, 4, 0.7, seed=0)
        assert chosen == [1, 3, 5, 6]

    def test_every_arm_keeps_every_sampled_image(self):
        """Sampling at the strictest threshold means no arm empties an image."""
        chosen = select_images(self.DATA, 3, 0.7, seed=2)
        for positive, band in ((0.5, None), (0.7, None), (0.7, 0.3)):
            out = filter_annotations(self.DATA, positive, band, chosen)
            with_positive = {a["image_id"] for a in out["annotations"] if not a["iscrowd"]}
            assert with_positive == set(chosen) == {im["id"] for im in out["images"]}

    def test_asking_for_too_many_fails(self):
        with pytest.raises(SystemExit, match="only 4"):
            select_images(self.DATA, 5, 0.7, seed=0)
