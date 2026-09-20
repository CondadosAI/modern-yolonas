"""Backbone distillation: the cache contract and the training step.

The expensive failure here is silent. A cache read at the wrong offset, or a flip
that mirrors the grid instead of reading the mirrored forward, produces a loss
that falls perfectly well while teaching the wrong alignment.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest
import torch

from modern_yolonas.training.distill import (
    BackboneDistillModule,
    CachedFeatureDataset,
    distill_collate_fn,
)

GRID, DIM, SIZE, SHARD = 4, 8, 64, 3


def build_cache(tmp_path, count: int = 7, with_flip: bool = False,
                lengths: list[int] | None = None, record_lengths: bool = True):
    """A cache whose every feature is a recognisable constant, so a wrong offset shows.

    `lengths` builds shards of uneven size, which is what the writer actually
    produces: it flushes once the buffer reaches shard_size, and the buffer grows a
    batch at a time, so shards overshoot.
    """
    images = tmp_path / "images"
    cache = tmp_path / "cache"
    images.mkdir()
    cache.mkdir()

    names = []
    for i in range(count):
        name = f"{i:012d}.jpg"
        cv2.imwrite(str(images / name), np.full((48, SIZE, 3), i * 7 % 255, np.uint8))
        names.append(name)

    if lengths is None:
        lengths = [len(range(s0, min(s0 + SHARD, count))) for s0 in range(0, count, SHARD)]
    assert sum(lengths) == count

    shards = []
    start = 0
    for n in lengths:
        shard = f"shard_{len(shards):05d}"
        for suffix, offset in ((("", 0),) if not with_flip else (("", 0), ("_flip", 100))):
            # Value encodes the global index, so a misread is visible in the value.
            q = np.stack([
                np.full((GRID * GRID, DIM), (start + j + offset) % 127, np.int8)
                for j in range(n)
            ])
            np.save(cache / f"{shard}{suffix}.npy", q)
            np.save(cache / f"{shard}{suffix}_norm.npy", np.ones((n, GRID * GRID), np.float16))
        shards.append(shard)
        start += n

    index = {
        "teacher": "test", "size": SIZE, "patch": 16, "grid": GRID, "dim": DIM,
        "shard_size": SHARD, "files": names, "shards": shards, "has_flip": with_flip,
    }
    if record_lengths:
        index["shard_lengths"] = lengths
    (cache / "index.json").write_text(json.dumps(index))
    return images, cache, names


class TestCachedFeatureDataset:
    def test_shapes_and_dtypes(self, tmp_path):
        images, cache, _ = build_cache(tmp_path)
        image, features = CachedFeatureDataset(images, cache, flip_prob=0.0, hsv_prob=0.0)[0]
        assert image.shape == (3, SIZE, SIZE) and image.dtype == np.uint8
        assert features.shape == (DIM, GRID, GRID) and features.dtype == np.float32

    @pytest.mark.parametrize("index", range(7))
    def test_every_index_reads_its_own_features_across_shards(self, tmp_path, index):
        """The shard/offset split is the one place an off-by-one silently mislabels."""
        images, cache, _ = build_cache(tmp_path)
        _, features = CachedFeatureDataset(images, cache, flip_prob=0.0, hsv_prob=0.0)[index]
        assert features.min() == features.max() == pytest.approx(index % 127)

    def test_a_flip_reads_the_flipped_forward_not_a_mirrored_grid(self, tmp_path):
        """The flip shards hold different values, so reading the wrong one shows."""
        images, cache, _ = build_cache(tmp_path, with_flip=True)
        dataset = CachedFeatureDataset(images, cache, flip_prob=1.0, hsv_prob=0.0)
        _, features = dataset[2]
        assert features.min() == pytest.approx((2 + 100) % 127)

    def test_no_flip_reads_the_plain_forward(self, tmp_path):
        images, cache, _ = build_cache(tmp_path, with_flip=True)
        dataset = CachedFeatureDataset(images, cache, flip_prob=0.0, hsv_prob=0.0)
        _, features = dataset[2]
        assert features.min() == pytest.approx(2)

    def test_flipping_without_a_flip_cache_is_refused(self, tmp_path):
        """A ViT is not flip-equivariant, so mirroring the grid is not an option."""
        images, cache, _ = build_cache(tmp_path, with_flip=False)
        with pytest.raises(ValueError, match="not flip-equivariant"):
            CachedFeatureDataset(images, cache, flip_prob=0.5)

    def test_a_missing_image_is_reported_rather_than_skipped(self, tmp_path):
        images, cache, names = build_cache(tmp_path)
        (images / names[3]).unlink()
        with pytest.raises(FileNotFoundError, match=names[3]):
            CachedFeatureDataset(images, cache, flip_prob=0.0)

    def test_photometric_jitter_leaves_the_target_alone(self, tmp_path):
        """HSV moves no pixel, so the cached target stays valid — that is the premise."""
        images, cache, _ = build_cache(tmp_path)
        plain = CachedFeatureDataset(images, cache, flip_prob=0.0, hsv_prob=0.0)[1][1]
        jittered = CachedFeatureDataset(images, cache, flip_prob=0.0, hsv_prob=1.0)[1][1]
        assert np.array_equal(plain, jittered)


class TestBackboneDistillModule:
    @staticmethod
    def _module(**kwargs) -> BackboneDistillModule:
        from modern_yolonas import yolo_nas_s

        return BackboneDistillModule(
            model=yolo_nas_s(pretrained=False, num_classes=80), teacher_dim=384, **kwargs
        )

    def test_uint8_batches_are_scaled_on_device(self):
        module = self._module()
        images = torch.full((2, 3, 8, 8), 255, dtype=torch.uint8)
        out, _ = module.on_after_batch_transfer((images, torch.zeros(2, 384, 2, 2)), 0)
        assert out.dtype == torch.float32 and torch.allclose(out, torch.ones_like(out))

    def test_float_batches_pass_through(self):
        module = self._module()
        images = torch.rand(2, 3, 8, 8)
        out, _ = module.on_after_batch_transfer((images, torch.zeros(2, 384, 2, 2)), 0)
        assert torch.equal(out, images)

    def test_a_perfect_prediction_gives_zero_loss(self):
        """Cosine distance, so matching direction is all that is required."""
        module = self._module()
        images = torch.rand(1, 3, 128, 128)
        with torch.no_grad():
            _, _, c4, _ = module.model.backbone(images)
            target = module.projection(c4)
        loss = module._step((images, target), "train")
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_the_loss_falls_when_overfitting_one_batch(self):
        """The only test that shows the thing actually learns."""
        torch.manual_seed(0)
        module = self._module(lr=1e-3, warmup_steps=1, max_steps=40)
        images = torch.rand(2, 3, 128, 128)
        target = torch.randn(2, 384, 8, 8)

        optimizer = torch.optim.AdamW(
            list(module.model.backbone.parameters()) + list(module.projection.parameters()),
            lr=1e-3,
        )
        first = module._step((images, target), "train").item()
        for _ in range(30):
            optimizer.zero_grad()
            module._step((images, target), "train").backward()
            optimizer.step()
        assert module._step((images, target), "train").item() < first - 0.05

    def test_the_neck_is_left_out_of_the_optimiser_by_default(self):
        """It has no teacher to match; optimising it would only let it drift."""
        module = self._module()
        optimised = {id(p) for p in module.configure_optimizers()["optimizer"].param_groups[0]["params"]}
        assert not any(id(p) in optimised for p in module.model.neck.parameters())
        assert all(id(p) in optimised for p in module.model.backbone.parameters())


def test_collate_stacks_images_and_features():
    batch = [(np.zeros((3, 8, 8), np.uint8), np.ones((4, 2, 2), np.float32)) for _ in range(3)]
    images, features = distill_collate_fn(batch)
    assert images.shape == (3, 3, 8, 8) and images.dtype == torch.uint8
    assert features.shape == (3, 4, 2, 2)


class TestUnevenShards:
    """Shards do not hold `shard_size` images each, and assuming they do reads
    another image's features -- silently, until an index runs off the end.

    The real cache had 119 shards of 2016 and one of 1786 while the index said
    2000. Index 2016 resolved to shard 1 offset 16 instead of shard 1 offset 0, and
    the drift grew by 16 per shard. The IndexError only arrived at the last shard;
    a cache whose image count divided evenly would have trained a whole run against
    mismatched targets without raising.
    """

    UNEVEN = [5, 5, 3]  # 13 images: two full shards that overshoot, one short

    def test_every_index_finds_its_own_features(self, tmp_path):
        images, cache, _ = build_cache(tmp_path, count=13, lengths=self.UNEVEN)
        dataset = CachedFeatureDataset(images, cache, flip_prob=0.0, hsv_prob=0.0)
        for index in range(13):
            _, features = dataset[index]
            assert features.min() == features.max() == pytest.approx(index % 127), (
                f"index {index} got features for another image"
            )

    def test_shard_boundaries_land_exactly(self, tmp_path):
        images, cache, _ = build_cache(tmp_path, count=13, lengths=self.UNEVEN)
        dataset = CachedFeatureDataset(images, cache, flip_prob=0.0, hsv_prob=0.0)
        assert dataset._locate(0) == (0, 0)
        assert dataset._locate(4) == (0, 4)
        assert dataset._locate(5) == (1, 0)      # first image of the second shard
        assert dataset._locate(9) == (1, 4)
        assert dataset._locate(10) == (2, 0)
        assert dataset._locate(12) == (2, 2)     # last image overall

    def test_lengths_are_recovered_when_the_index_omits_them(self, tmp_path):
        """Caches written before `shard_lengths` existed must still load."""
        images, cache, _ = build_cache(tmp_path, count=13, lengths=self.UNEVEN,
                                       record_lengths=False)
        dataset = CachedFeatureDataset(images, cache, flip_prob=0.0, hsv_prob=0.0)
        assert dataset.shard_lengths == self.UNEVEN
        for index in (0, 5, 9, 12):
            _, features = dataset[index]
            assert features.min() == pytest.approx(index % 127)

    def test_a_cache_that_does_not_add_up_is_refused(self, tmp_path):
        """Better to refuse than to read past the end halfway through a run."""
        images, cache, _ = build_cache(tmp_path, count=13, lengths=self.UNEVEN)
        index = json.loads((cache / "index.json").read_text())
        index["shard_lengths"] = [5, 5, 2]
        (cache / "index.json").write_text(json.dumps(index))
        with pytest.raises(ValueError, match="shards hold"):
            CachedFeatureDataset(images, cache, flip_prob=0.0)

    def test_the_flip_shards_use_the_same_mapping(self, tmp_path):
        images, cache, _ = build_cache(tmp_path, count=13, lengths=self.UNEVEN,
                                       with_flip=True)
        dataset = CachedFeatureDataset(images, cache, flip_prob=1.0, hsv_prob=0.0)
        for index in (0, 5, 9, 12):
            _, features = dataset[index]
            assert features.min() == pytest.approx((index + 100) % 127)


class TestWhatGetsLogged:
    """A fourteen-hour run is only as useful as what it records."""

    @staticmethod
    def _module():
        from modern_yolonas import yolo_nas_s

        return BackboneDistillModule(
            model=yolo_nas_s(pretrained=False, num_classes=80), teacher_dim=384
        )

    def test_both_loss_and_cosine_are_logged(self, monkeypatch):
        """The loss is 1 - cosine: the quantity to minimise, not the one to read."""
        module = self._module()
        logged: dict[str, float] = {}
        monkeypatch.setattr(module, "log", lambda name, value, **kw: logged.__setitem__(
            name, float(value)
        ))
        images = torch.rand(1, 3, 128, 128)
        with torch.no_grad():
            _, _, c4, _ = module.model.backbone(images)
            target = module.projection(c4)
        module._step((images, target), "train")

        assert set(logged) == {"train/loss", "train/cosine"}
        assert logged["train/loss"] + logged["train/cosine"] == pytest.approx(1.0, abs=1e-5)

    def test_the_validation_stage_gets_its_own_names(self, monkeypatch):
        module = self._module()
        logged: list[str] = []
        monkeypatch.setattr(module, "log", lambda name, value, **kw: logged.append(name))
        images = torch.rand(1, 3, 128, 128)
        module._step((images, torch.randn(1, 384, 8, 8)), "val")
        assert logged == ["val/loss", "val/cosine"]
