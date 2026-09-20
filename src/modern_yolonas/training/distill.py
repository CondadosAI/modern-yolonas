"""Distil DINOv3 into the backbone, using only COCO images.

This is what replaces Objects365 pretraining. Deci initialised YOLO-NAS on
Objects365's 2M images; that dataset's licence excludes us, and it is the single
largest contributor to the gap between a COCO-only run and their 47.5 AP. A
DINOv3 teacher supplies a comparable signal from COCO's own images, under a
licence that permits commercial use with attribution.

**The architecture does not change.** Only the initialisation does, so
``state_dict`` compatibility with super-gradients survives intact.

The alignment is exact rather than approximate: DINOv3 emits one patch token per
16x16 cell, and ``c4`` is stride 16 with 384 channels, which is DINOv3-S's
embedding width. At 640 both are 40x40x384, so the distillation head is a single
1x1 convolution.

Teacher features come from a cache built by ``tools/cache_teacher_features.py``
rather than from a teacher in the loop, which measured 36% of every step. The
cost of that choice is the augmentation budget -- see the module docstring there.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import lightning as L
import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset

from modern_yolonas.training.scheduler import cosine_with_warmup

__all__ = ["CachedFeatureDataset", "BackboneDistillModule"]


class CachedFeatureDataset(Dataset):
    """Images paired with their precomputed teacher features.

    Only augmentations a cached feature survives are applied: photometric ones,
    which move no pixel and so amount to invariance training, and -- when the cache
    was built with ``--with-flip`` -- the horizontal flip, which then reads a
    genuinely separate set of features.

    Mirroring the cached grid instead is wrong and silently so. A ViT carries
    positional embeddings and is not flip-equivariant: measured on val2017, a
    mirrored cache matches the teacher's own output for the mirrored image at
    cosine 0.867, where the unmirrored pair reaches 0.9999. Half the samples would
    have been trained against a target that is almost, but not, right.

    Args:
        images: Directories holding the images named in the cache index.
        cache: Directory written by ``tools/cache_teacher_features.py``.
        flip_prob: Probability of the horizontal flip. Requires a cache built with
            ``--with-flip``; raises otherwise rather than mirroring the grid.
        hsv_prob: Probability of the photometric jitter.
        pad_value: Letterbox fill, which must match the cache's.
    """

    def __init__(
        self,
        images: list[Path] | Path,
        cache: Path,
        flip_prob: float = 0.5,
        hsv_prob: float = 0.5,
        pad_value: int = 114,
    ):
        from modern_yolonas.data.transforms import HSVAugment

        self.cache = Path(cache)
        index = json.loads((self.cache / "index.json").read_text())
        self.size = index["size"]
        self.grid = index["grid"]
        self.dim = index["dim"]
        self.names: list[str] = index["files"]
        self.shards: list[str] = index["shards"]
        # Shards do not hold `shard_size` images each. The writer flushes once its
        # buffer *reaches* that figure, and the buffer grows one batch at a time, so
        # a shard overshoots to the next multiple of the batch size -- 2016 rather
        # than 2000 for COCO at batch 48, with the last one short.
        #
        # Dividing the index by shard_size therefore lands in the wrong shard at the
        # wrong offset, and returns another image's features. The error grows by the
        # overshoot with every shard, and an IndexError only arrives at the very end
        # -- so a cache whose image count divides evenly would never raise at all,
        # and would train the whole run against mismatched targets.
        #
        # Lengths are read from the index when present, and otherwise from the
        # shards themselves, so a cache written before this was recorded still works.
        lengths = index.get("shard_lengths")
        if not lengths:
            lengths = [
                int(np.load(self.cache / f"{name}.npy", mmap_mode="r").shape[0])
                for name in self.shards
            ]
        self.shard_lengths = lengths
        self._starts = np.cumsum([0, *lengths])
        if self._starts[-1] != len(self.names):
            raise ValueError(
                f"cache index lists {len(self.names)} images but the shards hold "
                f"{self._starts[-1]}"
            )

        directories = [Path(images)] if isinstance(images, (str, Path)) else [Path(p) for p in images]
        lookup = {p.name: p for directory in directories for p in directory.iterdir()}
        missing = [n for n in self.names if n not in lookup]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} images named in the cache index are not in {directories}; "
                f"first is {missing[0]}"
            )
        self.paths = [lookup[n] for n in self.names]

        self.has_flip = bool(index.get("has_flip", False))
        if flip_prob > 0 and not self.has_flip:
            raise ValueError(
                "flip_prob > 0 needs a cache built with --with-flip. Mirroring the cached "
                "grid is not equivalent: a ViT is not flip-equivariant, and the mirrored "
                "target matches at cosine 0.867 rather than 0.9999. Rebuild the cache, or "
                "set flip_prob=0."
            )
        self.flip_prob = flip_prob
        self.pad_value = pad_value
        self._hsv = HSVAugment(p=hsv_prob)
        # Opened lazily and per worker: a memmap handle cannot cross a fork safely.
        self._open: dict[tuple[int, bool], tuple[np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.paths)

    def _locate(self, index: int) -> tuple[int, int]:
        """Map a dataset index to (shard, offset within that shard)."""
        shard = int(np.searchsorted(self._starts, index, side="right") - 1)
        return shard, index - int(self._starts[shard])

    def _features(self, index: int, flipped: bool) -> np.ndarray:
        shard, offset = self._locate(index)
        key = (shard, flipped)
        if key not in self._open:
            name = self.shards[shard] + ("_flip" if flipped else "")
            self._open[key] = (
                np.load(self.cache / f"{name}.npy", mmap_mode="r"),
                np.load(self.cache / f"{name}_norm.npy", mmap_mode="r"),
            )
        quantized, norms = self._open[key]
        return quantized[offset].astype(np.float32) * norms[offset].astype(np.float32)[:, None]

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        image = cv2.imread(str(self.paths[index]))
        h, w = image.shape[:2]
        scale = self.size / max(h, w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        canvas = np.full((self.size, self.size, 3), self.pad_value, np.uint8)
        top, left = (self.size - new_h) // 2, (self.size - new_w) // 2
        canvas[top:top + new_h, left:left + new_w] = cv2.resize(
            image, (new_w, new_h), interpolation=cv2.INTER_LINEAR
        )

        empty = np.zeros((0, 5), np.float32)
        canvas, _ = self._hsv(canvas, empty)

        flipped = np.random.random() < self.flip_prob
        if flipped:
            canvas = canvas[:, ::-1]
        # A separate cached forward, never a mirrored grid.
        features = self._features(index, flipped).reshape(self.grid, self.grid, self.dim)

        image_chw = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1))
        return image_chw, np.ascontiguousarray(features.transpose(2, 0, 1))


def distill_collate_fn(batch: list[tuple[np.ndarray, np.ndarray]]) -> tuple[Tensor, Tensor]:
    images, features = zip(*batch)
    return (
        torch.from_numpy(np.stack(images)),
        torch.from_numpy(np.stack(features)),
    )


class BackboneDistillModule(L.LightningModule):
    """Train the backbone to reproduce the teacher's patch features.

    Args:
        model: A ``YoloNAS``. Only its backbone is trained; the neck and heads
            come along so the result is a checkpoint the detection stage can load
            without any surgery.
        teacher_dim: Channels of the cached features.
        lr: Peak learning rate.
        warmup_steps: Linear warmup before the cosine decay.
        max_steps: Total steps, for the schedule.
        train_neck: Include the neck in the optimised parameters. Off by default:
            the neck has no teacher to match and would only drift.
    """

    def __init__(
        self,
        model: nn.Module,
        teacher_dim: int = 384,
        lr: float = 1e-3,
        weight_decay: float = 0.05,
        warmup_steps: int = 1000,
        max_steps: int = 100_000,
        train_neck: bool = False,
    ):
        super().__init__()
        self.model = model
        self.projection = nn.Conv2d(model.backbone.out_channels[2], teacher_dim, 1)
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.train_neck = train_neck
        self.save_hyperparameters(ignore=["model"])

    def on_after_batch_transfer(self, batch, dataloader_idx):
        """Scale on the device, as the detection path does."""
        images, features = batch
        if images.dtype == torch.uint8:
            images = images.float().div_(255.0)
        return images, features

    def _step(self, batch, stage: str) -> Tensor:
        images, target = batch
        # The backbone alone: the neck and heads have no teacher here.
        _, _, c4, _ = self.model.backbone(images)
        predicted = self.projection(c4)

        similarity = nn.functional.cosine_similarity(predicted, target, dim=1).mean()
        loss = 1.0 - similarity

        self.log(f"{stage}/loss", loss, prog_bar=True, sync_dist=True)
        # The loss is 1 - cosine, which is the quantity to minimise but not the one
        # to read. Alignment is what the stage is for, and a cosine of 0.81 says
        # something a loss of 0.19 does not.
        self.log(f"{stage}/cosine", similarity, prog_bar=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx) -> Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx) -> Tensor:
        return self._step(batch, "val")

    def configure_optimizers(self):
        parameters = list(self.model.backbone.parameters()) + list(self.projection.parameters())
        if self.train_neck:
            parameters += list(self.model.neck.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = cosine_with_warmup(
            optimizer, warmup_steps=self.warmup_steps, total_steps=self.max_steps
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
