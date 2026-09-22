"""CloseMosaicCallback has to reach the dataloader's worker processes.

It mutates the transform object in the *main* process. Today the workers see that
only because they are respawned at each epoch boundary; with
``persistent_workers=True`` the forked copies would keep their own
``enabled=True``, mosaic would never close, and nothing would raise. Training
continues, the loss falls, and the only symptom is a worse final AP -- the last
epochs train on composites when they were meant to train on whole images.

**How mosaic is detected here.** Each synthetic image is a single flat colour
encoding its index, so counting distinct colours in a sample counts the source
images that went into it. That is what mosaic *is*, rather than a proxy for it.

An earlier version of this test counted targets per sample. That works on COCO --
9.60 with mosaic against 6.31 without -- but only because COCO images are smaller
than the input size, so four of them crowd the canvas. With images the size of the
canvas, this Mosaic (which never resizes before pasting) crops back to roughly one
image's worth and the count goes *down*. The proxy held for the wrong reason.
"""

from __future__ import annotations

import lightning as L
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from modern_yolonas.data.base import BaseDetectionDataset
from modern_yolonas.data.collate import detection_collate_fn
from modern_yolonas.data.transforms import (
    Compose,
    HorizontalFlip,
    LetterboxResize,
    Mosaic,
    Normalize,
    TrainTransformPipeline,
)
from modern_yolonas.training.callbacks import CloseMosaicCallback

SIZE = 64
PAD = 114


class FlatColours(BaseDetectionDataset):
    """One flat colour per image, so a sample's palette names its sources."""

    def __init__(self, n: int = 48):
        super().__init__(transforms=None, input_size=SIZE)
        self.n = n

    def __len__(self) -> int:
        return self.n

    def load_raw(self, index: int):
        # Spread across the range and away from PAD, so the pad value is never
        # mistaken for a source image.
        value = 10 + (index * 5) % 90
        image = np.full((SIZE, SIZE, 3), value, np.uint8)
        targets = np.array([[0, 0.5, 0.5, 0.2, 0.2]], np.float32)
        return image, targets

    def __getitem__(self, index: int):
        """Mirrors the real datasets, which `BaseDetectionDataset` does not.

        `TrainTransformPipeline` has two entry points: `apply(index, load_raw)`,
        which runs Mosaic, and `__call__(image, targets)`, which silently does not.
        `BaseDetectionDataset.__getitem__` only calls the second, so a dataset that
        merely inherits it gets no mosaic and nothing says so. Both real datasets
        override with this branch.
        """
        if self.transforms is not None and hasattr(self.transforms, "apply"):
            return self.transforms.apply(index, self.load_raw)
        image, targets = self.load_raw(index)
        if self.transforms is not None:
            image, targets = self.transforms(image, targets)
        return image, targets


def sources_in(sample: torch.Tensor) -> int:
    """Distinct source images in one CHW uint8 sample, ignoring letterbox padding."""
    flat = sample.reshape(sample.shape[0], -1)[0]
    return len({int(v) for v in torch.unique(flat)} - {PAD})


class CountSources(L.LightningModule):
    def __init__(self, sink: dict[int, list[int]]):
        super().__init__()
        self.sink = sink
        self.layer = torch.nn.Linear(1, 1)

    def training_step(self, batch, batch_idx):
        images, _ = batch
        self.sink.setdefault(self.current_epoch, []).extend(
            sources_in(images[i]) for i in range(images.shape[0])
        )
        return self.layer(torch.zeros(1, 1, device=self.device)).sum() * 0.0

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.0)


def run(num_workers: int, epochs: int = 3, close: int = 1) -> dict[int, list[int]]:
    dataset = FlatColours()
    dataset.transforms = TrainTransformPipeline(
        mosaic=Mosaic(dataset, input_size=SIZE, prob=1.0),
        per_image_transforms=Compose([HorizontalFlip(p=0.5)]),
        mixup=None,
        final_transforms=Compose([LetterboxResize(target_size=SIZE), Normalize(dtype="uint8")]),
    )
    sink: dict[int, list[int]] = {}
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=num_workers,
                        collate_fn=detection_collate_fn)
    L.Trainer(
        max_epochs=epochs, accelerator="cpu", logger=False,
        enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
        callbacks=[CloseMosaicCallback(close_mosaic_epochs=close)],
    ).fit(CountSources(sink), loader)
    return sink


def split(sink, epochs: int, close: int) -> tuple[float, float]:
    before = [np.mean(sink[e]) for e in sink if e < epochs - close]
    after = [np.mean(sink[e]) for e in sink if e >= epochs - close]
    return float(np.mean(before)), float(np.mean(after))


@pytest.mark.parametrize("num_workers", [0, 2])
def test_mosaic_closes_and_the_workers_find_out(num_workers):
    """With 0 workers the callback mutates the object the loader reads directly.
    With 2 it has to cross a process boundary, which is the case that matters."""
    before, after = split(run(num_workers=num_workers), epochs=3, close=1)
    assert before > 1.5, f"mosaic was not composing: {before:.2f} sources per sample"
    assert after == pytest.approx(1.0, abs=0.01), (
        f"after closing, samples still draw on {after:.2f} source images with "
        f"num_workers={num_workers}; the callback did not reach them"
    )


def test_without_the_callback_mosaic_stays_on():
    """Guards the other end: if mosaic stopped on its own, the test above would
    pass for the wrong reason."""
    dataset = FlatColours()
    pipeline = TrainTransformPipeline(
        mosaic=Mosaic(dataset, input_size=SIZE, prob=1.0),
        per_image_transforms=Compose([HorizontalFlip(p=0.5)]),
        mixup=None,
        final_transforms=Compose([LetterboxResize(target_size=SIZE), Normalize(dtype="uint8")]),
    )
    dataset.transforms = pipeline
    counts = [sources_in(torch.from_numpy(dataset[i][0])) for i in range(24)]
    assert np.mean(counts) > 1.5


def test_zero_close_epochs_never_disables_it():
    """`close_mosaic_epochs=0` must leave mosaic on for every epoch, including the
    last. The callback's condition is `epoch >= max_epochs - close`, which with
    close=0 is `epoch >= max_epochs` -- never true."""
    sink = run(num_workers=2, epochs=3, close=0)
    per_epoch = [float(np.mean(sink[e])) for e in sorted(sink)]
    assert len(per_epoch) == 3
    assert all(c > 1.5 for c in per_epoch), f"mosaic stopped somewhere: {per_epoch}"


def test_the_callback_cannot_affect_epoch_zero():
    """`close_mosaic_epochs >= max_epochs` does not disable mosaic everywhere.

    The condition is `epoch >= max_epochs - close`, true at epoch 0 when
    close == max_epochs -- but epoch zero's workers are already iterating by the
    time `on_train_epoch_start` fires, so they never see the mutation. Measured:
    epoch 0 still composes, every later epoch does not.

    This matters for reading a misconfigured run. `close_mosaic_epochs` equal to
    the epoch count leaves exactly one epoch of mosaic, not zero, and the log
    reports the configuration rather than the effect.
    """
    sink = run(num_workers=2, epochs=3, close=3)
    per_epoch = [float(np.mean(sink[e])) for e in sorted(sink)]
    assert per_epoch[0] > 1.5, f"epoch 0 should still compose, got {per_epoch[0]:.2f}"
    assert all(c == pytest.approx(1.0, abs=0.01) for c in per_epoch[1:]), per_epoch


def test_a_normal_close_window_lands_on_the_intended_epochs():
    """For `close < max_epochs` the callback behaves as documented: exactly the
    last `close` epochs run without mosaic."""
    sink = run(num_workers=2, epochs=3, close=1)
    per_epoch = [float(np.mean(sink[e])) for e in sorted(sink)]
    assert per_epoch[0] > 1.5 and per_epoch[1] > 1.5
    assert per_epoch[2] == pytest.approx(1.0, abs=0.01)
