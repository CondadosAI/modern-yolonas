"""Lightning DataModule for detection datasets."""

from __future__ import annotations

import lightning as L
from torch.utils.data import DataLoader, Dataset

from modern_yolonas.data.collate import detection_collate_fn


class DetectionDataModule(L.LightningDataModule):
    """DataModule wrapping training and validation detection datasets.

    Args:
        train_dataset: Training dataset.
        val_dataset: Optional validation dataset.
        batch_size: Batch size per device.
        num_workers: DataLoader workers.
    """

    def __init__(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset | None = None,
        batch_size: int = 32,
        num_workers: int = 8,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=detection_collate_fn,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=detection_collate_fn,
            pin_memory=True,
        )
