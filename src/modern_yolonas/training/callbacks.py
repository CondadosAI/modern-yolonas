"""Training callback system.

Callbacks are invoked by the Trainer at key points during training.
Implement any subset of hooks — unimplemented hooks are no-ops.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


class Callback:
    """Base callback class. Override any hook method to customize behavior."""

    def on_train_start(self, trainer: Any) -> None:
        """Called once before the training loop begins."""

    def on_epoch_start(self, trainer: Any, epoch: int) -> None:
        """Called at the start of each epoch."""

    def on_batch_end(self, trainer: Any, epoch: int, batch_idx: int, loss: float, loss_dict: dict) -> None:
        """Called after each training batch."""

    def on_epoch_end(self, trainer: Any, epoch: int, avg_loss: float) -> None:
        """Called at the end of each epoch."""

    def on_train_end(self, trainer: Any) -> None:
        """Called once after training completes."""


class RichProgressCallback(Callback):
    """Display a Rich progress bar during training."""

    def on_train_start(self, trainer):
        from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn

        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
        )
        self._progress.start()
        self._epoch_task = None

    def on_epoch_start(self, trainer, epoch):
        total_batches = len(trainer.train_loader)
        desc = f"Epoch {epoch + 1}/{trainer.epochs}"
        if self._epoch_task is not None:
            self._progress.remove_task(self._epoch_task)
        self._epoch_task = self._progress.add_task(desc, total=total_batches)

    def on_batch_end(self, trainer, epoch, batch_idx, loss, loss_dict):
        self._progress.advance(self._epoch_task)

    def on_train_end(self, trainer):
        self._progress.stop()


class CSVLoggerCallback(Callback):
    """Log training metrics to a CSV file."""

    def __init__(self, path: str | Path = "training_log.csv"):
        self.path = Path(path)
        self._writer = None
        self._file = None

    def on_train_start(self, trainer):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["epoch", "batch", "loss", "cls_loss", "iou_loss", "dfl_loss", "lr"])

    def on_batch_end(self, trainer, epoch, batch_idx, loss, loss_dict):
        lr = trainer.optimizer.param_groups[0]["lr"]
        self._writer.writerow([
            epoch,
            batch_idx,
            f"{loss:.6f}",
            f"{loss_dict.get('cls_loss', 0):.6f}",
            f"{loss_dict.get('iou_loss', 0):.6f}",
            f"{loss_dict.get('dfl_loss', 0):.6f}",
            f"{lr:.8f}",
        ])

    def on_train_end(self, trainer):
        if self._file:
            self._file.close()


class EarlyStoppingCallback(Callback):
    """Stop training if loss doesn't improve for ``patience`` epochs."""

    def __init__(self, patience: int = 20, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self._best_loss = float("inf")
        self._wait = 0

    def on_epoch_end(self, trainer, epoch, avg_loss):
        if avg_loss < self._best_loss - self.min_delta:
            self._best_loss = avg_loss
            self._wait = 0
        else:
            self._wait += 1
            if self._wait >= self.patience:
                trainer._stop_training = True
