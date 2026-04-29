"""Training callback system.

Callbacks are invoked by the Trainer at key points during training.
Implement any subset of hooks — unimplemented hooks are no-ops.
"""

from __future__ import annotations

import csv
import logging
import numpy as np
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


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

    def on_validation_end(self, trainer: Any, epoch: int, metrics: dict) -> None:
        """Called after each validation run with the computed detection metrics.

        Args:
            trainer: The :class:`Trainer` instance.
            epoch: Zero-based epoch index.
            metrics: Dict produced by :meth:`~modern_yolonas.training.metrics.DetectionMetrics.compute`,
                containing keys such as ``mAP``, ``mAP_50``, ``mAR_100``.
        """

    def on_validation_images(
        self, trainer: Any, epoch: int, images: list[np.ndarray]
    ) -> None:
        """Called after each validation run with a sample of annotated images.

        Images have ground-truth boxes drawn in green and prediction boxes drawn
        in class colours on top, making it easy to compare the two at a glance.

        Args:
            trainer: The :class:`Trainer` instance.
            epoch: Zero-based epoch index.
            images: List of HWC uint8 BGR :class:`numpy.ndarray` images.
        """

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
    """Log training metrics to a CSV file (one row per epoch)."""

    def __init__(self, path: str | Path = "training_log.csv"):
        self.path = Path(path)
        self._writer = None
        self._file = None
        self._loss_acc: dict[str, float] = {}
        self._batch_count = 0

    def on_train_start(self, trainer):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["epoch", "loss", "cls_loss", "iou_loss", "dfl_loss", "lr"])

    def on_epoch_start(self, trainer, epoch):
        self._loss_acc = {}
        self._batch_count = 0

    def on_batch_end(self, trainer, epoch, batch_idx, loss, loss_dict):
        self._batch_count += 1
        self._loss_acc["loss"] = self._loss_acc.get("loss", 0.0) + loss
        for k, v in loss_dict.items():
            self._loss_acc[k] = self._loss_acc.get(k, 0.0) + v

    def on_epoch_end(self, trainer, epoch, avg_loss):
        n = max(self._batch_count, 1)
        lr = trainer.optimizer.param_groups[0]["lr"]
        self._writer.writerow([
            epoch + 1,
            f"{self._loss_acc.get('loss', 0.0) / n:.6f}",
            f"{self._loss_acc.get('cls_loss', 0.0) / n:.6f}",
            f"{self._loss_acc.get('iou_loss', 0.0) / n:.6f}",
            f"{self._loss_acc.get('dfl_loss', 0.0) / n:.6f}",
            f"{lr:.8f}",
        ])

    def on_train_end(self, trainer):
        if self._file:
            self._file.close()


class EarlyStoppingCallback(Callback):
    """Stop training when ``mAP@0.50:0.95`` stops improving for ``patience`` validations.

    Monitors the ``val_metrics/mAP`` key emitted by the trainer after each
    validation run (mirrors the ``metric_to_watch`` in the official YOLO-NAS
    training recipe).  Falls back to watching train loss via ``on_epoch_end``
    when no validation metrics have been received yet.

    Args:
        patience: Number of validation rounds without improvement before stopping.
        min_delta: Minimum absolute improvement in mAP to count as progress.
    """

    def __init__(self, patience: int = 20, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self._best_map = -1.0
        self._wait = 0
        self._has_val = False

    def on_validation_end(self, trainer, epoch, metrics):
        self._has_val = True
        current_map = metrics.get("val_metrics/mAP", 0.0)
        if current_map > self._best_map + self.min_delta:
            self._best_map = current_map
            self._wait = 0
        else:
            self._wait += 1
            if self._wait >= self.patience:
                trainer._stop_training = True

    def on_epoch_end(self, trainer, epoch, avg_loss):
        # Only used as a fallback when validation is disabled (val_freq=0)
        if not self._has_val:
            pass  # no-op: mAP-based monitoring takes priority


class WandbCallback(Callback):
    """Log training losses, learning rate, and validation metrics to `Weights & Biases`_.

    Requires ``wandb`` to be installed (``pip install wandb``).

    Each training step is logged under the ``"train/"`` prefix; validation metrics
    are logged under ``"val/"`` using the global step so that both streams share
    the same x-axis in the W&B UI.

    Args:
        project: W&B project name.
        name: Run display name.  If ``None`` W&B generates one automatically.
        config: Optional hyper-parameter dict attached to the run.
        tags: Optional list of tags for the run.
        resume_run_id: Existing run ID to resume.  When given, sets
            ``resume="must"`` in :func:`wandb.init`.

    .. _Weights & Biases: https://wandb.ai
    """

    def __init__(
        self,
        project: str,
        name: str | None = None,
        config: dict | None = None,
        tags: list[str] | None = None,
        resume_run_id: str | None = None,
    ) -> None:
        try:
            import wandb as _wandb  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "wandb is required for WandbCallback.  Install it with: pip install wandb"
            ) from exc

        self.project = project
        self.name = name
        self.config = config or {}
        self.tags = tags
        self.resume_run_id = resume_run_id
        self._run = None
        self._loss_acc: dict[str, float] = {}
        self._batch_count = 0

    def on_train_start(self, trainer: Any) -> None:
        import wandb

        init_kwargs: dict = dict(
            project=self.project,
            name=self.name,
            config=self.config,
            tags=self.tags,
        )
        if self.resume_run_id is not None:
            init_kwargs["id"] = self.resume_run_id
            init_kwargs["resume"] = "must"

        self._run = wandb.init(**init_kwargs)

    def on_epoch_start(self, trainer: Any, epoch: int) -> None:
        self._loss_acc = {}
        self._batch_count = 0

    def on_batch_end(
        self, trainer: Any, epoch: int, batch_idx: int, loss: float, loss_dict: dict
    ) -> None:
        self._batch_count += 1
        self._loss_acc["loss"] = self._loss_acc.get("loss", 0.0) + loss
        for k, v in loss_dict.items():
            self._loss_acc[k] = self._loss_acc.get(k, 0.0) + v

    def on_epoch_end(self, trainer: Any, epoch: int, avg_loss: float) -> None:
        import wandb

        n = max(self._batch_count, 1)
        wandb.log(
            {
                "train/loss":     self._loss_acc.get("loss", 0.0) / n,
                "train/cls_loss": self._loss_acc.get("cls_loss", 0.0) / n,
                "train/iou_loss": self._loss_acc.get("iou_loss", 0.0) / n,
                "train/dfl_loss": self._loss_acc.get("dfl_loss", 0.0) / n,
                "train/lr":       trainer.optimizer.param_groups[0]["lr"],
                "epoch":          epoch + 1,
            },
            step=epoch + 1,
            commit=False,
        )

    def on_validation_end(self, trainer: Any, epoch: int, metrics: dict) -> None:
        import wandb

        # metrics keys are already prefixed (e.g. "val/loss", "val metrics/mAP")
        wandb.log(dict(metrics), step=epoch + 1, commit=False)

    def on_validation_images(
        self, trainer: Any, epoch: int, images: list
    ) -> None:
        import wandb

        wandb.log(
            {
                "val/detections": [
                    wandb.Image(img[:, :, ::-1].copy(), caption=f"epoch {epoch + 1} [{i}]")
                    for i, img in enumerate(images)
                ]
            },
            step=epoch + 1,
            commit=False,
        )

    def on_train_end(self, trainer: Any) -> None:
        import wandb

        wandb.finish()
        self._run = None


class TensorBoardCallback(Callback):
    """Log training losses, learning rate, and validation metrics to TensorBoard.

    Requires ``tensorboard`` to be installed (``pip install tensorboard``).

    Training scalars are logged at every global step under ``"train/"``; validation
    metrics are logged after each evaluation run under ``"val/"``.

    The final event-file directory is ``<log_dir>/<experiment_name>``.  When you
    point ``tensorboard --logdir <log_dir>`` at the parent, every experiment
    appears as a separate run in the UI.

    Args:
        log_dir: Root directory for TensorBoard event files.
        experiment_name: Sub-directory name that identifies this run.  Defaults
            to ``"default"``; use a descriptive string such as
            ``"yolo_nas_s_coco_lr2e-4"`` to keep runs apart.
    """

    def __init__(
        self,
        log_dir: str | Path = "runs/tensorboard",
        experiment_name: str = "default",
    ) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "TensorBoard is required for TensorBoardCallback.  "
                "Install it with: pip install tensorboard"
            ) from exc

        self.log_dir = Path(log_dir) / experiment_name
        self._writer = None
        self._loss_acc: dict[str, float] = {}
        self._batch_count = 0

    def on_train_start(self, trainer: Any) -> None:
        from torch.utils.tensorboard import SummaryWriter

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(log_dir=str(self.log_dir))

    def on_epoch_start(self, trainer: Any, epoch: int) -> None:
        self._loss_acc = {}
        self._batch_count = 0

    def on_batch_end(
        self, trainer: Any, epoch: int, batch_idx: int, loss: float, loss_dict: dict
    ) -> None:
        self._batch_count += 1
        self._loss_acc["loss"] = self._loss_acc.get("loss", 0.0) + loss
        for k, v in loss_dict.items():
            self._loss_acc[k] = self._loss_acc.get(k, 0.0) + v

    def on_epoch_end(self, trainer: Any, epoch: int, avg_loss: float) -> None:
        if self._writer is None:
            return
        n = max(self._batch_count, 1)
        step = epoch + 1
        self._writer.add_scalar("train/loss",     self._loss_acc.get("loss", 0.0) / n,     step)
        self._writer.add_scalar("train/cls_loss", self._loss_acc.get("cls_loss", 0.0) / n, step)
        self._writer.add_scalar("train/iou_loss", self._loss_acc.get("iou_loss", 0.0) / n, step)
        self._writer.add_scalar("train/dfl_loss", self._loss_acc.get("dfl_loss", 0.0) / n, step)
        self._writer.add_scalar("train/lr",       trainer.optimizer.param_groups[0]["lr"],  step)

    def on_validation_end(self, trainer: Any, epoch: int, metrics: dict) -> None:
        if self._writer is None:
            return
        step = epoch + 1
        for key, value in metrics.items():
            self._writer.add_scalar(f"{key}", value, step)

    def on_validation_images(
        self, trainer: Any, epoch: int, images: list
    ) -> None:
        if self._writer is None:
            return
        step = epoch + 1
        for i, img_bgr in enumerate(images):
            # TensorBoard expects [C, H, W] uint8 or float; convert BGR → RGB
            img_rgb_chw = img_bgr[:, :, ::-1].transpose(2, 0, 1)
            self._writer.add_image(f"val/detections/{i}", img_rgb_chw, step)

    def on_train_end(self, trainer: Any) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
