"""Training callback system.

Callbacks are invoked by the Trainer at key points during training.
Implement any subset of hooks — unimplemented hooks are no-ops.
"""

from __future__ import annotations

import csv
import logging
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
                containing keys such as ``mAP``, ``mAP_50``, ``mAP_75``, ``mAR_100``, etc.
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
        log_every: Log training scalars every *n* global steps (default: 1).
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
        log_every: int = 1,
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
        self.log_every = log_every
        self.resume_run_id = resume_run_id
        self._run = None

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

    def on_batch_end(
        self, trainer: Any, epoch: int, batch_idx: int, loss: float, loss_dict: dict
    ) -> None:
        step = trainer.global_step
        if step % self.log_every != 0:
            return

        import wandb

        wandb.log(
            {
                "train/loss":     loss,
                "train/cls_loss": loss_dict.get("cls_loss", 0.0),
                "train/iou_loss": loss_dict.get("iou_loss", 0.0),
                "train/dfl_loss": loss_dict.get("dfl_loss", 0.0),
                "train/lr":       trainer.optimizer.param_groups[0]["lr"],
            },
            step=step,
        )

    def on_epoch_end(self, trainer: Any, epoch: int, avg_loss: float) -> None:
        import wandb

        wandb.log(
            {"train/epoch_loss": avg_loss, "epoch": epoch + 1},
            step=trainer.global_step,
        )

    def on_validation_end(self, trainer: Any, epoch: int, metrics: dict) -> None:
        import wandb

        wandb.log(
            {f"val/{k}": v for k, v in metrics.items()},
            step=trainer.global_step,
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

    Args:
        log_dir: Directory where TensorBoard event files are written.
        log_every: Write training scalars every *n* global steps (default: 1).
    """

    def __init__(
        self,
        log_dir: str | Path = "runs/tensorboard",
        log_every: int = 1,
    ) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "TensorBoard is required for TensorBoardCallback.  "
                "Install it with: pip install tensorboard"
            ) from exc

        self.log_dir = Path(log_dir)
        self.log_every = log_every
        self._writer = None

    def on_train_start(self, trainer: Any) -> None:
        from torch.utils.tensorboard import SummaryWriter

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(log_dir=str(self.log_dir))

    def on_batch_end(
        self, trainer: Any, epoch: int, batch_idx: int, loss: float, loss_dict: dict
    ) -> None:
        step = trainer.global_step
        if step % self.log_every != 0 or self._writer is None:
            return

        self._writer.add_scalar("train/loss",     loss,                                step)
        self._writer.add_scalar("train/cls_loss", loss_dict.get("cls_loss", 0.0),      step)
        self._writer.add_scalar("train/iou_loss", loss_dict.get("iou_loss", 0.0),      step)
        self._writer.add_scalar("train/dfl_loss", loss_dict.get("dfl_loss", 0.0),      step)
        self._writer.add_scalar("train/lr",       trainer.optimizer.param_groups[0]["lr"], step)

    def on_epoch_end(self, trainer: Any, epoch: int, avg_loss: float) -> None:
        if self._writer is None:
            return
        self._writer.add_scalar("train/epoch_loss", avg_loss, trainer.global_step)

    def on_validation_end(self, trainer: Any, epoch: int, metrics: dict) -> None:
        if self._writer is None:
            return
        step = trainer.global_step
        for key, value in metrics.items():
            self._writer.add_scalar(f"val/{key}", value, step)

    def on_train_end(self, trainer: Any) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
