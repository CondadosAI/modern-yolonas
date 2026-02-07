"""Quantization-Aware Training loop."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from rich.console import Console

from modern_yolonas.training.loss import PPYoloELoss
from modern_yolonas.training.ema import ModelEMA
from modern_yolonas.training.optimizer import create_optimizer
from modern_yolonas.training.scheduler import cosine_with_warmup
from modern_yolonas.quantization.prepare import QuantizedYoloNAS

console = Console()


class QATTrainer:
    """Quantization-Aware Training loop.

    Key differences from :class:`~modern_yolonas.training.trainer.Trainer`:

    * AMP is always **disabled** (fake-quant nodes conflict with autocast).
    * Lower default LR (2e-5 vs 2e-4).
    * Shorter default training (10 epochs).
    * Observer / BN statistics are frozen after a warmup period.
    * Single-GPU only (no DDP).

    Args:
        model: :class:`QuantizedYoloNAS` from :func:`prepare_model_qat`.
        train_loader: Training data.
        val_loader: Optional validation data.
        num_classes: Number of object classes.
        epochs: QAT fine-tuning epochs.
        lr: Learning rate.
        optimizer_name: ``'adamw'`` or ``'sgd'``.
        weight_decay: Weight decay.
        warmup_steps: LR warmup steps.
        freeze_bn_after_epoch: Freeze BN running stats after this epoch.
        freeze_observer_after_epoch: Stop updating fake-quant ranges.
        use_ema: Track EMA of model weights.
        output_dir: Checkpoint output directory.
        device: Training device.
    """

    def __init__(
        self,
        model: QuantizedYoloNAS,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        num_classes: int = 80,
        epochs: int = 10,
        lr: float = 2e-5,
        optimizer_name: str = "adamw",
        weight_decay: float = 1e-5,
        warmup_steps: int = 200,
        freeze_bn_after_epoch: int = 3,
        freeze_observer_after_epoch: int = 5,
        use_ema: bool = True,
        output_dir: str | Path = "runs/qat",
        device: str | torch.device = "cuda",
    ):
        self.epochs = epochs
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.freeze_bn_after_epoch = freeze_bn_after_epoch
        self.freeze_observer_after_epoch = freeze_observer_after_epoch

        model = model.to(self.device)
        model.train()
        self.model = model

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = PPYoloELoss(num_classes=num_classes)
        self.optimizer = create_optimizer(model, optimizer_name, lr, weight_decay)

        total_steps = epochs * len(train_loader)
        self.scheduler = cosine_with_warmup(
            self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )

        self.ema = ModelEMA(model) if use_ema else None
        self.start_epoch = 0

    def train(self):
        """Run QAT fine-tuning loop."""
        for epoch in range(self.start_epoch, self.epochs):
            self.model.train()

            # Freeze BN running stats after warmup
            if epoch >= self.freeze_bn_after_epoch:
                for m in self.model.backbone_neck.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()

            # Freeze observer / fake-quant range statistics
            if epoch >= self.freeze_observer_after_epoch:
                self.model.backbone_neck.apply(
                    torch.ao.quantization.disable_observer
                )

            epoch_loss = 0.0
            num_batches = 0

            console.print(f"\n[bold]QAT Epoch {epoch + 1}/{self.epochs}[/bold]")

            for batch_idx, (images, targets) in enumerate(self.train_loader):
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                # No AMP — fake-quant nodes are incompatible with autocast
                predictions = self.model(images)
                loss, loss_dict = self.criterion(predictions, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                if self.ema is not None:
                    self.ema.update(self.model)

                epoch_loss += loss.item()
                num_batches += 1

                if (batch_idx + 1) % 50 == 0:
                    avg_loss = epoch_loss / num_batches
                    lr = self.optimizer.param_groups[0]["lr"]
                    console.print(
                        f"  [{batch_idx + 1}/{len(self.train_loader)}] "
                        f"loss={avg_loss:.4f} "
                        f"cls={loss_dict['cls_loss']:.4f} "
                        f"iou={loss_dict['iou_loss']:.4f} "
                        f"dfl={loss_dict['dfl_loss']:.4f} "
                        f"lr={lr:.6f}"
                    )

            avg_loss = epoch_loss / max(num_batches, 1)
            console.print(f"  QAT Epoch {epoch + 1} avg loss: {avg_loss:.4f}")
            self._save_checkpoint(epoch)

    def _save_checkpoint(self, epoch: int):
        state = {
            "epoch": epoch + 1,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }
        if self.ema is not None:
            state["ema"] = self.ema.state_dict()

        torch.save(state, self.output_dir / "qat_last.pt")
