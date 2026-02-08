"""PyTorch Lightning module for YOLO-NAS training."""

from __future__ import annotations

from pathlib import Path

import lightning as L
import torch
from torch import nn

from modern_yolonas.training.loss import PPYoloELoss
from modern_yolonas.training.optimizer import create_optimizer
from modern_yolonas.training.scheduler import cosine_with_warmup


class YoloNASLightningModule(L.LightningModule):
    """Lightning module wrapping a YOLO-NAS model for training.

    Args:
        model: YoloNAS or QuantizedYoloNAS model.
        num_classes: Number of object classes.
        lr: Learning rate.
        optimizer_name: Optimizer name ('adamw' or 'sgd').
        weight_decay: Weight decay.
        warmup_steps: LR warmup steps.
        cosine_final_lr_ratio: Final LR as fraction of initial LR.
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int = 80,
        lr: float = 2e-4,
        optimizer_name: str = "adamw",
        weight_decay: float = 1e-5,
        warmup_steps: int = 1000,
        cosine_final_lr_ratio: float = 0.1,
    ):
        super().__init__()
        self.model = model
        self.criterion = PPYoloELoss(num_classes=num_classes)
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        predictions = self.model(images)
        loss, loss_dict = self.criterion(predictions, targets)

        self.log("train/loss", loss, prog_bar=True)
        self.log("train/cls_loss", loss_dict["cls_loss"])
        self.log("train/iou_loss", loss_dict["iou_loss"])
        self.log("train/dfl_loss", loss_dict["dfl_loss"])

        if self._trainer is not None:
            schedulers = self.lr_schedulers()
            if schedulers is not None:
                lr = schedulers.get_last_lr()[0]
                self.log("train/lr", lr, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        predictions = self.model(images)
        loss, loss_dict = self.criterion(predictions, targets)

        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/cls_loss", loss_dict["cls_loss"], sync_dist=True)
        self.log("val/iou_loss", loss_dict["iou_loss"], sync_dist=True)
        self.log("val/dfl_loss", loss_dict["dfl_loss"], sync_dist=True)

        return loss

    def configure_optimizers(self):
        optimizer = create_optimizer(
            self.model,
            name=self.hparams.optimizer_name,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        total_steps = self.trainer.estimated_stepping_batches
        scheduler = cosine_with_warmup(
            optimizer,
            warmup_steps=self.hparams.warmup_steps,
            total_steps=total_steps,
            cosine_final_lr_ratio=self.hparams.cosine_final_lr_ratio,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


def extract_model_state_dict(checkpoint_path: str | Path) -> dict:
    """Load model weights from either a Lightning .ckpt or legacy .pt checkpoint.

    Handles:
    - Lightning format: state_dict keys prefixed with ``model.``
    - Legacy format: ``model_state_dict`` key or EMA ``ema.ema_state_dict``
    - Plain state_dict (e.g. from super-gradients pretrained weights)
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Lightning checkpoint format
    if "state_dict" in ckpt:
        sd = ckpt["state_dict"]
        # Strip 'model.' prefix added by LightningModule
        prefix = "model."
        stripped = {}
        for k, v in sd.items():
            if k.startswith(prefix):
                stripped[k[len(prefix):]] = v
            else:
                stripped[k] = v
        return stripped

    # Legacy format with EMA (prefer EMA weights if available)
    if "ema" in ckpt and "ema_state_dict" in ckpt["ema"]:
        return ckpt["ema"]["ema_state_dict"]

    # Legacy format with model_state_dict key
    if "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]

    # Plain state_dict
    return ckpt
