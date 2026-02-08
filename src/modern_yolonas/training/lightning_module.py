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
        val_ann_file: Path to COCO annotation JSON for mAP evaluation.
            When set, validation computes mAP instead of loss.
        val_dataset_ids: List of COCO image IDs matching validation dataset order.
            If None, extracted from the val dataloader's dataset.
        conf_threshold: Confidence threshold for mAP evaluation.
        iou_threshold: NMS IoU threshold for mAP evaluation.
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
        val_ann_file: str | Path | None = None,
        val_dataset_ids: list[int] | None = None,
        conf_threshold: float = 0.001,
        iou_threshold: float = 0.65,
    ):
        super().__init__()
        self.model = model
        self.criterion = PPYoloELoss(num_classes=num_classes)
        self.val_ann_file = val_ann_file
        self.val_dataset_ids = val_dataset_ids
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self._evaluator = None
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

    def on_validation_epoch_start(self):
        if self.val_ann_file is not None:
            from modern_yolonas.training.metrics import COCOEvaluator

            self._evaluator = COCOEvaluator(self.val_ann_file)
            # Cache image IDs from the val dataset if not provided
            if self.val_dataset_ids is None and self._trainer is not None:
                val_dl = self.trainer.val_dataloaders
                if val_dl is not None and hasattr(val_dl.dataset, "ids"):
                    self.val_dataset_ids = val_dl.dataset.ids

    def validation_step(self, batch, batch_idx):
        images, targets = batch

        # Model is in eval mode during validation (set by Lightning).
        # In eval mode, NDFLHeads returns (bboxes, scores) directly.
        # In train mode, it returns ((decoded, raw)).
        if self._evaluator is not None:
            from modern_yolonas.inference.postprocess import postprocess

            pred_bboxes, pred_scores = self.model(images)
            results = postprocess(
                pred_bboxes, pred_scores,
                conf_threshold=self.conf_threshold,
                iou_threshold=self.iou_threshold,
            )

            # Resolve image IDs for this batch
            batch_size = images.shape[0]
            start_idx = batch_idx * batch_size
            if self.val_dataset_ids is not None:
                image_ids = [
                    self.val_dataset_ids[start_idx + i]
                    for i in range(batch_size)
                    if start_idx + i < len(self.val_dataset_ids)
                ]
            else:
                image_ids = list(range(start_idx, start_idx + batch_size))

            boxes_list = [r[0] for r in results]
            scores_list = [r[1] for r in results]
            class_ids_list = [r[2] for r in results]
            self._evaluator.update(image_ids, boxes_list, scores_list, class_ids_list)
        else:
            # No mAP evaluator — run model in eval mode and compute loss
            # by forcing training mode temporarily for loss computation.
            self.model.train()
            predictions = self.model(images)
            loss, loss_dict = self.criterion(predictions, targets)
            self.model.eval()

            self.log("val/loss", loss, prog_bar=True, sync_dist=True)
            self.log("val/cls_loss", loss_dict["cls_loss"], sync_dist=True)
            self.log("val/iou_loss", loss_dict["iou_loss"], sync_dist=True)
            self.log("val/dfl_loss", loss_dict["dfl_loss"], sync_dist=True)

    def on_validation_epoch_end(self):
        if self._evaluator is not None:
            metrics = self._evaluator.evaluate()
            self.log("val/mAP", metrics["mAP"], prog_bar=True, sync_dist=True)
            self.log("val/mAP_50", metrics["mAP_50"], sync_dist=True)
            self.log("val/mAP_75", metrics["mAP_75"], sync_dist=True)
            self._evaluator = None

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
