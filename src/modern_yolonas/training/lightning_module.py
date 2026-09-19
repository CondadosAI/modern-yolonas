"""PyTorch Lightning module for YOLO-NAS training."""

from __future__ import annotations

from pathlib import Path

import lightning as L
import torch
from torch import nn

from modern_yolonas.training.loss import PPYoloELoss
from modern_yolonas.training.optimizer import create_optimizer
from modern_yolonas.training.scheduler import cosine_with_warmup
from modern_yolonas.weights import extract_model_state_dict

__all__ = ["YoloNASLightningModule", "extract_model_state_dict"]


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
        input_size: Square size the images are letterboxed to. The evaluator needs it to
            map predictions back to original image coordinates before comparing them
            with the ground truth.
        channels_last: Hold activations in NHWC. Ampere and newer run convolutions
            under AMP faster in that layout — measured +15% on a 3060 Laptop,
            2026-09-19 — and it changes memory layout only, not results.
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
        input_size: int = 640,
        channels_last: bool = False,
    ):
        super().__init__()
        self.model = model.to(memory_format=torch.channels_last) if channels_last else model
        self.channels_last = channels_last
        self.criterion = PPYoloELoss(num_classes=num_classes)
        self.val_ann_file = val_ann_file
        self.val_dataset_ids = val_dataset_ids
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        self._evaluator = None
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x):
        return self.model(x)

    def on_after_batch_transfer(self, batch, dataloader_idx):
        """Finish preprocessing on the device, after the batch has crossed the bus.

        ``Normalize(dtype="uint8")`` leaves the divide-by-255 to this hook so that a
        quarter as many bytes cross ``pin_memory`` and PCIe. The check is on the dtype
        rather than on a flag, so a float32 pipeline passes through untouched and
        either kind of dataloader works against either kind of module.
        """
        images, targets = batch
        if images.dtype == torch.uint8:
            images = images.float().div_(255.0)
        if self.channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        return images, targets

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

            self._evaluator = COCOEvaluator(self.val_ann_file, input_size=self.input_size)
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
            # The loss needs the raw, undecoded predictions, which the head normally
            # returns only in training mode. Flipping the model to train() to get them
            # would let BatchNorm overwrite its running statistics with validation-set
            # statistics — `torch.no_grad` stops gradients, not buffer updates — and
            # then persist them into the checkpoint. `return_raw_outputs` changes the
            # return value without touching any module's training flag.
            heads = getattr(self.model, "heads", None)
            if heads is None:
                raise AttributeError(
                    "validation loss needs raw predictions from NDFLHeads, but the model "
                    "has no `heads` attribute"
                )
            heads.return_raw_outputs = True
            try:
                predictions = self.model(images)
                loss, loss_dict = self.criterion(predictions, targets)
            finally:
                heads.return_raw_outputs = False

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
