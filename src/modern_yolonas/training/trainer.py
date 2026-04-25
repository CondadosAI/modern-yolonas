"""Training loop with DDP, AMP, EMA, checkpointing."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from rich.console import Console

from modern_yolonas.inference.postprocess import postprocess
from modern_yolonas.inference.visualize import annotate_validation_sample
from modern_yolonas.training.callbacks import Callback
from modern_yolonas.training.loss import PPYoloELoss
from modern_yolonas.training.ema import ModelEMA
from modern_yolonas.training.metrics import DetectionMetrics
from modern_yolonas.training.optimizer import create_optimizer
from modern_yolonas.training.scheduler import cosine_with_warmup

console = Console()


class Trainer:
    """YOLO-NAS training loop.

    Args:
        model: YoloNAS model.
        train_loader: Training DataLoader.
        val_loader: Optional validation DataLoader.
        num_classes: Number of classes.
        epochs: Total training epochs.
        lr: Learning rate.
        optimizer_name: Optimizer name ('adamw' or 'sgd').
        weight_decay: Weight decay.
        warmup_steps: LR warmup steps.
        use_amp: Enable automatic mixed precision.
        use_ema: Enable exponential moving average.
        output_dir: Directory for checkpoints.
        device: Training device.
        local_rank: Local rank for DDP (-1 for single GPU).
        class_names: Optional list of class names used when drawing validation
            images.  When ``None`` the visualiser falls back to COCO names.
        val_freq: Run validation every *n* epochs (default: 10).  Set to 1 to
            validate after every epoch, or to a larger value to reduce overhead.
        val_vis_images: Number of sample images to annotate and log to
            WandB / TensorBoard during each validation run.  Set to 0 to
            disable image logging.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        num_classes: int = 80,
        epochs: int = 300,
        lr: float = 2e-4,
        optimizer_name: str = "adamw",
        weight_decay: float = 1e-5,
        warmup_steps: int = 1000,
        use_amp: bool = True,
        use_ema: bool = True,
        output_dir: str | Path = "runs/train",
        device: str | torch.device = "cuda",
        local_rank: int = -1,
        callbacks: list[Callback] | None = None,
        class_names: list[str] | None = None,
        val_freq: int = 10,
        val_vis_images: int = 8,
    ):
        self.epochs = epochs
        self.callbacks = callbacks or []
        self._stop_training = False
        self.device = torch.device(device)
        self.local_rank = local_rank
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.is_main = local_rank <= 0

        # DDP setup
        if local_rank >= 0:
            model = model.to(self.device)
            model = DDP(model, device_ids=[local_rank])
        else:
            model = model.to(self.device)

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Loss
        self.criterion = PPYoloELoss(num_classes=num_classes)

        # Optimizer
        raw_model = model.module if isinstance(model, DDP) else model
        self.optimizer = create_optimizer(raw_model, optimizer_name, lr, weight_decay)

        # Scheduler
        total_steps = epochs * len(train_loader)
        self.scheduler = cosine_with_warmup(
            self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )

        # AMP
        self.use_amp = use_amp
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

        # Speed: let cuDNN auto-select fastest convolution algorithms for
        # the fixed input size used throughout training.
        # if self.device.type == "cuda":
        #     torch.backends.cudnn.benchmark = True

        # EMA
        self.ema = ModelEMA(raw_model) if use_ema else None

        self.start_epoch = 0
        self.best_map = 0.0
        # Global optimisation step counter (incremented once per batch).
        # Persisted through checkpoints so that loggers keep a monotonic x-axis
        # when training is resumed.
        self.global_step = 0
        self.class_names = class_names
        self.val_freq = val_freq
        self.val_vis_images = val_vis_images

    def _fire(self, hook: str, *args, **kwargs):
        for cb in self.callbacks:
            getattr(cb, hook)(self, *args, **kwargs)

    def train(self):
        """Run training loop."""
        self._fire("on_train_start")
        for epoch in range(self.start_epoch, self.epochs):
            if self._stop_training:
                break
            self.model.train()

            if isinstance(self.train_loader.sampler, DistributedSampler):
                self.train_loader.sampler.set_epoch(epoch)

            epoch_loss = 0.0
            num_batches = 0

            self._fire("on_epoch_start", epoch)
            if self.is_main:
                console.print(f"\n[bold]Epoch {epoch + 1}/{self.epochs}[/bold]")

            for batch_idx, (images, targets) in enumerate(self.train_loader):
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                with torch.amp.autocast(str(self.device), enabled=self.use_amp):
                    predictions = self.model(images)
                    loss, loss_dict = self.criterion(
                        predictions, targets,
                        input_size=(images.shape[2], images.shape[3]),
                        epoch=epoch,
                    )

                # Backward
                self.optimizer.zero_grad(set_to_none=True)
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    # self.scaler.unscale_(self.optimizer)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                self.scheduler.step()

                if self.ema is not None:
                    raw_model = self.model.module if isinstance(self.model, DDP) else self.model
                    self.ema.update(raw_model)

                epoch_loss += loss.item()
                num_batches += 1
                self.global_step += 1

                self._fire("on_batch_end", epoch, batch_idx, loss.item(), loss_dict)

                if self.is_main and (batch_idx + 1) % 50 == 0:
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
            if self.is_main:
                console.print(f"  Epoch {epoch + 1} avg loss: {avg_loss:.4f}")

            self._fire("on_epoch_end", epoch, avg_loss)

            # Validation
            if self.val_loader is not None and self.is_main and (epoch + 1) % self.val_freq == 0:
                self._validate(epoch)

            # Save checkpoint
            if self.is_main:
                self._save_checkpoint(epoch)

        self._fire("on_train_end")

    @torch.no_grad()
    def _validate(self, epoch: int) -> dict[str, float]:
        """Run one pass over the validation set, computing losses and detection metrics.

        The model is called in training mode (so raw predictions are available for
        the loss) but inside :func:`torch.no_grad`, so no gradients are accumulated.
        Predicted boxes decoded in eval mode are used separately for mAP/mAR.

        Args:
            epoch: Zero-based epoch index (used for logging only).

        Returns:
            Dict with keys ``"val/loss"``, ``"val/cls_loss"``, ``"val/iou_loss"``,
            ``"val/dfl_loss"`` (averaged over batches) plus ``"val metrics/mAP"``,
            ``"val metrics/mAP_50"``, ``"val metrics/mAR_100"``.
        """
        raw_model = self.model.module if isinstance(self.model, DDP) else self.model
        eval_model = self.ema.ema if self.ema is not None else raw_model

        # Training mode gives raw predictions needed for the loss; no gradients
        # are computed because we are inside torch.no_grad().
        eval_model.train()

        det_metrics = DetectionMetrics(device=self.device)
        num_batches = 0
        vis_images: list = []

        loss_sum: dict[str, float] = {"total": 0.0, "cls": 0.0, "iou": 0.0, "dfl": 0.0}

        for images, targets in self.val_loader:
            images  = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            with torch.amp.autocast(str(self.device), enabled=self.use_amp):
                predictions = eval_model(images)
                loss, loss_dict = self.criterion(
                    predictions, targets,
                    input_size=(images.shape[2], images.shape[3]),
                    epoch=epoch,
                )

            loss_sum["total"] += loss.item()
            loss_sum["cls"]   += loss_dict.get("cls_loss", 0.0)
            loss_sum["iou"]   += loss_dict.get("iou_loss", 0.0)
            loss_sum["dfl"]   += loss_dict.get("dfl_loss", 0.0)

            # Decoded predictions (pred_bboxes, pred_scores) are the first element
            pred_bboxes, pred_scores = predictions[0]
            detections = postprocess(pred_bboxes, pred_scores)
            preds = [
                {
                    "boxes":  boxes.float(),
                    "scores": scores.float(),
                    "labels": labels.int(),
                }
                for boxes, scores, labels in detections
            ]

            # Convert collated targets [sum_N, 6] → per-image xyxy dicts
            batch_size = images.shape[0]
            img_h, img_w = images.shape[2], images.shape[3]
            target_list: list[dict] = []
            for i in range(batch_size):
                mask = targets[:, 0] == i
                t = targets[mask]
                if t.shape[0] == 0:
                    target_list.append({
                        "boxes":  torch.zeros(0, 4, device=self.device),
                        "labels": torch.zeros(0, dtype=torch.int, device=self.device),
                    })
                else:
                    cx = t[:, 2] * img_w
                    cy = t[:, 3] * img_h
                    bw = t[:, 4] * img_w
                    bh = t[:, 5] * img_h
                    boxes = torch.stack(
                        [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], dim=-1
                    )
                    target_list.append({
                        "boxes":  boxes,
                        "labels": t[:, 1].int(),
                    })

            det_metrics.update(preds, target_list)
            num_batches += 1

            # Collect visual samples from the first batch only
            if num_batches == 1 and self.val_vis_images > 0:
                import numpy as np
                n_vis = min(self.val_vis_images, images.shape[0])
                imgs_np = images[:n_vis].cpu().numpy()
                for idx in range(n_vis):
                    p_boxes  = preds[idx]["boxes"].cpu().numpy()
                    p_scores = preds[idx]["scores"].cpu().numpy()
                    p_labels = preds[idx]["labels"].cpu().numpy()
                    g_boxes  = target_list[idx]["boxes"].cpu().numpy()
                    g_labels = target_list[idx]["labels"].cpu().numpy()
                    vis_images.append(
                        annotate_validation_sample(
                            imgs_np[idx], p_boxes, p_scores, p_labels, g_boxes, g_labels,
                            class_names=self.class_names,
                        )
                    )

        # Restore eval mode for any subsequent inference
        eval_model.eval()

        n = max(num_batches, 1)
        map_results = det_metrics.compute()

        results = {
            # Losses — averaged across all validation batches
            "val/loss":     loss_sum["total"] / n,
            "val/cls_loss": loss_sum["cls"]   / n,
            "val/iou_loss": loss_sum["iou"]   / n,
            "val/dfl_loss": loss_sum["dfl"]   / n,
            # Detection metrics — in their own group
            "val_metrics/mAP":     map_results["mAP"],
            "val_metrics/mAP_50":  map_results["mAP_50"],
            "val_metrics/mAR_100": map_results["mAR_100"],
        }

        if self.is_main:
            console.print(
                f"  Val [{num_batches} batches] "
                f"loss={results['val/loss']:.4f}  "
                f"cls={results['val/cls_loss']:.4f}  "
                f"iou={results['val/iou_loss']:.4f}  "
                f"dfl={results['val/dfl_loss']:.4f}  "
                f"mAP={results['val_metrics/mAP']:.4f}  "
                f"mAP_50={results['val_metrics/mAP_50']:.4f}  "
                f"mAR_100={results['val_metrics/mAR_100']:.4f}"
            )

        # Persist the best checkpoint when mAP_50 improves
        if results["val_metrics/mAP_50"] > self.best_map:
            self.best_map = results["val_metrics/mAP_50"]
            self._save_checkpoint(epoch, is_best=True)

        self._fire("on_validation_end", epoch, results)
        if vis_images:
            self._fire("on_validation_images", epoch, vis_images)
        return results

    def _save_checkpoint(self, epoch: int, *, is_best: bool = False):
        raw_model = self.model.module if isinstance(self.model, DDP) else self.model
        state = {
            "epoch": epoch + 1,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_map": self.best_map,
            "global_step": self.global_step,
        }
        if self.ema is not None:
            state["ema"] = self.ema.state_dict()
        if self.scaler is not None:
            state["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(state, self.output_dir / "last.pt")
        if is_best:
            torch.save(state, self.output_dir / "best.pt")
        if (epoch + 1) % 50 == 0:
            torch.save(state, self.output_dir / f"epoch_{epoch + 1}.pt")

    def resume(self, checkpoint_path: str | Path):
        """Resume training from checkpoint."""
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)

        raw_model = self.model.module if isinstance(self.model, DDP) else self.model
        raw_model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.start_epoch = ckpt["epoch"]
        self.best_map = ckpt.get("best_map", 0.0)
        self.global_step = ckpt.get("global_step", self.start_epoch * len(self.train_loader))

        if self.ema is not None and "ema" in ckpt:
            self.ema.load_state_dict(ckpt["ema"])
        if self.scaler is not None and "scaler_state_dict" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])

        console.print(f"Resumed from epoch {self.start_epoch}")
