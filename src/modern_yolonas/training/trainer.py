"""Training loop with DDP, AMP, EMA, checkpointing."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from rich.console import Console

from modern_yolonas.inference.postprocess import postprocess
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

        # EMA
        self.ema = ModelEMA(raw_model) if use_ema else None

        self.start_epoch = 0
        self.best_map = 0.0
        # Global optimisation step counter (incremented once per batch).
        # Persisted through checkpoints so that loggers keep a monotonic x-axis
        # when training is resumed.
        self.global_step = 0

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

                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    predictions = self.model(images)
                    loss, loss_dict = self.criterion(
                        predictions, targets,
                        input_size=(images.shape[2], images.shape[3]),
                        epoch=epoch,
                    )

                # Backward
                self.optimizer.zero_grad()
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
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
            if self.val_loader is not None and self.is_main and (epoch + 1) % 10 == 0:
                self._validate(epoch)

            # Save checkpoint
            if self.is_main:
                self._save_checkpoint(epoch)

        self._fire("on_train_end")

    @torch.no_grad()
    def _validate(self, epoch: int) -> dict[str, float]:
        """Run one pass over the validation set and compute detection metrics.

        Predictions are decoded from the model, filtered with NMS, then compared
        against ground-truth boxes using
        :class:`~modern_yolonas.training.metrics.DetectionMetrics`.  The best
        checkpoint (``best.pt``) is updated whenever ``mAP_50`` improves.

        Args:
            epoch: Zero-based epoch index (used for logging only).

        Returns:
            Dict produced by :meth:`~DetectionMetrics.compute`.
        """
        raw_model = self.model.module if isinstance(self.model, DDP) else self.model
        eval_model = self.ema.ema if self.ema is not None else raw_model
        eval_model.eval()

        metrics = DetectionMetrics(device=self.device)
        num_batches = 0

        for images, targets in self.val_loader:
            images  = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            pred_bboxes, pred_scores = eval_model(images)  # [B,N,4], [B,N,C]

            # Apply confidence filtering + per-class NMS
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
                t = targets[mask]  # [N_i, 6]: [batch_idx, cls, cx, cy, w, h] normalised
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

            metrics.update(preds, target_list)
            num_batches += 1

        results = metrics.compute()

        if self.is_main:
            console.print(
                f"  Val [{num_batches} batches] "
                f"mAP={results['mAP']:.4f}  "
                f"mAP_50={results['mAP_50']:.4f}  "
                f"mAP_75={results['mAP_75']:.4f}  "
                f"mAR_100={results['mAR_100']:.4f}"
            )

        # Persist the best checkpoint when mAP_50 improves
        if results["mAP_50"] > self.best_map:
            self.best_map = results["mAP_50"]
            self._save_checkpoint(epoch, is_best=True)

        self._fire("on_validation_end", epoch, results)
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
