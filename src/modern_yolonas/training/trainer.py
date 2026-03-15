"""Training loop with DDP, AMP, EMA, checkpointing."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from rich.console import Console

from modern_yolonas.training.loss import PPYoloELoss
from modern_yolonas.training.ema import ModelEMA
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
    ):
        self.epochs = epochs
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

    def train(self):
        """Run training loop."""
        for epoch in range(self.start_epoch, self.epochs):
            self.model.train()

            if isinstance(self.train_loader.sampler, DistributedSampler):
                self.train_loader.sampler.set_epoch(epoch)

            epoch_loss = 0.0
            num_batches = 0

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

            if self.is_main:
                avg_loss = epoch_loss / max(num_batches, 1)
                console.print(f"  Epoch {epoch + 1} avg loss: {avg_loss:.4f}")

            # Validation
            if self.val_loader is not None and self.is_main and (epoch + 1) % 10 == 0:
                self._validate(epoch)

            # Save checkpoint
            if self.is_main:
                self._save_checkpoint(epoch)

    @torch.no_grad()
    def _validate(self, epoch: int):
        eval_model = self.ema.ema if self.ema is not None else self.model
        eval_model.eval()

        num_batches = 0

        for images, _targets in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            eval_model(images)
            num_batches += 1

        console.print(f"  Validation done ({num_batches} batches)")

    def _save_checkpoint(self, epoch: int):
        raw_model = self.model.module if isinstance(self.model, DDP) else self.model
        state = {
            "epoch": epoch + 1,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_map": self.best_map,
        }
        if self.ema is not None:
            state["ema"] = self.ema.state_dict()
        if self.scaler is not None:
            state["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(state, self.output_dir / "last.pt")
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

        if self.ema is not None and "ema" in ckpt:
            self.ema.load_state_dict(ckpt["ema"])
        if self.scaler is not None and "scaler_state_dict" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])

        console.print(f"Resumed from epoch {self.start_epoch}")
