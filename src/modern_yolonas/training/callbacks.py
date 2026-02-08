"""Lightning callbacks for YOLO-NAS training."""

from __future__ import annotations

import math
from copy import deepcopy

import lightning as L
import torch
from torch import nn


class EMACallback(L.Callback):
    """Exponential Moving Average of model weights.

    Maintains a shadow copy of model parameters updated with an exponential
    moving average. During validation and checkpointing, EMA weights are
    swapped in.

    Args:
        decay: EMA decay factor.
        warmup_steps: Steps before reaching full decay.
    """

    def __init__(self, decay: float = 0.9997, warmup_steps: int = 2000):
        super().__init__()
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.updates = 0
        self.ema_model: nn.Module | None = None
        self._orig_state: dict | None = None

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        self.ema_model = deepcopy(pl_module.model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / self.warmup_steps))
        model_sd = pl_module.model.state_dict()
        for k, v in self.ema_model.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(model_sd[k], alpha=1.0 - d)

    def _swap_ema_in(self, pl_module: L.LightningModule):
        self._orig_state = {k: v.clone() for k, v in pl_module.model.state_dict().items()}
        pl_module.model.load_state_dict(self.ema_model.state_dict())

    def _swap_ema_out(self, pl_module: L.LightningModule):
        if self._orig_state is not None:
            pl_module.model.load_state_dict(self._orig_state)
            self._orig_state = None

    def on_validation_start(self, trainer, pl_module):
        if self.ema_model is not None:
            self._swap_ema_in(pl_module)

    def on_validation_end(self, trainer, pl_module):
        self._swap_ema_out(pl_module)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if self.ema_model is not None:
            checkpoint["ema_state_dict"] = self.ema_model.state_dict()
            checkpoint["ema_updates"] = self.updates
            # Save EMA weights as the model weights in the checkpoint
            ema_sd = self.ema_model.state_dict()
            prefix = "model."
            for k, v in ema_sd.items():
                checkpoint["state_dict"][prefix + k] = v

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        if "ema_state_dict" in checkpoint:
            self.updates = checkpoint.get("ema_updates", 0)
            # ema_model is created in on_fit_start; defer loading
            self._pending_ema_state = checkpoint["ema_state_dict"]

    def on_train_start(self, trainer, pl_module):
        if hasattr(self, "_pending_ema_state") and self._pending_ema_state is not None:
            if self.ema_model is not None:
                self.ema_model.load_state_dict(self._pending_ema_state)
            self._pending_ema_state = None


class QATCallback(L.Callback):
    """Freeze BatchNorm and observers during Quantization-Aware Training.

    Args:
        freeze_bn_after_epoch: Freeze BN running stats after this epoch.
        freeze_observer_after_epoch: Stop updating fake-quant ranges after this epoch.
    """

    def __init__(self, freeze_bn_after_epoch: int = 3, freeze_observer_after_epoch: int = 5):
        super().__init__()
        self.freeze_bn_after_epoch = freeze_bn_after_epoch
        self.freeze_observer_after_epoch = freeze_observer_after_epoch

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        epoch = trainer.current_epoch

        if epoch >= self.freeze_bn_after_epoch:
            for m in pl_module.model.backbone_neck.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

        if epoch >= self.freeze_observer_after_epoch:
            pl_module.model.backbone_neck.apply(torch.ao.quantization.disable_observer)
