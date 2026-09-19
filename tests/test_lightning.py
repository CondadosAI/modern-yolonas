"""Tests for Lightning module, callbacks, data module, and checkpoint extraction."""

from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from modern_yolonas import yolo_nas_s
from modern_yolonas.training.lightning_module import YoloNASLightningModule, extract_model_state_dict
from modern_yolonas.training.callbacks import EMACallback, QATCallback
from modern_yolonas.training.data_module import DetectionDataModule


@pytest.fixture
def small_model():
    return yolo_nas_s(pretrained=False)


@pytest.fixture
def lit_module(small_model):
    return YoloNASLightningModule(model=small_model, num_classes=80, warmup_steps=10)


@pytest.fixture
def synthetic_batch():
    images = torch.randn(2, 3, 640, 640)
    targets = torch.tensor([
        [0, 5, 0.5, 0.5, 0.1, 0.1],
        [0, 10, 0.3, 0.7, 0.2, 0.15],
        [1, 0, 0.6, 0.4, 0.3, 0.3],
    ])
    return images, targets


class TestLightningModule:
    def test_training_step(self, lit_module, synthetic_batch):
        lit_module.model.train()
        loss = lit_module.training_step(synthetic_batch, 0)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.requires_grad

    def test_validation_step(self, lit_module, synthetic_batch):
        # Without val_ann_file, validation uses train-mode fallback for loss
        lit_module.model.train()
        lit_module.validation_step(synthetic_batch, 0)

    def test_configure_optimizers(self, lit_module):
        # Mock trainer with estimated_stepping_batches
        mock_trainer = MagicMock()
        mock_trainer.estimated_stepping_batches = 10000
        lit_module.trainer = mock_trainer

        result = lit_module.configure_optimizers()
        assert "optimizer" in result
        assert "lr_scheduler" in result
        assert result["lr_scheduler"]["interval"] == "step"

        optimizer = result["optimizer"]
        assert optimizer.__class__.__name__ == "AdamW"

    def test_forward_delegates_to_model(self, lit_module, small_model):
        lit_module.model.eval()
        x = torch.randn(1, 3, 640, 640)
        bboxes, scores = lit_module(x)
        assert bboxes.shape == (1, 8400, 4)
        assert scores.shape == (1, 8400, 80)


class TestEMACallback:
    def test_creates_shadow_copy(self, lit_module):
        callback = EMACallback()
        callback.on_fit_start(trainer=MagicMock(), pl_module=lit_module)
        assert callback.ema_model is not None
        # EMA model should have same parameter count
        orig_params = sum(p.numel() for p in lit_module.model.parameters())
        ema_params = sum(p.numel() for p in callback.ema_model.parameters())
        assert orig_params == ema_params

    def test_updates_weights_on_batch_end(self, lit_module):
        callback = EMACallback(decay=0.999, warmup_steps=100)
        callback.on_fit_start(trainer=MagicMock(), pl_module=lit_module)

        # Capture initial EMA weights
        initial_ema = {k: v.clone() for k, v in callback.ema_model.state_dict().items()}

        # Perturb model weights
        with torch.no_grad():
            for p in lit_module.model.parameters():
                p.add_(torch.randn_like(p) * 0.01)

        callback.on_train_batch_end(
            trainer=MagicMock(), pl_module=lit_module,
            outputs=None, batch=None, batch_idx=0,
        )

        # EMA weights should have changed
        changed = False
        for k, v in callback.ema_model.state_dict().items():
            if v.dtype.is_floating_point and not torch.equal(v, initial_ema[k]):
                changed = True
                break
        assert changed

    def test_swap_ema_during_validation(self, lit_module):
        callback = EMACallback()
        callback.on_fit_start(trainer=MagicMock(), pl_module=lit_module)

        # Make EMA weights different
        with torch.no_grad():
            for p in callback.ema_model.parameters():
                p.mul_(0.5)

        orig_weight = next(iter(lit_module.model.parameters())).clone()

        # Swap in
        callback.on_validation_start(trainer=MagicMock(), pl_module=lit_module)
        val_weight = next(iter(lit_module.model.parameters())).clone()
        assert not torch.equal(orig_weight, val_weight)

        # Swap out
        callback.on_validation_end(trainer=MagicMock(), pl_module=lit_module)
        restored_weight = next(iter(lit_module.model.parameters())).clone()
        assert torch.equal(orig_weight, restored_weight)


class TestQATCallback:
    def test_freezes_bn_after_threshold(self):
        callback = QATCallback(freeze_bn_after_epoch=2, freeze_observer_after_epoch=5)

        # Create a simple mock with backbone_neck containing BN layers
        model = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16))
        pl_module = MagicMock()
        pl_module.model.backbone_neck = model
        trainer = MagicMock()

        # Before threshold: BN should remain in training mode
        trainer.current_epoch = 1
        model.train()
        callback.on_train_epoch_start(trainer, pl_module)
        assert model[1].training  # BN still in train mode

        # After threshold: BN should be eval
        trainer.current_epoch = 2
        model.train()
        callback.on_train_epoch_start(trainer, pl_module)
        assert not model[1].training  # BN in eval mode


class TestDetectionDataModule:
    def test_creates_dataloaders(self):
        # Create dummy datasets
        train_data = [(torch.randn(3, 640, 640).numpy(), torch.tensor([[0, 0.5, 0.5, 0.1, 0.1]]).numpy())]
        val_data = [(torch.randn(3, 640, 640).numpy(), torch.tensor([[0, 0.5, 0.5, 0.1, 0.1]]).numpy())]

        # Use simple list-based dataset
        dm = DetectionDataModule(
            train_dataset=train_data,
            val_dataset=val_data,
            batch_size=1,
            num_workers=0,
        )

        train_dl = dm.train_dataloader()
        assert train_dl is not None

        val_dl = dm.val_dataloader()
        assert val_dl is not None

    def test_no_val_dataset(self):
        train_data = [(torch.randn(3, 640, 640).numpy(), torch.tensor([[0, 0.5, 0.5, 0.1, 0.1]]).numpy())]
        dm = DetectionDataModule(train_dataset=train_data, batch_size=1, num_workers=0)
        assert dm.val_dataloader() is None


class TestExtractModelStateDict:
    def test_lightning_checkpoint(self, small_model, tmp_path):
        # Simulate Lightning checkpoint format
        sd = {"model." + k: v for k, v in small_model.state_dict().items()}
        ckpt = {"state_dict": sd, "epoch": 5}
        path = tmp_path / "lightning.ckpt"
        torch.save(ckpt, path)

        extracted = extract_model_state_dict(path)
        # Should strip 'model.' prefix
        for key in small_model.state_dict():
            assert key in extracted

    def test_legacy_checkpoint(self, small_model, tmp_path):
        ckpt = {"model_state_dict": small_model.state_dict(), "epoch": 5}
        path = tmp_path / "legacy.pt"
        torch.save(ckpt, path)

        extracted = extract_model_state_dict(path)
        for key in small_model.state_dict():
            assert key in extracted

    def test_legacy_with_ema(self, small_model, tmp_path):
        ema_sd = small_model.state_dict()
        ckpt = {
            "model_state_dict": small_model.state_dict(),
            "ema": {"ema_state_dict": ema_sd, "decay": 0.9997, "updates": 100},
        }
        path = tmp_path / "ema.pt"
        torch.save(ckpt, path)

        extracted = extract_model_state_dict(path)
        for key in small_model.state_dict():
            assert key in extracted

    def test_plain_state_dict(self, small_model, tmp_path):
        path = tmp_path / "plain.pt"
        torch.save(small_model.state_dict(), path)

        extracted = extract_model_state_dict(path)
        for key in small_model.state_dict():
            assert key in extracted
