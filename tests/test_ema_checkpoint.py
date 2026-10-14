"""A checkpoint must resume from the raw weights, and still serve the EMA ones.

``EMACallback`` used to overwrite a checkpoint's ``state_dict`` with the EMA
weights. ``--resume`` loads ``state_dict`` back into the model, so every resumed
run silently continued from the average rather than from where optimisation
stopped. Nothing raised: the loss curve just jumped, by an amount no one would
notice without a run that was never paused to compare against.
"""

from __future__ import annotations

import lightning as L
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from modern_yolonas.training.callbacks import EMACallback
from modern_yolonas.weights import extract_model_state_dict


class Toy(L.LightningModule):
    """The only thing EMACallback needs from a module is a ``.model``."""

    def __init__(self):
        super().__init__()
        self.model = nn.Linear(4, 1)

    def training_step(self, batch, _):
        x, y = batch
        return nn.functional.mse_loss(self.model(x), y)

    def configure_optimizers(self):
        # A large step, so the raw weights run well away from their average.
        return torch.optim.SGD(self.parameters(), lr=0.5)


class Probe(L.Callback):
    """Record the weights a resumed run actually starts from."""

    def __init__(self, ema: EMACallback):
        self.ema = ema
        self.model = self.ema_weights = None

    def on_train_start(self, trainer, pl_module):
        self.model = {k: v.clone() for k, v in pl_module.model.state_dict().items()}
        self.ema_weights = {k: v.clone() for k, v in self.ema.ema_model.state_dict().items()}


def loader():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(64, 4, generator=g)
    return DataLoader(TensorDataset(x, x.sum(1, keepdim=True)), batch_size=8)


def trainer(max_steps, callbacks, tmp_path):
    return L.Trainer(
        max_steps=max_steps, callbacks=callbacks, default_root_dir=tmp_path,
        accelerator="cpu", logger=False, enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False,
    )


@pytest.fixture
def saved(tmp_path):
    """Train a few steps, save, and return the checkpoint with the weights at save time."""
    torch.manual_seed(0)
    module, ema = Toy(), EMACallback(decay=0.9, warmup_steps=1)
    first = trainer(6, [ema], tmp_path)
    first.fit(module, loader())
    raw = {k: v.clone() for k, v in module.model.state_dict().items()}
    averaged = {k: v.clone() for k, v in ema.ema_model.state_dict().items()}
    assert not torch.allclose(raw["weight"], averaged["weight"]), "test needs EMA != raw"
    path = tmp_path / "step6.ckpt"
    first.save_checkpoint(path)
    return path, raw, averaged


def test_resume_continues_from_the_raw_weights(saved, tmp_path):
    path, raw, averaged = saved
    ema = EMACallback(decay=0.9, warmup_steps=1)
    probe = Probe(ema)
    trainer(8, [ema, probe], tmp_path).fit(Toy(), loader(), ckpt_path=path)
    for k in raw:
        torch.testing.assert_close(probe.model[k], raw[k])
        torch.testing.assert_close(probe.ema_weights[k], averaged[k])


def test_the_checkpoint_still_serves_ema_weights(saved):
    """Evaluation, export and inference all read through this, and must get EMA."""
    path, _, averaged = saved
    extracted = extract_model_state_dict(path)
    for k in averaged:
        torch.testing.assert_close(extracted[k], averaged[k])


def test_ema_update_count_survives_the_resume(saved, tmp_path):
    path, _, _ = saved
    ema = EMACallback(decay=0.9, warmup_steps=1)
    trainer(8, [ema], tmp_path).fit(Toy(), loader(), ckpt_path=path)
    assert ema.updates == 8


def test_a_save_during_validation_still_stores_raw_weights():
    """Mid-validation the model holds the EMA weights; the save must look past them."""
    module, ema = Toy(), EMACallback()
    ema.on_fit_start(None, module)
    with torch.no_grad():
        for p in ema.ema_model.parameters():
            p.add_(1.0)
    raw = {k: v.clone() for k, v in module.model.state_dict().items()}

    ema._swap_ema_in(module)
    checkpoint = {"state_dict": {f"model.{k}": v for k, v in module.model.state_dict().items()}}
    ema.on_save_checkpoint(None, module, checkpoint)
    ema._swap_ema_out(module)

    for k, v in raw.items():
        torch.testing.assert_close(checkpoint["state_dict"][f"model.{k}"], v)
        torch.testing.assert_close(checkpoint["ema_state_dict"][k], v + 1.0)
