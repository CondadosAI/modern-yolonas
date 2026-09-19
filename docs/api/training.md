# Training API

Training runs on [PyTorch Lightning](https://lightning.ai/). `run_training` is the entry
point the `yolonas train` CLI calls; the pieces below are what it wires together, and each
can be used on its own if you would rather drive Lightning yourself.

## Entry point

::: modern_yolonas.training.run.run_training

## Lightning module and data

::: modern_yolonas.training.lightning_module.YoloNASLightningModule

::: modern_yolonas.training.data_module.DetectionDataModule

## Loss and metrics

::: modern_yolonas.training.loss.PPYoloELoss

::: modern_yolonas.training.metrics.DetectionMetrics

## Callbacks

::: modern_yolonas.training.callbacks.EMACallback

::: modern_yolonas.training.callbacks.CloseMosaicCallback

::: modern_yolonas.training.callbacks.QATCallback

## Checkpoints

::: modern_yolonas.weights.extract_model_state_dict
