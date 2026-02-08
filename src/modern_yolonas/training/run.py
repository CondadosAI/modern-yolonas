"""Shared training runner used by benchmark commands."""

from __future__ import annotations

from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

from modern_yolonas import yolo_nas_s, yolo_nas_m, yolo_nas_l
from modern_yolonas.data.transforms import (
    Compose,
    HSVAugment,
    HorizontalFlip,
    LetterboxResize,
    Normalize,
    RandomAffine,
)
from modern_yolonas.training.callbacks import EMACallback
from modern_yolonas.training.data_module import DetectionDataModule
from modern_yolonas.training.lightning_module import YoloNASLightningModule

MODEL_BUILDERS = {
    "yolo_nas_s": yolo_nas_s,
    "yolo_nas_m": yolo_nas_m,
    "yolo_nas_l": yolo_nas_l,
}


def build_transforms(recipe: dict, *, train: bool) -> Compose:
    """Build transform pipeline from a recipe dict."""
    input_size = recipe["input_size"]
    if train:
        aug = recipe["augmentations"]
        steps = []
        if aug.get("hsv"):
            steps.append(HSVAugment())
        if aug.get("flip"):
            steps.append(HorizontalFlip())
        steps.append(
            RandomAffine(
                degrees=aug.get("affine_degrees", 0.0),
                translate=aug.get("affine_translate", 0.1),
                scale=aug.get("affine_scale", (0.5, 1.5)),
            )
        )
        steps.append(LetterboxResize(target_size=input_size))
        steps.append(Normalize())
        return Compose(steps)
    return Compose([LetterboxResize(target_size=input_size), Normalize()])


def run_training(
    model_name: str,
    recipe: dict,
    train_dataset,
    val_dataset,
    val_ann_file: str | Path | None = None,
    output_dir: str | Path = "runs/train",
    num_classes: int = 80,
    pretrained: bool = True,
    devices: str | int | list[int] = "auto",
    logger: str = "csv",
    resume_path: str | Path | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
) -> Path:
    """Run a full training session, returning path to the best checkpoint.

    Args:
        model_name: One of 'yolo_nas_s', 'yolo_nas_m', 'yolo_nas_l'.
        recipe: Recipe dict (from recipes.py).
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        val_ann_file: COCO annotation file for mAP evaluation.
        output_dir: Directory for checkpoints and logs.
        num_classes: Number of object classes.
        pretrained: Whether to start from pretrained COCO weights.
        devices: Lightning devices spec.
        logger: Logger backend ('csv', 'tensorboard', 'wandb').
        resume_path: Optional checkpoint to resume from.
        epochs: Override recipe epochs.
        batch_size: Override recipe batch_size.

    Returns:
        Path to the best checkpoint file.
    """
    output_dir = Path(output_dir)
    epochs = epochs or recipe["epochs"]
    batch_size = batch_size or recipe["batch_size"]

    # Build model
    builder = MODEL_BUILDERS[model_name]
    model = builder(pretrained=pretrained, num_classes=num_classes)

    # Warmup steps
    warmup_epochs = recipe.get("warmup_epochs", 3)
    steps_per_epoch = max(1, len(train_dataset) // batch_size)
    warmup_steps = warmup_epochs * steps_per_epoch

    # Lightning module
    lit_model = YoloNASLightningModule(
        model=model,
        num_classes=num_classes,
        lr=recipe["lr"],
        optimizer_name=recipe["optimizer"],
        weight_decay=recipe["weight_decay"],
        warmup_steps=warmup_steps,
        cosine_final_lr_ratio=recipe["cosine_final_lr_ratio"],
        val_ann_file=val_ann_file,
        conf_threshold=recipe.get("conf_threshold", 0.001),
        iou_threshold=recipe.get("iou_threshold", 0.65),
    )

    # Data module
    data_module = DetectionDataModule(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=batch_size,
        num_workers=recipe.get("workers", 8),
    )

    # Logger
    if logger == "csv":
        logger_instance = L.pytorch.loggers.CSVLogger(str(output_dir))
    elif logger == "tensorboard":
        logger_instance = L.pytorch.loggers.TensorBoardLogger(str(output_dir))
    else:
        logger_instance = L.pytorch.loggers.WandbLogger(project="yolonas", save_dir=str(output_dir))

    # Callbacks
    monitor = "val/mAP" if val_ann_file else "val/loss"
    mode = "max" if val_ann_file else "min"
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        monitor=monitor,
        mode=mode,
        save_last=True,
        filename="best-{epoch:02d}-{" + monitor.replace("/", "_") + ":.4f}",
    )
    callbacks = [
        EMACallback(decay=recipe.get("ema_decay", 0.9997)),
        checkpoint_cb,
    ]

    # Trainer
    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=devices,
        strategy="auto",
        precision=recipe.get("precision", "16-mixed"),
        callbacks=callbacks,
        logger=logger_instance,
        default_root_dir=str(output_dir),
    )

    if resume_path:
        trainer.fit(lit_model, datamodule=data_module, ckpt_path=str(resume_path))
    else:
        trainer.fit(lit_model, datamodule=data_module)

    best_path = checkpoint_cb.best_model_path
    return Path(best_path) if best_path else output_dir / "checkpoints" / "last.ckpt"
