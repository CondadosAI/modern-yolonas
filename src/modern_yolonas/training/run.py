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
    Mixup,
    Mosaic,
    Normalize,
    RandomAffine,
    RandomChannelShuffle,
    RandomCrop,
    TrainTransformPipeline,
    VerticalFlip,
)
from modern_yolonas.training.callbacks import CloseMosaicCallback, EMACallback
from modern_yolonas.training.data_module import DetectionDataModule
from modern_yolonas.training.lightning_module import YoloNASLightningModule

MODEL_BUILDERS = {
    "yolo_nas_s": yolo_nas_s,
    "yolo_nas_m": yolo_nas_m,
    "yolo_nas_l": yolo_nas_l,
}


def build_transforms(recipe: dict, *, train: bool, dataset=None):
    """Build transform pipeline from a recipe dict.

    Args:
        recipe: Recipe dict containing 'input_size' and 'augmentations'.
        train: Whether to build training transforms.
        dataset: Training dataset instance. When provided and Mosaic/Mixup are
            enabled, returns a ``TrainTransformPipeline`` that supports
            dataset-aware augmentations.

    Returns:
        A ``Compose`` (val, or train without dataset-aware augs) or
        ``TrainTransformPipeline`` (train with Mosaic/Mixup).
    """
    input_size = recipe["input_size"]

    if not train:
        return Compose([LetterboxResize(target_size=input_size), Normalize()])

    aug = recipe.get("augmentations", {})

    # Build per-image transforms
    per_image_steps = []
    per_image_steps.append(
        RandomAffine(
            degrees=aug.get("affine_degrees", 0.0),
            translate=aug.get("affine_translate", 0.1),
            scale=aug.get("affine_scale", (0.5, 1.5)),
        )
    )
    if aug.get("channel_shuffle"):
        per_image_steps.append(RandomChannelShuffle(p=aug.get("channel_shuffle_prob", 0.5)))
    if aug.get("hsv"):
        per_image_steps.append(HSVAugment(p=aug.get("hsv_prob", 1.0)))
    if aug.get("flip"):
        per_image_steps.append(HorizontalFlip(p=aug.get("flip_prob", 0.5)))
    if aug.get("vertical_flip"):
        per_image_steps.append(VerticalFlip(p=aug.get("vertical_flip_prob", 0.5)))
    if aug.get("random_crop"):
        per_image_steps.append(RandomCrop(
            min_scale=aug.get("random_crop_min_scale", 0.3),
            max_scale=aug.get("random_crop_max_scale", 1.0),
            p=aug.get("random_crop_prob", 1.0),
        ))

    per_image_compose = Compose(per_image_steps)
    final_compose = Compose([LetterboxResize(target_size=input_size), Normalize()])

    use_mosaic = aug.get("mosaic", False) and dataset is not None
    use_mixup = aug.get("mixup", False) and dataset is not None

    if not use_mosaic and not use_mixup:
        # Simple pipeline — no dataset-aware augmentations
        return Compose(per_image_steps + [LetterboxResize(target_size=input_size), Normalize()])

    # Build TrainTransformPipeline with Mosaic and/or Mixup
    mosaic = None
    if use_mosaic:
        mosaic = Mosaic(
            dataset=dataset,
            input_size=input_size,
            prob=aug.get("mosaic_prob", 1.0),
        )

    mixup = None
    if use_mixup:
        mixup = Mixup(
            dataset=dataset,
            prob=aug.get("mixup_prob", 0.5),
        )
        # Mixup's second image gets the same per-image transforms
        mixup.inner_transforms = per_image_compose

    return TrainTransformPipeline(
        mosaic=mosaic,
        per_image_transforms=per_image_compose,
        mixup=mixup,
        final_transforms=final_compose,
    )


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

    # CloseMosaicCallback
    aug = recipe.get("augmentations", {})
    close_mosaic_epochs = aug.get("close_mosaic_epochs", 0)
    if close_mosaic_epochs > 0 and (aug.get("mosaic") or aug.get("mixup")):
        callbacks.append(CloseMosaicCallback(close_mosaic_epochs=close_mosaic_epochs))

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
