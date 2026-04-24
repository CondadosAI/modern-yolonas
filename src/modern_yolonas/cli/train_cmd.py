"""CLI: yolonas train"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated

import typer


class ModelName(str, Enum):
    yolo_nas_s = "yolo_nas_s"
    yolo_nas_m = "yolo_nas_m"
    yolo_nas_l = "yolo_nas_l"


class DataFormat(str, Enum):
    yolo = "yolo"
    coco = "coco"


def train(
    data: Annotated[str, typer.Option(help="Path to dataset root.")],
    model: Annotated[ModelName, typer.Option(help="Model variant.")] = ModelName.yolo_nas_s,
    data_format: Annotated[DataFormat, typer.Option("--format", help="Dataset format.")] = DataFormat.yolo,
    epochs: Annotated[int, typer.Option(help="Number of training epochs.")] = 300,
    batch_size: Annotated[int, typer.Option(help="Batch size per GPU.")] = 32,
    lr: Annotated[float, typer.Option(help="Learning rate.")] = 2e-4,
    device: Annotated[str, typer.Option(help="Device.")] = "cuda",
    output: Annotated[str, typer.Option(help="Output directory.")] = "runs/train",
    resume_path: Annotated[str | None, typer.Option("--resume", help="Checkpoint path to resume from.")] = None,
    input_size: Annotated[int, typer.Option(help="Model input size.")] = 640,
    workers: Annotated[int, typer.Option(help="DataLoader workers.")] = 8,
    pretrained: Annotated[bool, typer.Option("--pretrained/--no-pretrained", help="Use pretrained COCO weights.")] = True,
    num_classes: Annotated[int, typer.Option(help="Number of classes (auto-detected from dataset when 0).")] = 0,
    # Logging
    wandb: Annotated[bool, typer.Option("--wandb/--no-wandb", help="Enable Weights & Biases logging.")] = False,
    wandb_project: Annotated[str, typer.Option(help="W&B project name.")] = "yolo-nas",
    wandb_name: Annotated[str | None, typer.Option(help="W&B run name (auto-generated when omitted).")] = None,
    tensorboard: Annotated[bool, typer.Option("--tensorboard/--no-tensorboard", help="Enable TensorBoard logging.")] = False,
    tensorboard_dir: Annotated[str, typer.Option(help="TensorBoard root log directory.")] = "runs/tensorboard",
    tensorboard_name: Annotated[str, typer.Option(help="TensorBoard experiment name (sub-directory under tensorboard-dir).")] = "default",
    val_freq: Annotated[int, typer.Option(help="Run validation every N epochs.")] = 10,
    # COCO-specific path overrides
    train_images: Annotated[str | None, typer.Option(help="[COCO] Training images directory. Defaults to <data>/images/train.")] = None,
    val_images: Annotated[str | None, typer.Option(help="[COCO] Validation images directory. Defaults to <data>/images/val.")] = None,
    train_ann: Annotated[str | None, typer.Option(help="[COCO] Training annotation JSON. Defaults to <data>/annotations/train.json.")] = None,
    val_ann: Annotated[str | None, typer.Option(help="[COCO] Validation annotation JSON. Defaults to <data>/annotations/val.json.")] = None,
):
    """Train a YOLO-NAS model."""
    from rich.console import Console

    from modern_yolonas import yolo_nas_s, yolo_nas_m, yolo_nas_l
    from modern_yolonas.data.transforms import Compose, HSVAugment, HorizontalFlip, RandomAffine, LetterboxResize, Normalize
    from modern_yolonas.data.collate import detection_collate_fn
    from modern_yolonas.training.trainer import Trainer
    from torch.utils.data import DataLoader

    console = Console()
    data_path = Path(data)

    train_transforms = Compose([
        HSVAugment(),
        HorizontalFlip(),
        RandomAffine(degrees=0.0, translate=0.1, scale=(0.5, 1.5)),
        LetterboxResize(target_size=input_size),
        Normalize(),
    ])
    val_transforms = Compose([LetterboxResize(target_size=input_size), Normalize()])

    if data_format == DataFormat.yolo:
        from modern_yolonas.data.yolo import YOLODetectionDataset

        train_dataset = YOLODetectionDataset(data, split="train", transforms=train_transforms, input_size=input_size)
        val_dataset = YOLODetectionDataset(data, split="val", transforms=val_transforms, input_size=input_size)
        if num_classes == 0:
            num_classes = train_dataset.num_classes

    else:  # COCO format
        from modern_yolonas.data.coco import COCODetectionDataset

        # Resolve paths with sensible defaults matching the most common layouts
        _train_images = Path(train_images) if train_images else data_path / "train"
        _val_images   = Path(val_images)   if val_images   else data_path / "val"
        _train_ann    = Path(train_ann)    if train_ann    else data_path / "annotations" / "train.json"
        _val_ann      = Path(val_ann)      if val_ann      else data_path / "annotations" / "val.json"

        # Fallback to the standard COCO 2017 layout if the defaults don't exist
        if not _train_images.exists() and (data_path / "images" / "train2017").exists():
            _train_images = data_path / "images" / "train2017"
        if not _val_images.exists() and (data_path / "images" / "val2017").exists():
            _val_images = data_path / "images" / "val2017"
        if not _train_ann.exists() and (data_path / "annotations" / "instances_train2017.json").exists():
            _train_ann = data_path / "annotations" / "instances_train2017.json"
        if not _val_ann.exists() and (data_path / "annotations" / "instances_val2017.json").exists():
            _val_ann = data_path / "annotations" / "instances_val2017.json"

        train_dataset = COCODetectionDataset(_train_images, _train_ann, transforms=train_transforms, input_size=input_size)
        val_dataset   = COCODetectionDataset(_val_images,   _val_ann,   transforms=val_transforms,   input_size=input_size)

        if num_classes == 0:
            num_classes = len(train_dataset.cat_id_to_label)

    # Resolve class names from dataset (used for annotation in val image logging)
    class_names: list[str] | None = getattr(train_dataset, "class_names", None)
    if class_names:
        console.print(f"Class names: {class_names}")

    console.print(f"Classes: {num_classes} | Train: {len(train_dataset)} images | Val: {len(val_dataset)} images")

    # Build model -------------------------------------------------------
    builders = {"yolo_nas_s": yolo_nas_s, "yolo_nas_m": yolo_nas_m, "yolo_nas_l": yolo_nas_l}
    if pretrained and num_classes != 80:
        # Transfer learning: load pretrained backbone+neck with strict=True,
        # then swap heads for a freshly initialised num_classes version.
        from modern_yolonas.weights import transfer_to

        console.print(
            f"[yellow]Transfer learning: loading pretrained {model.value} backbone+neck, "
            f"re-initialising heads for {num_classes} classes.[/yellow]"
        )
        yolo_model = transfer_to(model.value, num_classes=num_classes)
    else:
        console.print(f"Building {model.value} (pretrained={pretrained}, num_classes={num_classes})...")
        yolo_model = builders[model.value](pretrained=pretrained, num_classes=num_classes)

    # DataLoaders --------------------------------------------------------
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers,
        collate_fn=detection_collate_fn, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
        collate_fn=detection_collate_fn, pin_memory=True,
    )

    # Callbacks ----------------------------------------------------------
    callbacks = []
    if wandb:
        from modern_yolonas.training.callbacks import WandbCallback
        callbacks.append(WandbCallback(project=wandb_project, name=wandb_name))
        console.print(f"[green]W&B logging enabled → project={wandb_project!r}[/green]")
    if tensorboard:
        from modern_yolonas.training.callbacks import TensorBoardCallback
        callbacks.append(TensorBoardCallback(log_dir=tensorboard_dir, experiment_name=tensorboard_name))
        console.print(f"[green]TensorBoard logging enabled → {tensorboard_dir}/{tensorboard_name}[/green]")

    trainer = Trainer(
        model=yolo_model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        epochs=epochs,
        lr=lr,
        output_dir=output,
        device=device,
        callbacks=callbacks,
        class_names=class_names,
        val_freq=val_freq,
    )

    if resume_path:
        trainer.resume(resume_path)

    trainer.train()

