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
):
    """Train a YOLO-NAS model."""
    from rich.console import Console

    from modern_yolonas import yolo_nas_s, yolo_nas_m, yolo_nas_l
    from modern_yolonas.data.transforms import Compose, HSVAugment, HorizontalFlip, RandomAffine, LetterboxResize, Normalize
    from modern_yolonas.data.collate import detection_collate_fn
    from modern_yolonas.training.trainer import Trainer

    console = Console()

    builders = {"yolo_nas_s": yolo_nas_s, "yolo_nas_m": yolo_nas_m, "yolo_nas_l": yolo_nas_l}
    console.print(f"Building {model.value} (pretrained={pretrained})...")
    yolo_model = builders[model.value](pretrained=pretrained)

    transforms = Compose([
        HSVAugment(),
        HorizontalFlip(),
        RandomAffine(degrees=0.0, translate=0.1, scale=(0.5, 1.5)),
        LetterboxResize(target_size=input_size),
        Normalize(),
    ])

    if data_format == DataFormat.yolo:
        from modern_yolonas.data.yolo import YOLODetectionDataset
        from torch.utils.data import DataLoader

        train_dataset = YOLODetectionDataset(data, split="train", transforms=transforms, input_size=input_size)
        val_dataset = YOLODetectionDataset(data, split="val", transforms=Compose([
            LetterboxResize(target_size=input_size), Normalize()
        ]), input_size=input_size)
    else:
        from modern_yolonas.data.coco import COCODetectionDataset
        from torch.utils.data import DataLoader

        data_path = Path(data)
        train_dataset = COCODetectionDataset(
            data_path / "images" / "train2017",
            data_path / "annotations" / "instances_train2017.json",
            transforms=transforms,
            input_size=input_size,
        )
        val_dataset = COCODetectionDataset(
            data_path / "images" / "val2017",
            data_path / "annotations" / "instances_val2017.json",
            transforms=Compose([LetterboxResize(target_size=input_size), Normalize()]),
            input_size=input_size,
        )

    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers,
        collate_fn=detection_collate_fn, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
        collate_fn=detection_collate_fn, pin_memory=True,
    )

    console.print(f"Train: {len(train_dataset)} images, Val: {len(val_dataset)} images")

    trainer = Trainer(
        model=yolo_model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        output_dir=output,
        device=device,
    )

    if resume_path:
        trainer.resume(resume_path)

    trainer.train()
