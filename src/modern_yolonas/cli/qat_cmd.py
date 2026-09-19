"""CLI: yolonas qat — Quantization-Aware Training."""

from __future__ import annotations

from enum import Enum
from typing import Annotated

import typer

from modern_yolonas.cli.quantize_cmd import Backend
from modern_yolonas.cli.train_cmd import DataFormat, ModelName


class LoggerBackend(str, Enum):
    csv = "csv"
    tensorboard = "tensorboard"
    wandb = "wandb"


def qat(
    data: Annotated[str, typer.Option(help="Path to dataset root.")],
    model: Annotated[ModelName, typer.Option(help="Model variant.")] = ModelName.yolo_nas_s,
    data_format: Annotated[DataFormat, typer.Option("--format", help="Dataset format.")] = DataFormat.yolo,
    epochs: Annotated[int, typer.Option(help="QAT fine-tuning epochs.")] = 10,
    batch_size: Annotated[int, typer.Option(help="Batch size.")] = 32,
    lr: Annotated[float, typer.Option(help="Learning rate (lower than full training).")] = 2e-5,
    backend: Annotated[Backend, typer.Option(help="Quantization backend.")] = Backend.x86,
    output: Annotated[str, typer.Option(help="Output directory.")] = "runs/qat",
    checkpoint: Annotated[str | None, typer.Option(help="Checkpoint to start from.")] = None,
    input_size: Annotated[int, typer.Option(help="Model input size.")] = 640,
    workers: Annotated[int, typer.Option(help="DataLoader workers.")] = 8,
    pretrained: Annotated[bool, typer.Option("--pretrained/--no-pretrained", help="Use pretrained COCO weights.")] = True,
    devices: Annotated[str, typer.Option(help="Devices to use (e.g. 'auto', '1', '0,1').")] = "auto",
    logger: Annotated[LoggerBackend, typer.Option(help="Logger backend.")] = LoggerBackend.csv,
):
    """Run Quantization-Aware Training on a YOLO-NAS model."""
    from pathlib import Path

    import lightning as L
    from lightning.pytorch.callbacks import ModelCheckpoint
    import torch
    from rich.console import Console

    from modern_yolonas import yolo_nas_s, yolo_nas_m, yolo_nas_l
    from modern_yolonas.quantization import prepare_model_qat, convert_quantized, export_quantized_onnx
    from modern_yolonas.training import DetectionDataModule, QATCallback, YoloNASLightningModule
    from modern_yolonas.weights import extract_model_state_dict
    from modern_yolonas.training.recipes import COCO_RECIPE
    from modern_yolonas.training.run import build_transforms

    console = Console()

    # QAT recipe: no Mosaic/Mixup
    recipe = {
        **COCO_RECIPE,
        "input_size": input_size,
        "workers": workers,
        "augmentations": {
            **COCO_RECIPE["augmentations"],
            "mosaic": False,
            "mixup": False,
            "close_mosaic_epochs": 0,
        },
    }

    # Build model
    builders = {"yolo_nas_s": yolo_nas_s, "yolo_nas_m": yolo_nas_m, "yolo_nas_l": yolo_nas_l}
    console.print(f"Building {model.value}...")

    if checkpoint:
        yolo_model = builders[model.value](pretrained=False)
        yolo_model.load_state_dict(extract_model_state_dict(checkpoint))
    else:
        yolo_model = builders[model.value](pretrained=pretrained)

    # Prepare for QAT
    console.print(f"Preparing model for QAT (backend={backend.value})...")
    example_input = torch.randn(1, 3, input_size, input_size)
    qat_model = prepare_model_qat(yolo_model, backend=backend.value, example_input=example_input)

    # Build datasets
    train_transforms = build_transforms(recipe, train=True)
    val_transforms = build_transforms(recipe, train=False)

    if data_format == DataFormat.yolo:
        from modern_yolonas.data.yolo import YOLODetectionDataset

        train_dataset = YOLODetectionDataset(data, split="train", transforms=train_transforms, input_size=input_size)
        val_dataset = YOLODetectionDataset(data, split="val", transforms=val_transforms, input_size=input_size)
    else:
        data_path = Path(data)
        from modern_yolonas.data.coco import COCODetectionDataset

        train_dataset = COCODetectionDataset(
            data_path / "images" / "train2017",
            data_path / "annotations" / "instances_train2017.json",
            transforms=train_transforms,
            input_size=input_size,
        )
        val_dataset = COCODetectionDataset(
            data_path / "images" / "val2017",
            data_path / "annotations" / "instances_val2017.json",
            transforms=val_transforms,
            input_size=input_size,
        )

    console.print(f"Train: {len(train_dataset)} images, Val: {len(val_dataset)} images")

    # Lightning components
    warmup_steps = min(200, len(train_dataset) // batch_size * 1)
    lit_model = YoloNASLightningModule(
        model=qat_model,
        num_classes=80,
        lr=lr,
        warmup_steps=warmup_steps,
    )

    data_module = DetectionDataModule(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=batch_size,
        num_workers=workers,
    )

    # Logger
    if logger == LoggerBackend.csv:
        logger_instance = L.pytorch.loggers.CSVLogger(output)
    elif logger == LoggerBackend.tensorboard:
        logger_instance = L.pytorch.loggers.TensorBoardLogger(output)
    else:
        logger_instance = L.pytorch.loggers.WandbLogger(project="yolonas", save_dir=output)

    # Parse devices
    parsed_devices: str | int | list[int] = devices
    if devices != "auto":
        if "," in devices:
            parsed_devices = [int(d) for d in devices.split(",")]
        else:
            parsed_devices = int(devices)

    callbacks = [
        QATCallback(freeze_bn_after_epoch=3, freeze_observer_after_epoch=5),
        ModelCheckpoint(dirpath=output, save_last=True),
    ]

    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=parsed_devices,
        strategy="auto",
        precision="32-true",  # No AMP — fake-quant incompatible with autocast
        callbacks=callbacks,
        logger=logger_instance,
        default_root_dir=output,
    )

    trainer.fit(lit_model, datamodule=data_module)

    # Convert and export
    console.print("Converting QAT model to quantized form...")
    quantized_model = convert_quantized(qat_model)

    onnx_path = str(Path(output) / "model_qat.onnx")
    console.print(f"Exporting to {onnx_path}...")
    export_quantized_onnx(quantized_model, onnx_path, input_size=input_size)
    console.print(f"[green]QAT complete. Model saved to {output}[/green]")
