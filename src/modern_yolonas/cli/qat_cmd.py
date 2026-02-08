"""CLI: yolonas qat — Quantization-Aware Training."""

from __future__ import annotations

import click


@click.command()
@click.option("--model", default="yolo_nas_s", type=click.Choice(["yolo_nas_s", "yolo_nas_m", "yolo_nas_l"]))
@click.option("--data", required=True, help="Path to dataset root.")
@click.option("--format", "data_format", default="yolo", type=click.Choice(["yolo", "coco"]))
@click.option("--epochs", default=10, help="QAT fine-tuning epochs.")
@click.option("--batch-size", default=32, help="Batch size.")
@click.option("--lr", default=2e-5, help="Learning rate (lower than full training).")
@click.option("--backend", default="x86", type=click.Choice(["x86", "qnnpack", "onednn"]))
@click.option("--output", default="runs/qat", help="Output directory.")
@click.option("--checkpoint", default=None, help="Checkpoint to start from.")
@click.option("--input-size", default=640, help="Model input size.")
@click.option("--workers", default=8, help="DataLoader workers.")
@click.option("--pretrained/--no-pretrained", default=True, help="Use pretrained COCO weights.")
@click.option("--devices", default="auto", help="Devices to use (e.g. 'auto', '1', '0,1').")
@click.option(
    "--logger",
    default="csv",
    type=click.Choice(["csv", "tensorboard", "wandb"]),
    help="Logger backend.",
)
def qat(
    model: str,
    data: str,
    data_format: str,
    epochs: int,
    batch_size: int,
    lr: float,
    backend: str,
    output: str,
    checkpoint: str | None,
    input_size: int,
    workers: int,
    pretrained: bool,
    devices: str,
    logger: str,
):
    """Run Quantization-Aware Training on a YOLO-NAS model."""
    from pathlib import Path

    import lightning as L
    from lightning.pytorch.callbacks import ModelCheckpoint
    import torch
    from rich.console import Console

    from modern_yolonas import yolo_nas_s, yolo_nas_m, yolo_nas_l
    from modern_yolonas.quantization import prepare_model_qat, convert_quantized, export_quantized_onnx
    from modern_yolonas.training import YoloNASLightningModule, QATCallback, DetectionDataModule
    from modern_yolonas.training.lightning_module import extract_model_state_dict
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
    console.print(f"Building {model}...")

    if checkpoint:
        yolo_model = builders[model](pretrained=False)
        sd = extract_model_state_dict(checkpoint)
        yolo_model.load_state_dict(sd)
    else:
        yolo_model = builders[model](pretrained=pretrained)

    # Prepare for QAT
    console.print(f"Preparing model for QAT (backend={backend})...")
    example_input = torch.randn(1, 3, input_size, input_size)
    qat_model = prepare_model_qat(yolo_model, backend=backend, example_input=example_input)

    # Build datasets
    train_transforms = build_transforms(recipe, train=True)
    val_transforms = build_transforms(recipe, train=False)

    if data_format == "yolo":
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
    if logger == "csv":
        logger_instance = L.pytorch.loggers.CSVLogger(output)
    elif logger == "tensorboard":
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
