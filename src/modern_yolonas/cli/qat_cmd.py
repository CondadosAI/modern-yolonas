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
@click.option("--device", default="cuda", help="Device.")
@click.option("--output", default="runs/qat", help="Output directory.")
@click.option("--checkpoint", default=None, help="Checkpoint to start from.")
@click.option("--input-size", default=640, help="Model input size.")
@click.option("--workers", default=8, help="DataLoader workers.")
@click.option("--pretrained/--no-pretrained", default=True, help="Use pretrained COCO weights.")
def qat(
    model: str,
    data: str,
    data_format: str,
    epochs: int,
    batch_size: int,
    lr: float,
    backend: str,
    device: str,
    output: str,
    checkpoint: str | None,
    input_size: int,
    workers: int,
    pretrained: bool,
):
    """Run Quantization-Aware Training on a YOLO-NAS model."""
    from pathlib import Path

    import torch
    from rich.console import Console

    from modern_yolonas import yolo_nas_s, yolo_nas_m, yolo_nas_l
    from modern_yolonas.data.transforms import (
        Compose,
        HSVAugment,
        HorizontalFlip,
        RandomAffine,
        LetterboxResize,
        Normalize,
    )
    from modern_yolonas.data.collate import detection_collate_fn
    from modern_yolonas.quantization import prepare_model_qat, convert_quantized, export_quantized_onnx
    from modern_yolonas.quantization.qat_trainer import QATTrainer

    console = Console()

    # Build model
    builders = {"yolo_nas_s": yolo_nas_s, "yolo_nas_m": yolo_nas_m, "yolo_nas_l": yolo_nas_l}
    console.print(f"Building {model}...")

    if checkpoint:
        yolo_model = builders[model](pretrained=False)
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
        sd = ckpt.get("model_state_dict", ckpt)
        yolo_model.load_state_dict(sd)
    else:
        yolo_model = builders[model](pretrained=pretrained)

    # Prepare for QAT
    console.print(f"Preparing model for QAT (backend={backend})...")
    example_input = torch.randn(1, 3, input_size, input_size)
    qat_model = prepare_model_qat(yolo_model, backend=backend, example_input=example_input)

    # Build datasets
    train_transforms = Compose([
        HSVAugment(),
        HorizontalFlip(),
        RandomAffine(degrees=0.0, translate=0.1, scale=(0.5, 1.5)),
        LetterboxResize(target_size=input_size),
        Normalize(),
    ])
    val_transforms = Compose([LetterboxResize(target_size=input_size), Normalize()])

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

    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        collate_fn=detection_collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=detection_collate_fn,
        pin_memory=True,
    )

    console.print(f"Train: {len(train_dataset)} images, Val: {len(val_dataset)} images")

    # QAT Training
    trainer = QATTrainer(
        model=qat_model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        output_dir=output,
        device=device,
    )
    trainer.train()

    # Convert and export
    console.print("Converting QAT model to quantized form...")
    eval_model = trainer.ema.ema if trainer.ema is not None else trainer.model
    quantized_model = convert_quantized(eval_model)

    onnx_path = str(Path(output) / "model_qat.onnx")
    console.print(f"Exporting to {onnx_path}...")
    export_quantized_onnx(quantized_model, onnx_path, input_size=input_size)
    console.print(f"[green]QAT complete. Model saved to {output}[/green]")
