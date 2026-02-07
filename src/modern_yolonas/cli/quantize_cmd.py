"""CLI: yolonas quantize — Post-Training Quantization."""

from __future__ import annotations

import click


@click.command()
@click.option("--model", default="yolo_nas_s", type=click.Choice(["yolo_nas_s", "yolo_nas_m", "yolo_nas_l"]))
@click.option("--data", required=True, help="Path to calibration dataset root.")
@click.option("--format", "data_format", default="yolo", type=click.Choice(["yolo", "coco"]))
@click.option("--num-batches", default=100, help="Number of calibration batches.")
@click.option("--batch-size", default=32, help="Batch size for calibration.")
@click.option("--backend", default="x86", type=click.Choice(["x86", "qnnpack", "onednn"]))
@click.option("--input-size", default=640, help="Model input size.")
@click.option("--output", default="model_ptq.onnx", help="Output file (.onnx or .pt).")
@click.option("--checkpoint", default=None, help="Custom checkpoint path.")
@click.option("--device", default="cpu", help="Calibration device.")
@click.option("--workers", default=4, help="DataLoader workers.")
def quantize(
    model: str,
    data: str,
    data_format: str,
    num_batches: int,
    batch_size: int,
    backend: str,
    input_size: int,
    output: str,
    checkpoint: str | None,
    device: str,
    workers: int,
):
    """Run Post-Training Quantization (PTQ) on a YOLO-NAS model."""
    from pathlib import Path

    import torch
    from rich.console import Console

    from modern_yolonas import yolo_nas_s, yolo_nas_m, yolo_nas_l
    from modern_yolonas.data.transforms import Compose, LetterboxResize, Normalize
    from modern_yolonas.data.collate import detection_collate_fn
    from modern_yolonas.quantization import (
        prepare_model_ptq,
        run_calibration,
        convert_quantized,
        export_quantized_onnx,
    )

    console = Console()

    # Build model
    builders = {"yolo_nas_s": yolo_nas_s, "yolo_nas_m": yolo_nas_m, "yolo_nas_l": yolo_nas_l}
    console.print(f"Loading {model}...")

    if checkpoint:
        yolo_model = builders[model](pretrained=False)
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
        sd = ckpt.get("model_state_dict", ckpt)
        yolo_model.load_state_dict(sd)
    else:
        yolo_model = builders[model](pretrained=True)

    # Prepare for PTQ
    console.print(f"Preparing model for PTQ (backend={backend})...")
    example_input = torch.randn(1, 3, input_size, input_size)
    ptq_model = prepare_model_ptq(yolo_model, backend=backend, example_input=example_input)

    # Build calibration dataloader
    transforms = Compose([LetterboxResize(target_size=input_size), Normalize()])

    if data_format == "yolo":
        from modern_yolonas.data.yolo import YOLODetectionDataset

        dataset = YOLODetectionDataset(data, split="val", transforms=transforms, input_size=input_size)
    else:
        from modern_yolonas.data.coco import COCODetectionDataset

        data_path = Path(data)
        dataset = COCODetectionDataset(
            data_path / "images" / "val2017",
            data_path / "annotations" / "instances_val2017.json",
            transforms=transforms,
            input_size=input_size,
        )

    from torch.utils.data import DataLoader

    cal_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        collate_fn=detection_collate_fn,
        pin_memory=False,
    )

    # Calibrate
    run_calibration(ptq_model, cal_loader, num_batches=num_batches, device=device)

    # Convert
    console.print("Converting to quantized model...")
    quantized_model = convert_quantized(ptq_model)

    # Export
    console.print(f"Exporting to {output}...")
    if output.endswith(".onnx"):
        export_quantized_onnx(quantized_model, output, input_size=input_size)
    else:
        torch.save(quantized_model.state_dict(), output)

    console.print(f"[green]PTQ complete. Saved to {output}[/green]")
