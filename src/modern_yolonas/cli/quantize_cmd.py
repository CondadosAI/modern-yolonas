"""CLI: yolonas quantize — Post-Training Quantization."""

from __future__ import annotations

from enum import Enum
from typing import Annotated

import typer

from modern_yolonas.cli.train_cmd import DataFormat, ModelName


class Backend(str, Enum):
    x86 = "x86"
    qnnpack = "qnnpack"
    onednn = "onednn"


def quantize(
    data: Annotated[str, typer.Option(help="Path to calibration dataset root.")],
    model: Annotated[ModelName, typer.Option(help="Model variant.")] = ModelName.yolo_nas_s,
    data_format: Annotated[DataFormat, typer.Option("--format", help="Dataset format.")] = DataFormat.yolo,
    num_batches: Annotated[int, typer.Option(help="Number of calibration batches.")] = 100,
    batch_size: Annotated[int, typer.Option(help="Batch size for calibration.")] = 32,
    backend: Annotated[Backend, typer.Option(help="Quantization backend.")] = Backend.x86,
    input_size: Annotated[int, typer.Option(help="Model input size.")] = 640,
    output: Annotated[str, typer.Option(help="Output file (.onnx or .pt).")] = "model_ptq.onnx",
    checkpoint: Annotated[str | None, typer.Option(help="Custom checkpoint path.")] = None,
    device: Annotated[str, typer.Option(help="Calibration device.")] = "cpu",
    workers: Annotated[int, typer.Option(help="DataLoader workers.")] = 4,
):
    """Run Post-Training Quantization (PTQ) on a YOLO-NAS model."""
    from pathlib import Path

    import torch
    from rich.console import Console
    from torch.utils.data import DataLoader

    from modern_yolonas import yolo_nas_l, yolo_nas_m, yolo_nas_s
    from modern_yolonas.data.collate import detection_collate_fn
    from modern_yolonas.data.transforms import Compose, LetterboxResize, Normalize
    from modern_yolonas.quantization import (
        convert_quantized,
        export_quantized_onnx,
        prepare_model_ptq,
        run_calibration,
    )
    from modern_yolonas.weights import extract_model_state_dict

    console = Console()

    builders = {"yolo_nas_s": yolo_nas_s, "yolo_nas_m": yolo_nas_m, "yolo_nas_l": yolo_nas_l}
    console.print(f"Loading {model.value}...")

    if checkpoint:
        yolo_model = builders[model.value](pretrained=False)
        yolo_model.load_state_dict(extract_model_state_dict(checkpoint))
    else:
        yolo_model = builders[model.value](pretrained=True)

    console.print(f"Preparing model for PTQ (backend={backend.value})...")
    example_input = torch.randn(1, 3, input_size, input_size)
    ptq_model = prepare_model_ptq(yolo_model, backend=backend.value, example_input=example_input)

    transforms = Compose([LetterboxResize(target_size=input_size), Normalize()])

    if data_format == DataFormat.yolo:
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

    cal_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        collate_fn=detection_collate_fn,
        pin_memory=False,
    )

    run_calibration(ptq_model, cal_loader, num_batches=num_batches, device=device)

    console.print("Converting to quantized model...")
    quantized_model = convert_quantized(ptq_model)

    console.print(f"Exporting to {output}...")
    if output.endswith(".onnx"):
        export_quantized_onnx(quantized_model, output, input_size=input_size)
    else:
        torch.save(quantized_model.state_dict(), output)

    console.print(f"[green]PTQ complete. Saved to {output}[/green]")
