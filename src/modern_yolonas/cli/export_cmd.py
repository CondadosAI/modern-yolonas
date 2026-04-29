"""CLI: yolonas export"""

from __future__ import annotations

from enum import Enum
from typing import Annotated

import typer


class ModelName(str, Enum):
    yolo_nas_s = "yolo_nas_s"
    yolo_nas_m = "yolo_nas_m"
    yolo_nas_l = "yolo_nas_l"


class ExportFormat(str, Enum):
    onnx = "onnx"
    openvino = "openvino"


class ExportTarget(str, Enum):
    generic = "generic"
    frigate = "frigate"


def export(
    model: Annotated[ModelName, typer.Option(help="Model variant.")] = ModelName.yolo_nas_s,
    export_format: Annotated[ExportFormat, typer.Option("--format", help="Export format.")] = ExportFormat.onnx,
    output: Annotated[str | None, typer.Option(help="Output file path.")] = None,
    input_size: Annotated[int, typer.Option(help="Model input size.")] = 640,
    opset: Annotated[int, typer.Option(help="ONNX opset version.")] = 17,
    checkpoint: Annotated[str | None, typer.Option(help="Custom checkpoint path.")] = None,
    num_classes: Annotated[int, typer.Option(help="Number of classes (must match checkpoint; default 80 for COCO).")] = 80,
    target: Annotated[ExportTarget, typer.Option(help="Export target.")] = ExportTarget.generic,
    conf_threshold: Annotated[float, typer.Option(help="Confidence threshold (frigate target).")] = 0.25,
    iou_threshold: Annotated[float, typer.Option(help="IoU threshold for NMS (frigate target).")] = 0.45,
    max_detections: Annotated[int, typer.Option(help="Max detections per image (frigate target).")] = 20,
):
    """Export model to ONNX or OpenVINO format."""
    import torch
    from rich.console import Console

    from modern_yolonas import yolo_nas_s, yolo_nas_m, yolo_nas_l, load_checkpoint

    console = Console()

    if output is None:
        ext = "xml" if export_format == ExportFormat.openvino else "onnx"
        suffix = "_frigate" if target == ExportTarget.frigate else ""
        output = f"{model.value}{suffix}.{ext}"

    builders = {"yolo_nas_s": yolo_nas_s, "yolo_nas_m": yolo_nas_m, "yolo_nas_l": yolo_nas_l}
    console.print(f"Loading {model.value}...")

    if checkpoint:
        yolo_model = builders[model.value](pretrained=False, num_classes=num_classes)
        load_checkpoint(yolo_model, checkpoint)
    else:
        yolo_model = builders[model.value](pretrained=True, num_classes=num_classes)

    yolo_model.eval()

    for module in yolo_model.modules():
        if hasattr(module, "fuse_block_residual_branches"):
            module.fuse_block_residual_branches()

    dummy = torch.randn(1, 3, input_size, input_size)

    if target == ExportTarget.frigate:
        _export_frigate(yolo_model, dummy, output, export_format, opset, conf_threshold, iou_threshold, max_detections, console)
    elif export_format == ExportFormat.openvino:
        import openvino as ov

        console.print("Exporting to OpenVINO IR...")
        ov_model = ov.convert_model(yolo_model, example_input=dummy)
        ov.save_model(ov_model, output)
    else:
        console.print(f"Exporting to ONNX (opset {opset})...")
        torch.onnx.export(
            yolo_model,
            dummy,
            output,
            input_names=["images"],
            output_names=["pred_bboxes", "pred_scores"],
            dynamic_axes={
                "images": {0: "batch"},
                "pred_bboxes": {0: "batch"},
                "pred_scores": {0: "batch"},
            },
            opset_version=opset,
        )

    console.print(f"[green]Exported to {output}[/green]")


def _export_frigate(yolo_model, dummy, output, export_format, opset, conf_threshold, iou_threshold, max_detections, console):
    """Export with Frigate-compatible preprocessing + NMS baked in."""
    import tempfile
    from pathlib import Path

    import torch

    from modern_yolonas.export.frigate import make_frigate_onnx

    with tempfile.TemporaryDirectory() as tmpdir:
        base_onnx = str(Path(tmpdir) / "base.onnx")

        console.print(f"Exporting base ONNX (opset {opset})...")
        torch.onnx.export(
            yolo_model,
            dummy,
            base_onnx,
            input_names=["images"],
            output_names=["pred_bboxes", "pred_scores"],
            opset_version=opset,
        )

        if export_format == ExportFormat.openvino:
            frigate_onnx = str(Path(tmpdir) / "frigate.onnx")
        else:
            frigate_onnx = output

        console.print("Applying Frigate graph surgery (preproc + NMS)...")
        make_frigate_onnx(
            base_onnx,
            frigate_onnx,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
        )

        if export_format == ExportFormat.openvino:
            import openvino as ov

            console.print("Converting Frigate ONNX to OpenVINO IR...")
            ov_model = ov.convert_model(frigate_onnx)
            ov.save_model(ov_model, output)
