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
    tensorrt = "tensorrt"


class Precision(str, Enum):
    fp32 = "fp32"
    fp16 = "fp16"
    int8 = "int8"


class ExportTarget(str, Enum):
    generic = "generic"
    end2end = "end2end"
    frigate = "frigate"


_DEFAULT_EXT = {ExportFormat.onnx: "onnx", ExportFormat.openvino: "xml", ExportFormat.tensorrt: "engine"}


def export(
    model: Annotated[ModelName, typer.Option(help="Model variant.")] = ModelName.yolo_nas_s,
    export_format: Annotated[ExportFormat, typer.Option("--format", help="Export format.")] = ExportFormat.onnx,
    output: Annotated[str | None, typer.Option(help="Output file path.")] = None,
    input_size: Annotated[int, typer.Option(help="Model input size. Baked into the graph — one file per size.")] = 640,
    precision: Annotated[
        Precision,
        typer.Option(help="Weight precision. openvino: fp32/fp16/int8; tensorrt: fp32/fp16; onnx: fp32 only."),
    ] = Precision.fp32,
    opset: Annotated[int, typer.Option(help="ONNX opset version.")] = 18,
    static_batch: Annotated[
        bool, typer.Option("--static-batch/--dynamic-batch", help="Fix batch at 1 instead of leaving it symbolic.")
    ] = False,
    hardware_compatible: Annotated[
        bool,
        typer.Option(
            "--hardware-compatible",
            help="TensorRT: build an AMPERE_PLUS engine that loads on any sm_80+ GPU, not only this one. "
            "Slower than a native build — that is the price of a publishable engine.",
        ),
    ] = False,
    version_compatible: Annotated[
        bool,
        typer.Option("--version-compatible", help="TensorRT: let the engine load under a later TensorRT release."),
    ] = False,
    calibration_dir: Annotated[
        str | None, typer.Option(help="Directory of representative images for INT8 calibration (openvino int8).")
    ] = None,
    calibration_images: Annotated[int, typer.Option(help="How many images to calibrate on.")] = 128,
    checkpoint: Annotated[str | None, typer.Option(help="Custom checkpoint path.")] = None,
    num_classes: Annotated[int, typer.Option(help="Number of classes (must match checkpoint; default 80 for COCO).")] = 80,
    target: Annotated[ExportTarget, typer.Option(help="Export target.")] = ExportTarget.generic,
    conf_threshold: Annotated[float, typer.Option(help="Confidence threshold baked in (end2end/frigate).")] = 0.25,
    iou_threshold: Annotated[float, typer.Option(help="NMS IoU threshold baked in (end2end/frigate).")] = 0.45,
    max_detections: Annotated[
        int, typer.Option(help="Max detections per class per image, baked in (end2end/frigate).")
    ] = 20,
):
    """Export model to ONNX, OpenVINO IR or a TensorRT engine.

    `--target end2end` bakes NMS into the graph, so the model returns detections
    `[D, 7]` instead of raw `[N, 4]` + `[N, C]` tensors. Preprocessing stays outside:
    the letterbox scale and padding are per-image and the caller needs them to map
    boxes back to original pixels.
    """
    import tempfile
    from pathlib import Path

    from rich.console import Console

    from modern_yolonas import load_checkpoint, yolo_nas_l, yolo_nas_m, yolo_nas_s
    from modern_yolonas.export import export_onnx

    console = Console()

    if output is None:
        suffix = "" if target == ExportTarget.generic else f"_{target.value}"
        prec = "" if precision == Precision.fp32 else f"_{precision.value}"
        output = f"{model.value}_{input_size}{prec}{suffix}.{_DEFAULT_EXT[export_format]}"

    if export_format == ExportFormat.onnx and precision != Precision.fp32:
        raise typer.BadParameter(
            f"--format onnx exports fp32 weights; {precision.value} is a property of the runtime that consumes "
            "the graph. Use --format openvino or --format tensorrt, or `yolonas quantize` for a QDQ graph."
        )
    if export_format == ExportFormat.tensorrt and precision == Precision.int8:
        raise typer.BadParameter(
            "TensorRT 11 removed the implicit INT8 calibrator. Produce a QDQ ONNX with `yolonas quantize` "
            "and build it with --precision fp16; TensorRT honours the Q/DQ nodes it finds."
        )
    if export_format == ExportFormat.tensorrt and target == ExportTarget.frigate:
        raise typer.BadParameter("the frigate target produces ONNX or OpenVINO IR, not a TensorRT engine")
    if target != ExportTarget.generic and precision == Precision.int8:
        raise typer.BadParameter(
            f"the {target.value} target bakes NMS into the graph, which NNCF cannot calibrate through; "
            "export int8 with --target generic"
        )

    builders = {"yolo_nas_s": yolo_nas_s, "yolo_nas_m": yolo_nas_m, "yolo_nas_l": yolo_nas_l}
    console.print(f"Loading {model.value}...")

    if checkpoint:
        yolo_model = builders[model.value](pretrained=False, num_classes=num_classes)
        load_checkpoint(yolo_model, checkpoint)
    else:
        yolo_model = builders[model.value](pretrained=True, num_classes=num_classes)

    if target == ExportTarget.frigate:
        import torch

        from modern_yolonas.export.onnx import fuse_for_inference

        fuse_for_inference(yolo_model)
        dummy = torch.randn(1, 3, input_size, input_size)
        _export_frigate(
            yolo_model, dummy, output, export_format, opset, conf_threshold, iou_threshold, max_detections, console
        )
        console.print(f"[green]Exported to {output}[/green]")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        needs_surgery = target == ExportTarget.end2end
        if export_format == ExportFormat.onnx and not needs_surgery:
            onnx_path = Path(output)
        else:
            onnx_path = Path(tmpdir) / "base.onnx"

        console.print(f"Exporting ONNX (opset {opset}, {input_size}x{input_size})...")
        export_onnx(
            yolo_model,
            onnx_path,
            input_size=input_size,
            opset=opset,
            # Only the plain ONNX artifact keeps a symbolic batch. TensorRT would need
            # an optimisation profile for it, OpenVINO's INT8 calibration is simpler on
            # a static shape, and the NMS surgery indexes the batch dimension.
            dynamic_batch=(export_format == ExportFormat.onnx and not needs_surgery and not static_batch),
            # TensorRT 11 is strongly typed: FP16 has to be in the graph, not a flag.
            half=(export_format == ExportFormat.tensorrt and precision == Precision.fp16),
        )

        if needs_surgery:
            from modern_yolonas.export.nms import make_end2end_onnx

            console.print(f"Baking NMS into the graph (conf {conf_threshold}, IoU {iou_threshold})...")
            e2e_path = Path(output) if export_format == ExportFormat.onnx else Path(tmpdir) / "end2end.onnx"
            make_end2end_onnx(
                str(onnx_path),
                str(e2e_path),
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                max_detections=max_detections,
            )
            onnx_path = e2e_path

        if export_format == ExportFormat.openvino:
            from modern_yolonas.export.openvino import export_openvino

            console.print(f"Converting to OpenVINO IR ({precision.value})...")
            export_openvino(
                onnx_path,
                output,
                precision=precision.value,
                calibration_dir=calibration_dir,
                input_size=input_size,
                calibration_images=calibration_images,
            )
        elif export_format == ExportFormat.tensorrt:
            from modern_yolonas.export.tensorrt import build_engine

            console.print(f"Building TensorRT engine ({precision.value}) — this takes a few minutes...")
            build_engine(
                onnx_path, output, hardware_compatible=hardware_compatible, version_compatible=version_compatible
            )
            if hardware_compatible:
                console.print("[yellow]AMPERE_PLUS engine: loads on any sm_80+ GPU, slower than a native build.[/yellow]")
            else:
                console.print(
                    "[yellow]An engine only loads on the GPU and TensorRT version it was built for; "
                    f"{Path(output).name}.json records which. Pass --hardware-compatible to widen that.[/yellow]"
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
