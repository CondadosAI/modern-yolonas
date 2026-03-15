"""CLI: yolonas export"""

from __future__ import annotations

import click


@click.command()
@click.option("--model", default="yolo_nas_s", type=click.Choice(["yolo_nas_s", "yolo_nas_m", "yolo_nas_l"]))
@click.option("--format", "export_format", default="onnx", type=click.Choice(["onnx", "openvino"]))
@click.option("--output", default=None, help="Output file path (default: model.onnx or model.xml).")
@click.option("--input-size", default=640, help="Model input size.")
@click.option("--opset", default=17, help="ONNX opset version.")
@click.option("--checkpoint", default=None, help="Custom checkpoint path.")
@click.option("--target", default="generic", type=click.Choice(["generic", "frigate"]), help="Export target.")
@click.option("--conf-threshold", default=0.25, help="Confidence threshold (frigate target).")
@click.option("--iou-threshold", default=0.45, help="IoU threshold for NMS (frigate target).")
@click.option("--max-detections", default=20, help="Max detections per image (frigate target).")
def export(
    model: str,
    export_format: str,
    output: str | None,
    input_size: int,
    opset: int,
    checkpoint: str | None,
    target: str,
    conf_threshold: float,
    iou_threshold: float,
    max_detections: int,
):
    """Export model to ONNX or OpenVINO format."""
    import torch
    from rich.console import Console

    from modern_yolonas import yolo_nas_s, yolo_nas_m, yolo_nas_l

    console = Console()

    if output is None:
        ext = "xml" if export_format == "openvino" else "onnx"
        suffix = "_frigate" if target == "frigate" else ""
        output = f"{model}{suffix}.{ext}"

    builders = {"yolo_nas_s": yolo_nas_s, "yolo_nas_m": yolo_nas_m, "yolo_nas_l": yolo_nas_l}
    console.print(f"Loading {model}...")

    if checkpoint:
        yolo_model = builders[model](pretrained=False)
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
        sd = ckpt.get("model_state_dict", ckpt)
        yolo_model.load_state_dict(sd)
    else:
        yolo_model = builders[model](pretrained=True)

    yolo_model.eval()

    # Fuse RepVGG blocks for deployment
    for module in yolo_model.modules():
        if hasattr(module, "fuse_block_residual_branches"):
            module.fuse_block_residual_branches()

    dummy = torch.randn(1, 3, input_size, input_size)

    if target == "frigate":
        _export_frigate(yolo_model, dummy, output, export_format, opset, conf_threshold, iou_threshold, max_detections, console)
    elif export_format == "openvino":
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

        if export_format == "openvino":
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

        if export_format == "openvino":
            import openvino as ov

            console.print("Converting Frigate ONNX to OpenVINO IR...")
            ov_model = ov.convert_model(frigate_onnx)
            ov.save_model(ov_model, output)
