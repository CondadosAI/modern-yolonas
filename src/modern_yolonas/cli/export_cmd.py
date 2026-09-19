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
    embedding = "embedding"
    combined = "combined"
    objects = "objects"


class EmbedPooling(str, Enum):
    avg = "avg"
    max = "max"


def export(
    model: Annotated[ModelName, typer.Option(help="Model variant.")] = ModelName.yolo_nas_s,
    export_format: Annotated[ExportFormat, typer.Option("--format", help="Export format.")] = ExportFormat.onnx,
    output: Annotated[str | None, typer.Option(help="Output file path.")] = None,
    input_size: Annotated[int, typer.Option(help="Model input size.")] = 640,
    opset: Annotated[int, typer.Option(help="ONNX opset version (18 is torch's floor here; lower is requested, not honoured).")] = 18,
    checkpoint: Annotated[str | None, typer.Option(help="Custom checkpoint path.")] = None,
    num_classes: Annotated[int, typer.Option(help="Number of classes (must match checkpoint; default 80 for COCO).")] = 80,
    target: Annotated[ExportTarget, typer.Option(help="Export target.")] = ExportTarget.generic,
    conf_threshold: Annotated[float, typer.Option(help="Confidence threshold (frigate/objects targets).")] = 0.25,
    iou_threshold: Annotated[float, typer.Option(help="IoU threshold for NMS (frigate/objects targets).")] = 0.45,
    max_detections: Annotated[
        int, typer.Option(help="Max detections per class (frigate/objects targets) — ONNX NMS counts per class, not per image.")
    ] = 20,
    embed_layers: Annotated[
        str, typer.Option(help="Comma-separated feature maps to pool (embedding/combined targets).")
    ] = "c5",
    embed_pooling: Annotated[
        EmbedPooling, typer.Option(help="Spatial pooling (embedding/combined targets).")
    ] = EmbedPooling.avg,
    normalize: Annotated[
        bool, typer.Option(help="L2-normalize the embedding, so a dot product is a cosine.")
    ] = True,
):
    """Export model to ONNX or OpenVINO format."""
    import torch
    from rich.console import Console

    from modern_yolonas import yolo_nas_s, yolo_nas_m, yolo_nas_l, load_checkpoint

    console = Console()

    if output is None:
        ext = "xml" if export_format == ExportFormat.openvino else "onnx"
        suffix = "" if target == ExportTarget.generic else f"_{target.value}"
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

    if target == ExportTarget.objects:
        _export_objects(
            yolo_model, output, export_format, opset, input_size,
            embed_layers, embed_pooling.value, normalize,
            conf_threshold, iou_threshold, max_detections, console,
        )
    elif target in (ExportTarget.embedding, ExportTarget.combined):
        _export_embedding(
            yolo_model, output, export_format, opset, input_size, target,
            embed_layers, embed_pooling.value, normalize, console,
        )
    elif target == ExportTarget.frigate:
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
            # See `_onnx_export`: without this the weights land in a sibling
            # `<name>.onnx.data` and the .onnx alone is not a working model.
            external_data=False,
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


def _export_embedding(
    yolo_model, output, export_format, opset, input_size, target,
    embed_layers, embed_pooling, normalize, console,
):
    """Export a graph that emits feature embeddings, with or without detections.

    Both graphs take a second input, ``valid_region`` — ``[B, 4]`` int64
    ``(left, top, right, bottom)`` in canvas pixels, from
    ``modern_yolonas.inference.embed.valid_region``. It says where the real pixels
    sit inside the letterbox, and it cannot be baked in because it depends on the
    aspect ratio of the image being embedded. Pooling without it makes embeddings
    cluster by aspect ratio rather than by content.
    """
    import torch

    from modern_yolonas.export.embedding import DetectAndEmbedGraph, EmbeddingGraph
    from modern_yolonas.inference.embed import FeaturePooler

    layers = tuple(name.strip() for name in embed_layers.split(",") if name.strip())
    pooler = FeaturePooler(layers=layers, pooling=embed_pooling, normalize=normalize)

    if target == ExportTarget.combined:
        graph = DetectAndEmbedGraph(yolo_model, pooler, input_size).eval()
        output_names = ["pred_bboxes", "pred_scores", "embedding"]
    else:
        graph = EmbeddingGraph(yolo_model, pooler, input_size).eval()
        output_names = ["embedding"]

    dummy = (
        torch.randn(1, 3, input_size, input_size),
        torch.tensor([[0, 0, input_size, input_size]], dtype=torch.long),
    )
    input_names = ["images", "valid_region"]

    console.print(f"Pooling {'+'.join(layers)} ({embed_pooling}, {'normalized' if normalize else 'raw'})...")

    if export_format == ExportFormat.openvino:
        import tempfile
        from pathlib import Path

        import openvino as ov

        with tempfile.TemporaryDirectory() as tmpdir:
            base = str(Path(tmpdir) / "embedding.onnx")
            console.print(f"Exporting base ONNX (opset {opset})...")
            _onnx_export(graph, dummy, base, input_names, output_names, opset)
            console.print("Converting to OpenVINO IR...")
            ov.save_model(ov.convert_model(base), output)
    else:
        console.print(f"Exporting to ONNX (opset {opset})...")
        _onnx_export(graph, dummy, output, input_names, output_names, opset)


def _onnx_export(graph, dummy, path, input_names, output_names, opset):
    import torch

    torch.onnx.export(
        graph,
        dummy,
        path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={name: {0: "batch"} for name in input_names + output_names},
        opset_version=opset,
        # One self-contained file. Torch's dynamo exporter otherwise writes the
        # weights to a sibling `<name>.onnx.data`, and an .onnx shipped without
        # its sidecar loads and then fails — a bad way to find out.
        external_data=False,
    )


def _export_objects(
    yolo_model, output, export_format, opset, input_size,
    embed_layers, embed_pooling, normalize,
    conf_threshold, iou_threshold, max_detections, console,
):
    """Export a self-contained graph: image in, detections and per-object vectors out.

    Unlike the `embedding` and `combined` targets this one cannot be traced
    straight out of PyTorch — which boxes exist depends on which survive NMS, a
    data-dependent shape ``torch.export`` will not produce. So a base graph is
    exported with its feature maps as outputs, and NMS, ROI pooling and the
    normalization are added as ONNX nodes on top.
    """
    import tempfile

    from pathlib import Path

    import torch

    from modern_yolonas.export.embedding import DetectAndFeatureGraph
    from modern_yolonas.export.objects import make_object_embedding_onnx
    from modern_yolonas.inference.embed import FeaturePooler

    layers = tuple(name.strip() for name in embed_layers.split(",") if name.strip())
    pooler = FeaturePooler(layers=layers, pooling=embed_pooling, normalize=normalize)
    graph = DetectAndFeatureGraph(yolo_model, pooler, input_size).eval()

    dummy = (
        torch.randn(1, 3, input_size, input_size),
        torch.tensor([[0, 0, input_size, input_size]], dtype=torch.long),
    )

    console.print(f"Pooling {'+'.join(layers)} ({embed_pooling}, {'normalized' if normalize else 'raw'})...")

    with tempfile.TemporaryDirectory() as tmpdir:
        base = str(Path(tmpdir) / "base.onnx")
        console.print(f"Exporting base ONNX (opset {opset})...")
        _onnx_export(graph, dummy, base, ["images", "valid_region"], graph.output_names, opset)

        target_onnx = str(Path(tmpdir) / "objects.onnx") if export_format == ExportFormat.openvino else output

        console.print("Applying graph surgery (NMS + ROI pooling)...")
        make_object_embedding_onnx(
            base,
            target_onnx,
            layers=layers,
            canvas=input_size,
            pooling=embed_pooling,
            normalize=normalize,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
        )

        if export_format == ExportFormat.openvino:
            import openvino as ov

            console.print("Converting to OpenVINO IR...")
            ov.save_model(ov.convert_model(target_onnx), output)
