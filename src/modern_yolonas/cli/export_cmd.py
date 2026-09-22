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
    embedding = "embedding"
    combined = "combined"
    objects = "objects"


class EmbedPooling(str, Enum):
    avg = "avg"
    max = "max"


_DEFAULT_EXT = {ExportFormat.onnx: "onnx", ExportFormat.openvino: "xml", ExportFormat.tensorrt: "engine"}

# Targets that rewrite or replace the base graph. TensorRT can build the plain and
# end2end ones; the rest produce graphs with inputs or outputs its engine path does
# not model, so they stop at ONNX or OpenVINO IR.
_GRAPH_TARGETS = {ExportTarget.frigate, ExportTarget.embedding, ExportTarget.combined, ExportTarget.objects}


def export(
    model: Annotated[ModelName, typer.Option(help="Model variant.")] = ModelName.yolo_nas_s,
    export_format: Annotated[ExportFormat, typer.Option("--format", help="Export format.")] = ExportFormat.onnx,
    output: Annotated[str | None, typer.Option(help="Output file path.")] = None,
    input_size: Annotated[int, typer.Option(help="Model input size. Baked into the graph — one file per size.")] = 640,
    precision: Annotated[
        Precision,
        typer.Option(help="Weight precision. openvino: fp32/fp16/int8; tensorrt: fp32/fp16; onnx: fp32 only."),
    ] = Precision.fp32,
    opset: Annotated[
        int, typer.Option(help="ONNX opset version (18 is torch's floor here; lower is requested, not honoured).")
    ] = 18,
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
    conf_threshold: Annotated[
        float, typer.Option(help="Confidence threshold baked in (end2end/frigate/objects).")
    ] = 0.25,
    iou_threshold: Annotated[float, typer.Option(help="NMS IoU threshold baked in (end2end/frigate/objects).")] = 0.45,
    max_detections: Annotated[
        int,
        typer.Option(help="Max detections per class, baked in (end2end/frigate/objects) — ONNX NMS counts per class."),
    ] = 20,
    embed_layers: Annotated[
        str, typer.Option(help="Comma-separated feature maps to pool (embedding/combined/objects targets).")
    ] = "c5",
    embed_pooling: Annotated[
        EmbedPooling, typer.Option(help="Spatial pooling (embedding/combined/objects targets).")
    ] = EmbedPooling.avg,
    normalize: Annotated[
        bool, typer.Option(help="L2-normalize the embedding, so a dot product is a cosine.")
    ] = True,
):
    """Export model to ONNX, OpenVINO IR or a TensorRT engine.

    `--target end2end` bakes NMS into the graph, so the model returns detections
    `[D, 7]` instead of raw `[N, 4]` + `[N, C]` tensors. Preprocessing stays outside:
    the letterbox scale and padding are per-image and the caller needs them to map
    boxes back to original pixels.
    """
    import tempfile
    from pathlib import Path

    import torch
    from rich.console import Console

    from modern_yolonas import load_checkpoint, yolo_nas_l, yolo_nas_m, yolo_nas_s
    from modern_yolonas.export import export_onnx
    from modern_yolonas.export.onnx import fuse_for_inference

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
    if export_format == ExportFormat.tensorrt and target in _GRAPH_TARGETS:
        raise typer.BadParameter(
            f"the {target.value} target produces ONNX or OpenVINO IR, not a TensorRT engine"
        )
    if precision == Precision.int8 and target != ExportTarget.generic:
        raise typer.BadParameter(
            f"the {target.value} target rewrites the graph in ways NNCF cannot calibrate through; "
            "export int8 with --target generic"
        )

    builders = {"yolo_nas_s": yolo_nas_s, "yolo_nas_m": yolo_nas_m, "yolo_nas_l": yolo_nas_l}
    console.print(f"Loading {model.value}...")

    if checkpoint:
        yolo_model = builders[model.value](pretrained=False, num_classes=num_classes)
        load_checkpoint(yolo_model, checkpoint)
    else:
        yolo_model = builders[model.value](pretrained=True, num_classes=num_classes)

    fuse_for_inference(yolo_model)

    if target == ExportTarget.objects:
        _export_objects(
            yolo_model, output, export_format, opset, input_size,
            embed_layers, embed_pooling.value, normalize,
            conf_threshold, iou_threshold, max_detections, console,
        )
        console.print(f"[green]Exported to {output}[/green]")
        return

    if target in (ExportTarget.embedding, ExportTarget.combined):
        _export_embedding(
            yolo_model, output, export_format, opset, input_size, target,
            embed_layers, embed_pooling.value, normalize, console,
        )
        console.print(f"[green]Exported to {output}[/green]")
        return

    if target == ExportTarget.frigate:
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
            fuse=False,
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
