"""ONNX graph surgery for per-detection embeddings.

Turns a :class:`~modern_yolonas.export.embedding.DetectAndFeatureGraph` base
export into a self-contained model: image in, detections *and* one feature
vector per detection out, with NMS and ROI pooling inside the graph.

Per-object embeddings cannot be traced straight out of PyTorch, because which
boxes exist depends on which survive NMS — a data-dependent shape that
``torch.export`` will not produce. So the base graph emits its feature maps as
outputs, and the nodes added here consume them:

    pred_bboxes ─┐
    pred_scores ─┴─► NonMaxSuppression ─► selected_indices [D, 3]
                                              │
                          gather boxes ◄──────┤
                          clip to valid_region│
                                   │          │
    feat_<layer> ──► RoiAlign ◄─────┘          │
            └─► GlobalAvg/MaxPool ─► concat ─► L2 ─► object_embedding [D, E]
                                              │
                                              └─► detections [D, 7]

Row *i* of ``object_embedding`` describes row *i* of ``detections``.
"""

from __future__ import annotations

import numpy as np
import onnx

from onnx import TensorProto, helper

from modern_yolonas.export.frigate import make_constant

#: Grid an ROI is resampled to before pooling — must match
#: :data:`modern_yolonas.inference.embed._ROI_GRID`.
ROI_GRID = 3

#: Guard for the L2 normalization, matching ``torch.nn.functional.normalize``'s
#: ``max(norm, eps)``. A box that clips to zero area samples nothing, and this
#: keeps it a zero vector instead of turning it into NaN.
_NORM_EPS = 1e-12


def _feature_dims(graph: onnx.GraphProto, name: str) -> tuple[int, int]:
    """Static ``(channels, width)`` of a named graph output.

    Width sets the ROI spatial scale; channels add up to the embedding width, which
    is declared on the output so a caller can size an index without a dummy run.
    """
    for output in graph.output:
        if output.name == name:
            dims = output.type.tensor_type.shape.dim
            channels, width = dims[1].dim_value, dims[3].dim_value
            if channels <= 0 or width <= 0:
                raise ValueError(f"output {name!r} has no static shape; export at a fixed input size")
            return channels, width
    raise ValueError(f"base graph has no output named {name!r}")


def make_object_embedding_onnx(
    base_onnx_path: str,
    output_path: str,
    layers: tuple[str, ...],
    canvas: int,
    pooling: str = "avg",
    normalize: bool = True,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_detections: int = 20,
) -> None:
    """Rewrite a base graph into one that emits a vector per detection.

    Args:
        base_onnx_path: A :class:`DetectAndFeatureGraph` export.
        output_path: Where to write the result.
        layers: Feature maps to pool, in the order the base graph emits them.
        canvas: Letterbox side length the base graph was exported at.
        pooling: ``"avg"`` or ``"max"`` over the ROI grid.
        normalize: L2-normalize each vector.
        conf_threshold: Score threshold for NMS.
        iou_threshold: IoU threshold for NMS.
        max_detections: ``max_output_boxes_per_class`` — per class, not per
            image, which is what the ONNX NMS operator takes.

    The resulting model has input ``images`` and ``valid_region`` and outputs
    ``detections [D, 7]`` (``batch, x1, y1, x2, y2, confidence, class_id``, in
    canvas coordinates), ``object_embedding [D, E]`` and ``embedding [B, E]``.
    """
    model = onnx.load(base_onnx_path)
    graph = model.graph

    nodes: list[onnx.NodeProto] = []

    # ------------------------------------------------------------------
    # NMS — per class, the same operator and semantics the frigate target uses
    # ------------------------------------------------------------------
    nodes += [
        helper.make_node("Transpose", ["pred_scores"], ["scores_nms"], perm=[0, 2, 1]),
        make_constant("max_det", np.array([max_detections], dtype=np.int64)),
        make_constant("iou_thr", np.array([iou_threshold], dtype=np.float32)),
        make_constant("conf_thr", np.array([conf_threshold], dtype=np.float32)),
        helper.make_node(
            "NonMaxSuppression",
            ["pred_bboxes", "scores_nms", "max_det", "iou_thr", "conf_thr"],
            ["selected_indices"],
        ),
    ]

    # ------------------------------------------------------------------
    # Split the [D, 3] selection into its columns
    # ------------------------------------------------------------------
    nodes += [
        make_constant("col_0", np.array(0, dtype=np.int64)),
        make_constant("col_1", np.array(1, dtype=np.int64)),
        make_constant("col_2", np.array(2, dtype=np.int64)),
        helper.make_node("Gather", ["selected_indices", "col_0"], ["batch_idx"], axis=1),
        helper.make_node("Gather", ["selected_indices", "col_1"], ["class_idx"], axis=1),
        helper.make_node("Gather", ["selected_indices", "col_2"], ["box_idx"], axis=1),
        make_constant("axis_1", np.array([1], dtype=np.int64)),
        helper.make_node("Unsqueeze", ["batch_idx", "axis_1"], ["batch_2d"]),
        helper.make_node("Unsqueeze", ["class_idx", "axis_1"], ["class_2d"]),
        helper.make_node("Unsqueeze", ["box_idx", "axis_1"], ["box_2d"]),
        helper.make_node("Concat", ["batch_2d", "box_2d"], ["bbox_gather_idx"], axis=1),
        helper.make_node("GatherND", ["pred_bboxes", "bbox_gather_idx"], ["raw_boxes"]),
        helper.make_node("Concat", ["batch_2d", "box_2d", "class_2d"], ["score_gather_idx"], axis=1),
        helper.make_node("GatherND", ["pred_scores", "score_gather_idx"], ["selected_scores"]),
        helper.make_node("Unsqueeze", ["selected_scores", "axis_1"], ["scores_2d"]),
    ]

    # ------------------------------------------------------------------
    # Clip each box to its image's valid region
    #
    # `rescale_boxes` clips on the PyTorch side, and `embed_boxes` and
    # `predict(EMBED_OBJECTS)` were made to agree on embedding the clipped box:
    # outside the valid region there is only letterbox padding, so a detection
    # running off the edge should be described by the part of it that is real.
    # Without this the graph would reintroduce that disagreement on every box
    # that touches an edge.
    # ------------------------------------------------------------------
    nodes += [
        helper.make_node("Cast", ["valid_region"], ["region_f"], to=TensorProto.FLOAT),
        helper.make_node("Gather", ["region_f", "batch_idx"], ["region_per_box"], axis=0),
        make_constant("lo_start", np.array([0], dtype=np.int64)),
        make_constant("lo_end", np.array([2], dtype=np.int64)),
        make_constant("hi_end", np.array([4], dtype=np.int64)),
        helper.make_node("Slice", ["region_per_box", "lo_start", "lo_end", "axis_1"], ["region_lo"]),
        helper.make_node("Slice", ["region_per_box", "lo_end", "hi_end", "axis_1"], ["region_hi"]),
        helper.make_node("Slice", ["raw_boxes", "lo_start", "lo_end", "axis_1"], ["boxes_lo"]),
        helper.make_node("Slice", ["raw_boxes", "lo_end", "hi_end", "axis_1"], ["boxes_hi"]),
        # Clamp every coordinate into the region from *both* sides, which is what
        # `rescale_boxes` does on the PyTorch side. Clamping the lower corner only
        # from below leaves a box that sits entirely past the far edge with its
        # near corner outside, and any later "don't invert" fix-up then drags the
        # far corner back out with it. Clamping is monotonic, so lo <= hi still
        # holds afterwards and a fully-outside box collapses to zero area.
        helper.make_node("Max", ["boxes_lo", "region_lo"], ["lo_floor"]),
        helper.make_node("Min", ["lo_floor", "region_hi"], ["clip_lo"]),
        helper.make_node("Min", ["boxes_hi", "region_hi"], ["hi_ceil"]),
        helper.make_node("Max", ["hi_ceil", "region_lo"], ["clip_hi"]),
        helper.make_node("Concat", ["clip_lo", "clip_hi"], ["rois"], axis=1),
    ]

    # ------------------------------------------------------------------
    # ROI pooling, one branch per layer, concatenated
    # ------------------------------------------------------------------
    pooled_names = []
    embedding_width = 0
    for layer in layers:
        feature = f"feat_{layer}"
        channels, width = _feature_dims(graph, feature)
        embedding_width += channels
        pooled = f"pooled_{layer}"
        nodes += [
            helper.make_node(
                "RoiAlign",
                [feature, "rois", "batch_idx"],
                [f"grid_{layer}"],
                # `avg` regardless of `pooling`: FeaturePooler resamples the ROI
                # bilinearly and only then reduces the grid, so the max variant is
                # RoiAlign(avg) followed by a max over the 3x3, not RoiAlign(max).
                mode="avg",
                output_height=ROI_GRID,
                output_width=ROI_GRID,
                # torchvision's sampling_ratio=-1 (adaptive) is ONNX's 0.
                sampling_ratio=0,
                spatial_scale=float(width) / float(canvas),
                # torchvision's aligned=True.
                coordinate_transformation_mode="half_pixel",
            ),
            # Global pool over the grid rather than a Reduce op: opset 18 moved
            # `axes` from an attribute to an input, and this sidesteps it entirely.
            helper.make_node(
                "GlobalAveragePool" if pooling == "avg" else "GlobalMaxPool",
                [f"grid_{layer}"],
                [f"pool_{layer}"],
            ),
            helper.make_node("Flatten", [f"pool_{layer}"], [pooled], axis=1),
        ]
        pooled_names.append(pooled)

    if len(pooled_names) == 1:
        embedding_raw = pooled_names[0]
    else:
        embedding_raw = "object_embedding_raw"
        nodes.append(helper.make_node("Concat", pooled_names, [embedding_raw], axis=1))

    # ------------------------------------------------------------------
    # L2 normalize, matching F.normalize's max(norm, eps) rather than
    # LpNormalization, which turns a zero vector into NaN. A box that clips to
    # zero area samples nothing and stays a documented zero vector.
    # ------------------------------------------------------------------
    if normalize:
        nodes += [
            helper.make_node("Mul", [embedding_raw, embedding_raw], ["emb_sq"]),
            make_constant("reduce_axes", np.array([1], dtype=np.int64)),
            helper.make_node("ReduceSum", ["emb_sq", "reduce_axes"], ["emb_sumsq"], keepdims=1),
            helper.make_node("Sqrt", ["emb_sumsq"], ["emb_norm_raw"]),
            make_constant("norm_eps", np.array(_NORM_EPS, dtype=np.float32)),
            helper.make_node("Max", ["emb_norm_raw", "norm_eps"], ["emb_norm"]),
            helper.make_node("Div", [embedding_raw, "emb_norm"], ["object_embedding"]),
        ]
    else:
        nodes.append(helper.make_node("Identity", [embedding_raw], ["object_embedding"]))

    # ------------------------------------------------------------------
    # detections [D, 7] — the same layout the frigate target emits
    # ------------------------------------------------------------------
    nodes += [
        helper.make_node("Cast", ["batch_2d"], ["batch_float"], to=TensorProto.FLOAT),
        helper.make_node("Cast", ["class_2d"], ["class_float"], to=TensorProto.FLOAT),
        helper.make_node(
            "Concat",
            ["batch_float", "rois", "scores_2d", "class_float"],
            ["detections"],
            axis=1,
        ),
    ]

    # ------------------------------------------------------------------
    # Rebuild: keep the image-level embedding, drop the feature maps (they were
    # only ever an output so these nodes could reach them).
    # ------------------------------------------------------------------
    embedding_output = next(o for o in graph.output if o.name == "embedding")
    new_outputs = [
        helper.make_tensor_value_info("detections", TensorProto.FLOAT, [None, 7]),
        helper.make_tensor_value_info("object_embedding", TensorProto.FLOAT, [None, embedding_width]),
        embedding_output,
    ]

    new_graph = helper.make_graph(
        list(graph.node) + nodes,
        graph.name,
        list(graph.input),
        new_outputs,
        initializer=list(graph.initializer),
    )

    new_model = helper.make_model(
        new_graph,
        opset_imports=model.opset_import,
        # The dynamo exporter can emit local functions; make_model drops them
        # unless they are carried over explicitly.
        functions=list(model.functions),
    )
    new_model.ir_version = model.ir_version

    onnx.checker.check_model(new_model)
    onnx.save(new_model, output_path)
