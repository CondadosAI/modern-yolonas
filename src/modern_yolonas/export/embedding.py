"""Export graphs that emit feature embeddings, alone or beside detections.

The wrappers here exist because :class:`~modern_yolonas.inference.embed.YoloNASEmbedder`
is a Python object that reads images, letterboxes them and pools with Python
integers — none of which a traced graph can see. These are plain ``nn.Module``
forwards over tensors, which is what ``torch.onnx.export`` can follow.

**The second input.** Every graph here takes ``valid_region`` alongside ``images``:
an ``[B, 4]`` int64 tensor of ``(left, top, right, bottom)`` in canvas pixels,
saying where each image's real pixels sit inside the letterbox. It cannot be baked
into the graph because it depends on the aspect ratio of the image being embedded,
and it cannot be dropped: pooling the gray padding in makes embeddings cluster by
aspect ratio rather than by content. Get it from
:func:`~modern_yolonas.inference.embed.valid_region`, which takes what
:func:`~modern_yolonas.inference.preprocess.preprocess` already returns::

    tensor, scale, pad = preprocess(image, 640)
    region = valid_region(image, scale, pad)          # (left, top, right, bottom)
    session.run(None, {"images": tensor.numpy(), "valid_region": np.array([region])})

To embed the whole canvas anyway, pass ``(0, 0, canvas, canvas)`` — but that is the
behaviour the region input exists to avoid.
"""

from __future__ import annotations

from torch import Tensor, nn

from modern_yolonas.inference.embed import FeaturePooler


class EmbeddingGraph(nn.Module):
    """Backbone and neck to a pooled embedding. The detection head never runs.

    Args:
        model: A :class:`~modern_yolonas.model.YoloNAS`, already fused and in
            eval mode.
        pooler: How the feature maps become a vector.
        canvas: Letterbox side length the graph is exported at. Baked in, because
            only the batch dimension is dynamic.

    Forward:
        ``(images [B, 3, canvas, canvas], valid_region [B, 4])`` →
        ``embedding [B, D]``.
    """

    def __init__(self, model: nn.Module, pooler: FeaturePooler, canvas: int):
        super().__init__()
        self.model = model
        self.pooler = pooler
        self.canvas = canvas

    def forward(self, images: Tensor, valid_region: Tensor) -> Tensor:
        features = self.model.forward_features(images)
        embedding = self.pooler.pool_images_masked(features, valid_region, self.canvas)
        if self.pooler.normalize:
            embedding = nn.functional.normalize(embedding, p=2, dim=-1)
        return embedding


class DetectAndEmbedGraph(nn.Module):
    """Detections and an embedding from one pass, as one graph.

    The ONNX counterpart of
    :meth:`YoloNASDetector.predict(..., Task.DETECT | Task.EMBED)
    <modern_yolonas.inference.detect.YoloNASDetector.predict>`: the backbone and
    neck are evaluated once and both heads of the result read off them, rather
    than deploying two models over the same frames.

    Args:
        model: A :class:`~modern_yolonas.model.YoloNAS`, already fused and in
            eval mode.
        pooler: How the feature maps become a vector.
        canvas: Letterbox side length the graph is exported at.

    Forward:
        ``(images [B, 3, canvas, canvas], valid_region [B, 4])`` →
        ``(pred_bboxes [B, A, 4], pred_scores [B, A, C], embedding [B, D])``.

    Note:
        ``pred_bboxes`` are in canvas coordinates and NMS has not run — the same
        contract as the plain detection export, so the existing postprocessing
        applies unchanged.
    """

    def __init__(self, model: nn.Module, pooler: FeaturePooler, canvas: int):
        super().__init__()
        self.model = model
        self.pooler = pooler
        self.canvas = canvas

    def forward(self, images: Tensor, valid_region: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        features = self.model.forward_features(images)
        pred_bboxes, pred_scores = self.model.heads(
            (features["p3"], features["p4"], features["p5"])
        )
        embedding = self.pooler.pool_images_masked(features, valid_region, self.canvas)
        if self.pooler.normalize:
            embedding = nn.functional.normalize(embedding, p=2, dim=-1)
        return pred_bboxes, pred_scores, embedding


class DetectAndFeatureGraph(nn.Module):
    """Detections, an image embedding, and the raw feature maps behind them.

    Not a deployment artifact on its own — it is the base graph that
    :func:`~modern_yolonas.export.objects.make_object_embedding_onnx` performs
    surgery on. The feature maps have to leave the graph as outputs so the
    inserted ``RoiAlign`` nodes have something to reference; the surgery then
    consumes them and they do not appear in the final model.

    Args:
        model: A :class:`~modern_yolonas.model.YoloNAS`, already fused and in
            eval mode.
        pooler: How the feature maps become the image-level vector, and which
            maps the object-level pooling will read.
        canvas: Letterbox side length the graph is exported at.

    Forward:
        ``(images [B, 3, canvas, canvas], valid_region [B, 4])`` →
        ``(pred_bboxes, pred_scores, embedding, *feature maps in pooler order)``.
    """

    def __init__(self, model: nn.Module, pooler: FeaturePooler, canvas: int):
        super().__init__()
        self.model = model
        self.pooler = pooler
        self.canvas = canvas

    @property
    def output_names(self) -> list[str]:
        """Names to export with, in forward order."""
        return ["pred_bboxes", "pred_scores", "embedding"] + [f"feat_{n}" for n in self.pooler.layers]

    def forward(self, images: Tensor, valid_region: Tensor) -> tuple[Tensor, ...]:
        features = self.model.forward_features(images)
        pred_bboxes, pred_scores = self.model.heads(
            (features["p3"], features["p4"], features["p5"])
        )
        embedding = self.pooler.pool_images_masked(features, valid_region, self.canvas)
        if self.pooler.normalize:
            embedding = nn.functional.normalize(embedding, p=2, dim=-1)
        return (pred_bboxes, pred_scores, embedding, *(features[n] for n in self.pooler.layers))
