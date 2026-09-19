"""Image and object embeddings from the YOLO-NAS backbone and neck.

The detector throws away everything except boxes and scores. The feature maps it
throws away are a perfectly good visual representation, and this module exposes
them as fixed-length vectors for image retrieval, near-duplicate search,
clustering, dataset exploration and re-identification.

Three levels of API:

* :meth:`modern_yolonas.YoloNAS.forward_features` — raw feature maps, if you want
  to pool them yourself.
* :class:`YoloNASEmbedder` — image in, ``np.ndarray`` out, with the letterbox
  padding handled correctly. Embeddings only; the detection head never runs.
* :meth:`~modern_yolonas.inference.detect.YoloNASDetector.predict` with
  :class:`Task` flags — boxes *and* embeddings from a single forward pass.
"""

from __future__ import annotations

import enum

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import supervision as sv
import torch

from torch import Tensor

from modern_yolonas.inference.preprocess import preprocess
from modern_yolonas.validation import validate_device, validate_input_size, validate_model_name

#: Feature maps :meth:`YoloNAS.forward_features` returns, fine to coarse.
FEATURE_LAYERS = ("c2", "c3", "c4", "c5", "p3", "p4", "p5")

#: Grid an ROI is resampled to before pooling. 3x3 keeps a little coarse spatial
#: layout inside the box without multiplying the embedding width, which a
#: flattened grid would.
_ROI_GRID = 3


class Task(enum.Flag):
    """What a :meth:`YoloNASDetector.predict` call should produce.

    Combine with ``|``. The backbone and neck run exactly once no matter how many
    are asked for, which is the point — detection and embedding share everything
    up to the head, so getting both costs one pass rather than two::

        from modern_yolonas import Task, YoloNASDetector

        detector = YoloNASDetector("yolo_nas_s")
        result = detector.predict(image, Task.DETECT | Task.EMBED_OBJECTS)

        result.detections                          # sv.Detections
        result.detections.data["embedding"]        # (N, 768), one row per detection

    Members:
        DETECT: Boxes, scores and class ids, as ``result.detections``.
        EMBED: One vector for the whole image, as ``result.embedding``.
        EMBED_OBJECTS: One vector per detection, in
            ``result.detections.data["embedding"]``. Implies ``DETECT``, since
            the boxes are what gets embedded.
    """

    DETECT = enum.auto()
    EMBED = enum.auto()
    EMBED_OBJECTS = enum.auto()


@dataclass
class Prediction:
    """What :meth:`YoloNASDetector.predict` returns.

    Fields not asked for are ``None``, so a ``Task.EMBED``-only call does not have
    to invent an empty ``Detections`` to carry the vector.

    Attributes:
        detections: Present for :attr:`Task.DETECT` and :attr:`Task.EMBED_OBJECTS`.
            With ``EMBED_OBJECTS``, ``detections.data["embedding"]`` holds an
            ``(N, D)`` array whose rows stay aligned with the boxes through
            slicing — ``result.detections[mask]`` filters both together.
        embedding: Present for :attr:`Task.EMBED`. ``(D,)`` float32.
    """

    detections: sv.Detections | None = None
    embedding: np.ndarray | None = None


class FeaturePooler:
    """Turns feature maps into fixed-length vectors.

    Split out from :class:`YoloNASEmbedder` so the detector can reuse the exact
    same pooling on the features it already computed, rather than growing a second
    copy of it that could drift.

    Args:
        layers: Which feature maps to pool and concatenate, from
            :data:`FEATURE_LAYERS`. The default ``("c5",)`` is the deepest
            backbone map, after SPP: the most semantic representation the network
            has, and the one least entangled with box regression. It is 768
            channels on S, M and L alike, so vectors from different variants have
            the same width (they are *not* interchangeable — only same-width).
            The neck's ``p3``–``p5`` are available, but they are tuned to localize
            rather than to describe, so they make weaker retrieval keys.
        pooling: ``"avg"`` or ``"max"`` over the spatial dimensions.
        normalize: L2-normalize the result, which makes a dot product a cosine
            similarity. Leave it on unless you need raw activation magnitude.
    """

    def __init__(
        self,
        layers: tuple[str, ...] | list[str] = ("c5",),
        pooling: str = "avg",
        normalize: bool = True,
    ):
        layers = tuple(layers)
        if not layers:
            raise ValueError("layers must name at least one feature map")
        unknown = [name for name in layers if name not in FEATURE_LAYERS]
        if unknown:
            raise ValueError(f"unknown layer(s) {unknown}; choose from {list(FEATURE_LAYERS)}")
        if pooling not in ("avg", "max"):
            raise ValueError(f"pooling must be 'avg' or 'max', got {pooling!r}")

        self.layers = layers
        self.pooling = pooling
        self.normalize = normalize

    def dim(self, features: dict[str, Tensor]) -> int:
        """Width of the vector these feature maps will pool down to."""
        return sum(features[name].shape[1] for name in self.layers)

    def _reduce(self, flat: Tensor, dim: int) -> Tensor:
        return flat.mean(dim=dim) if self.pooling == "avg" else flat.amax(dim=dim)

    def _pool_valid(self, feature: Tensor, valid: tuple[int, int, int, int], canvas: int) -> Tensor:
        """Pool one sample's feature map over the non-padded region only.

        ``preprocess`` pads the image out to a square with gray (114), and on a
        16:9 frame that padding is most of the canvas. Pooling over the whole map
        averages the gray in and makes embeddings cluster by aspect ratio rather
        than by content — measured with the COCO ``yolo_nas_s`` weights,
        full-canvas pooling scores an unrelated noise image against a street photo
        at 0.958 cosine, *higher* than that photo against a second real photo,
        purely because both carry the same padding. Cropping to the valid region
        first puts the pair at 0.396.

        Args:
            feature: ``[C, H, W]`` map for one sample.
            valid: ``(left, top, right, bottom)`` in canvas pixels.
            canvas: Side length of the letterboxed input.
        """
        _, height, width = feature.shape
        stride = canvas / width
        left, top, right, bottom = valid

        x0 = int(np.floor(left / stride))
        y0 = int(np.floor(top / stride))
        x1 = int(np.ceil(right / stride))
        y1 = int(np.ceil(bottom / stride))

        # Clamp, and never let rounding produce an empty crop on a tiny feature map.
        x0, y0 = max(0, min(x0, width - 1)), max(0, min(y0, height - 1))
        x1, y1 = min(width, max(x1, x0 + 1)), min(height, max(y1, y0 + 1))

        region = feature[:, y0:y1, x0:x1]
        return self._reduce(region.reshape(region.shape[0], -1), dim=1)

    def pool_images(
        self,
        features: dict[str, Tensor],
        regions: list[tuple[int, int, int, int]],
        canvas: int,
    ) -> Tensor:
        """One vector per sample in the batch, padding excluded.

        Args:
            features: Output of :meth:`YoloNAS.forward_features`.
            regions: Per sample ``(left, top, right, bottom)`` in canvas pixels.
            canvas: Side length of the letterboxed input.

        Returns:
            ``[B, D]``.
        """
        # Each image has its own padding, so pooling is per sample rather than a
        # single batched mean over the whole map.
        return torch.stack(
            [
                torch.cat([self._pool_valid(features[name][i], region, canvas) for name in self.layers])
                for i, region in enumerate(regions)
            ]
        )

    def pool_images_masked(self, features: dict[str, Tensor], regions: Tensor, canvas: int) -> Tensor:
        """:meth:`pool_images`, written so it can be traced and exported.

        Same result, different mechanics. :meth:`pool_images` slices each sample's
        map with Python ``int`` bounds, which is data-dependent control flow that
        ``torch.export`` cannot see through. This builds a boolean mask from the
        region tensor and reduces over it instead, so every step is an op on
        tensors and the whole thing survives ONNX export.

        The bounds use integer arithmetic — ``(left * width) // canvas`` rather
        than ``floor(left / stride)`` — because float division at exact multiples
        is where the two implementations would drift apart. ``test_embed.py`` pins
        them to each other.

        Args:
            features: Output of :meth:`YoloNAS.forward_features`.
            regions: ``[B, 4]`` int64 ``(left, top, right, bottom)`` in canvas
                pixels — :func:`valid_region` per sample, stacked.
            canvas: Side length of the letterboxed input.

        Returns:
            ``[B, D]``.
        """
        pooled = []
        for name in self.layers:
            feature = features[name]
            height, width = feature.shape[-2], feature.shape[-1]
            mask = self._region_mask(regions, height, width, canvas, feature.device)

            if self.pooling == "avg":
                total = (feature * mask.unsqueeze(1)).sum(dim=(2, 3))
                count = mask.sum(dim=(1, 2)).unsqueeze(1).to(feature.dtype)
                pooled.append(total / count)
            else:
                floor = torch.finfo(feature.dtype).min
                pooled.append(feature.masked_fill(~mask.unsqueeze(1), floor).amax(dim=(2, 3)))
        return torch.cat(pooled, dim=1)

    @staticmethod
    def _region_mask(regions: Tensor, height: int, width: int, canvas: int, device) -> Tensor:
        """``[B, H, W]`` true where a sample's real pixels land on this level.

        Mirrors the bounds :meth:`_pool_valid` slices with, including its clamps —
        which exist so rounding can never produce an empty crop on a coarse map.
        """
        regions = regions.to(device=device, dtype=torch.long)
        left, top, right, bottom = (regions[:, i : i + 1] for i in range(4))

        # floor(coord / stride) and ceil(coord / stride), in exact integer terms.
        x0 = (left * width) // canvas
        y0 = (top * height) // canvas
        x1 = -((-right * width) // canvas)
        y1 = -((-bottom * height) // canvas)

        x0 = x0.clamp(0, width - 1)
        y0 = y0.clamp(0, height - 1)
        x1 = torch.maximum(x1, x0 + 1).clamp(max=width)
        y1 = torch.maximum(y1, y0 + 1).clamp(max=height)

        cols = torch.arange(width, device=device).unsqueeze(0)
        rows = torch.arange(height, device=device).unsqueeze(0)
        mask_x = (cols >= x0) & (cols < x1)
        mask_y = (rows >= y0) & (rows < y1)
        return mask_y.unsqueeze(2) & mask_x.unsqueeze(1)

    def pool_rois(self, features: dict[str, Tensor], rois: Tensor, canvas: int) -> Tensor:
        """One vector per box, cropped out of the feature maps with ``roi_align``.

        Args:
            features: Output of :meth:`YoloNAS.forward_features`.
            rois: ``[K, 5]`` as ``(batch_index, x1, y1, x2, y2)`` in **canvas**
                coordinates — the letterboxed geometry the features live in.
            canvas: Side length of the letterboxed input.

        Returns:
            ``[K, D]``.
        """
        from torchvision.ops import roi_align

        pooled = []
        for name in self.layers:
            feature = features[name].float()
            stride = canvas / feature.shape[-1]
            grid = roi_align(
                feature,
                rois.float(),
                output_size=(_ROI_GRID, _ROI_GRID),
                spatial_scale=1.0 / stride,
                sampling_ratio=-1,
                aligned=True,
            )
            pooled.append(self._reduce(grid.reshape(grid.shape[0], grid.shape[1], -1), dim=2))
        return torch.cat(pooled, dim=1)

    def finalize(self, vectors: Tensor) -> np.ndarray:
        """L2-normalize if configured to, and hand back float32 numpy."""
        if self.normalize:
            vectors = torch.nn.functional.normalize(vectors.float(), p=2, dim=-1)
        return vectors.float().cpu().numpy()


def valid_region(image: np.ndarray, scale: float, pad: tuple[int, int]) -> tuple[int, int, int, int]:
    """Where an image's real pixels land inside the letterboxed canvas.

    Args:
        image: The original BGR array, for its shape.
        scale: Second value from :func:`preprocess`.
        pad: Third value from :func:`preprocess`, ``(left, top)``.

    Returns:
        ``(left, top, right, bottom)`` in canvas pixels.
    """
    height, width = image.shape[:2]
    left, top = pad
    # Mirrors ``letterbox``'s own rounding, so the crop lines up exactly.
    return left, top, left + int(round(width * scale)), top + int(round(height * scale))


class YoloNASEmbedder:
    """Turn images (or boxes within them) into fixed-length feature vectors.

    The detection head never runs, so this is the cheapest path when embeddings
    are all you want. If you need boxes *and* embeddings, use
    :meth:`~modern_yolonas.inference.detect.YoloNASDetector.predict` with
    :class:`Task` flags instead — it gets both from one forward pass.

    Usage::

        from modern_yolonas import YoloNASEmbedder

        embedder = YoloNASEmbedder("yolo_nas_s", device="cuda")

        vec = embedder("image.jpg")              # (768,) L2-normalized
        bank = embedder.embed_batch(paths)       # (N, 768)
        similarity = bank @ vec                  # cosine, because both are normalized

        # Per-object vectors, straight from a detection result
        detections = YoloNASDetector("yolo_nas_s")(image)
        crops = embedder.embed_boxes(image, detections.xyxy)   # (len(detections), 768)

    Args:
        model: Variant name — ``yolo_nas_s``, ``yolo_nas_m`` or ``yolo_nas_l``.
        device: Torch device.
        input_size: Square letterbox size the image is resized into.
        layers: See :class:`FeaturePooler`.
        pooling: See :class:`FeaturePooler`.
        normalize: See :class:`FeaturePooler`.
        pretrained: Load the COCO checkpoint.
        precision: ``"fp32"`` or ``"fp16"``.
        weights: Optional checkpoint path, as in
            :class:`~modern_yolonas.inference.detect.YoloNASDetector`.
        num_classes: Head width of that checkpoint. It does not affect the
            embedding — the head is never run — but the architecture has to be
            built to match before the weights will load.
    """

    def __init__(
        self,
        model: str = "yolo_nas_s",
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
        input_size: int = 640,
        layers: tuple[str, ...] | list[str] = ("c5",),
        pooling: str = "avg",
        normalize: bool = True,
        pretrained: bool = True,
        precision: str = "fp32",
        weights: str | Path | None = None,
        num_classes: int = 80,
    ):
        from modern_yolonas import yolo_nas_s, yolo_nas_m, yolo_nas_l

        validate_model_name(model)
        validate_input_size(input_size)

        if precision not in ("fp32", "fp16"):
            raise ValueError(f"precision must be 'fp32' or 'fp16', got {precision!r}")

        builders = {"yolo_nas_s": yolo_nas_s, "yolo_nas_m": yolo_nas_m, "yolo_nas_l": yolo_nas_l}

        self.device = validate_device(device)
        self.input_size = input_size
        self.precision = precision
        self.pooler = FeaturePooler(layers=layers, pooling=pooling, normalize=normalize)

        if weights is not None:
            from modern_yolonas.weights import extract_model_state_dict

            self.model = builders[model](pretrained=False, num_classes=num_classes).to(self.device)
            self.model.load_state_dict(extract_model_state_dict(weights, map_location=str(self.device)))
        else:
            self.model = builders[model](pretrained=pretrained, num_classes=num_classes).to(self.device)

        if precision == "fp16":
            self.model = self.model.half()
        self.model.eval()

        self._dim: int | None = None

    @property
    def layers(self) -> tuple[str, ...]:
        """Feature maps being pooled."""
        return self.pooler.layers

    @property
    def embedding_dim(self) -> int:
        """Width of the vectors this embedder produces.

        Measured by a single forward pass on a blank image the first time it is
        asked, then cached — the channel counts are config-derived and there is no
        cheaper way to get them that cannot drift from what the model actually does.
        """
        if self._dim is None:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, self.input_size, self.input_size, device=self.device)
                if self.precision == "fp16":
                    dummy = dummy.half()
                self._dim = self.pooler.dim(self.model.forward_features(dummy))
        return self._dim

    def _read(self, source: str | Path | np.ndarray) -> np.ndarray:
        import cv2

        if isinstance(source, (str, Path)):
            image = cv2.imread(str(source))
            if image is None:
                raise FileNotFoundError(f"Cannot read image: {source}")
            return image
        return source

    def _forward(self, tensor: Tensor) -> dict[str, Tensor]:
        tensor = tensor.to(self.device)
        if self.precision == "fp16":
            tensor = tensor.half()
        with torch.amp.autocast("cuda", enabled=self.precision == "fp16"):
            return self.model.forward_features(tensor)

    @torch.no_grad()
    def __call__(self, source: str | Path | np.ndarray) -> np.ndarray:
        """Embed a single image.

        Args:
            source: File path or BGR numpy array.

        Returns:
            ``(embedding_dim,)`` float32.
        """
        return self.embed_batch([source])[0]

    @torch.no_grad()
    def embed_batch(self, sources: list[str | Path | np.ndarray]) -> np.ndarray:
        """Embed several images in one forward pass.

        Args:
            sources: File paths or BGR numpy arrays.

        Returns:
            ``(len(sources), embedding_dim)`` float32, in input order.
        """
        if not sources:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        tensors = []
        regions = []
        for source in sources:
            image = self._read(source)
            tensor, scale, pad = preprocess(image, self.input_size)
            tensors.append(tensor)
            regions.append(valid_region(image, scale, pad))

        features = self._forward(torch.cat(tensors, dim=0))
        return self.pooler.finalize(self.pooler.pool_images(features, regions, self.input_size))

    @torch.no_grad()
    def embed_boxes(self, source: str | Path | np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        """Embed regions of one image — one vector per box.

        This is the per-object counterpart of :meth:`__call__`, for recognition,
        re-identification and "find me more of this thing" search. The boxes are
        cropped out of the feature maps with ``roi_align`` rather than out of the
        image, so a whole frame's worth of objects costs a single forward pass.

        When the boxes come from this project's own detector, prefer
        :meth:`YoloNASDetector.predict` with :attr:`Task.EMBED_OBJECTS` — it
        reuses the detection pass instead of running a second one.

        Args:
            source: File path or BGR numpy array.
            xyxy: ``(K, 4)`` boxes in the original image's pixel coordinates —
                ``supervision.Detections.xyxy`` as it comes.

        Returns:
            ``(K, embedding_dim)`` float32, in box order. Boxes are clipped to the
            image first; one that clips to zero area samples nothing and comes back
            as a zero vector rather than a NaN.
        """
        xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        if len(xyxy) == 0:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        image = self._read(source)
        tensor, scale, pad = preprocess(image, self.input_size)

        # Clip to the frame. Outside it there is only letterbox padding, so an
        # unclipped box would be described partly by gray. This also makes these
        # vectors identical to the ones `YoloNASDetector.predict` produces, whose
        # boxes arrive already clipped by `rescale_boxes`.
        height, width = image.shape[:2]
        xyxy = xyxy.copy()
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, width)
        xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, height)
        features = self._forward(tensor)

        rois = boxes_to_rois(xyxy, scale, pad).to(self.device)
        return self.pooler.finalize(self.pooler.pool_rois(features, rois, self.input_size))


def boxes_to_rois(xyxy: np.ndarray, scale: float, pad: tuple[int, int], batch_index: int = 0) -> Tensor:
    """Map original-image boxes onto the letterboxed canvas as ``roi_align`` ROIs.

    Args:
        xyxy: ``(K, 4)`` in the original image's pixel coordinates.
        scale: Second value from :func:`preprocess`.
        pad: Third value from :func:`preprocess`, ``(left, top)``.
        batch_index: Which sample of the batch these boxes belong to.

    Returns:
        ``[K, 5]`` as ``(batch_index, x1, y1, x2, y2)`` in canvas coordinates.
    """
    left, top = pad
    boxes = torch.from_numpy(np.asarray(xyxy, dtype=np.float32).reshape(-1, 4).copy())
    boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + left
    boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + top
    return torch.cat([torch.full((len(boxes), 1), float(batch_index)), boxes], dim=1)
