"""Image and object embeddings from the YOLO-NAS backbone and neck.

The detector throws away everything except boxes and scores. The feature maps it
throws away are a perfectly good visual representation, and this module exposes
them as fixed-length vectors for image retrieval, near-duplicate search,
clustering, dataset exploration and re-identification.

Two levels of API:

* :meth:`modern_yolonas.YoloNAS.forward_features` — raw feature maps, if you want
  to pool them yourself.
* :class:`YoloNASEmbedder` — image in, ``np.ndarray`` out, with the letterbox
  padding handled correctly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from modern_yolonas.inference.preprocess import preprocess
from modern_yolonas.validation import validate_device, validate_input_size, validate_model_name

#: Feature maps :meth:`YoloNAS.forward_features` returns, coarse to fine.
FEATURE_LAYERS = ("c2", "c3", "c4", "c5", "p3", "p4", "p5")

#: Grid an ROI is resampled to before pooling, in :meth:`YoloNASEmbedder.embed_boxes`.
#: 3x3 keeps a little coarse spatial layout inside the box without multiplying the
#: embedding width, which a flattened grid would.
_ROI_GRID = 3


class YoloNASEmbedder:
    """Turn images (or boxes within them) into fixed-length feature vectors.

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

        layers = tuple(layers)
        if not layers:
            raise ValueError("layers must name at least one feature map")
        unknown = [name for name in layers if name not in FEATURE_LAYERS]
        if unknown:
            raise ValueError(f"unknown layer(s) {unknown}; choose from {list(FEATURE_LAYERS)}")

        if pooling not in ("avg", "max"):
            raise ValueError(f"pooling must be 'avg' or 'max', got {pooling!r}")
        if precision not in ("fp32", "fp16"):
            raise ValueError(f"precision must be 'fp32' or 'fp16', got {precision!r}")

        builders = {"yolo_nas_s": yolo_nas_s, "yolo_nas_m": yolo_nas_m, "yolo_nas_l": yolo_nas_l}

        self.device = validate_device(device)
        self.input_size = input_size
        self.layers = layers
        self.pooling = pooling
        self.normalize = normalize
        self.precision = precision

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
                features = self.model.forward_features(dummy)
            self._dim = sum(features[name].shape[1] for name in self.layers)
        return self._dim

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _read(self, source: str | Path | np.ndarray) -> np.ndarray:
        import cv2

        if isinstance(source, (str, Path)):
            image = cv2.imread(str(source))
            if image is None:
                raise FileNotFoundError(f"Cannot read image: {source}")
            return image
        return source

    def _pool_valid(self, feature: Tensor, valid: tuple[int, int, int, int]) -> Tensor:
        """Pool one sample's feature map over the non-padded region only.

        ``preprocess`` pads the image out to a square with gray (114), and on a
        16:9 frame that padding is most of the canvas. Pooling over the whole map
        averages the gray in and drags every embedding toward a common vector, so
        the region the real image occupies is cropped out first. ``valid`` is
        ``(left, top, right, bottom)`` in input pixels; it is mapped to feature
        coordinates by the stride this level happens to have.
        """
        _, height, width = feature.shape
        stride = self.input_size / width
        left, top, right, bottom = valid

        x0 = int(np.floor(left / stride))
        y0 = int(np.floor(top / stride))
        x1 = int(np.ceil(right / stride))
        y1 = int(np.ceil(bottom / stride))

        # Clamp, and never let rounding produce an empty crop on a tiny feature map.
        x0, y0 = max(0, min(x0, width - 1)), max(0, min(y0, height - 1))
        x1, y1 = min(width, max(x1, x0 + 1)), min(height, max(y1, y0 + 1))

        region = feature[:, y0:y1, x0:x1]
        flat = region.reshape(region.shape[0], -1)
        return flat.mean(dim=1) if self.pooling == "avg" else flat.amax(dim=1)

    def _finalize(self, vectors: Tensor) -> np.ndarray:
        if self.normalize:
            vectors = torch.nn.functional.normalize(vectors.float(), p=2, dim=-1)
        return vectors.float().cpu().numpy()

    def _valid_region(self, image: np.ndarray, scale: float, pad: tuple[int, int]) -> tuple[int, int, int, int]:
        """Where the real pixels live inside the letterboxed canvas."""
        height, width = image.shape[:2]
        left, top = pad
        # Mirrors ``letterbox``'s own rounding, so the crop lines up exactly.
        return left, top, left + int(round(width * scale)), top + int(round(height * scale))

    def _forward(self, tensor: Tensor) -> dict[str, Tensor]:
        tensor = tensor.to(self.device)
        if self.precision == "fp16":
            tensor = tensor.half()
        with torch.amp.autocast("cuda", enabled=self.precision == "fp16"):
            return self.model.forward_features(tensor)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
            regions.append(self._valid_region(image, scale, pad))

        features = self._forward(torch.cat(tensors, dim=0))

        # Each image has its own padding, so pooling is per sample rather than a
        # single batched mean over the whole map.
        pooled = [
            torch.cat([self._pool_valid(features[name][i], regions[i]) for name in self.layers])
            for i in range(len(sources))
        ]
        return self._finalize(torch.stack(pooled))

    @torch.no_grad()
    def embed_boxes(self, source: str | Path | np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        """Embed regions of one image — one vector per box.

        This is the per-object counterpart of :meth:`__call__`, for recognition,
        re-identification and "find me more of this thing" search. The boxes are
        cropped out of the feature maps with ``roi_align`` rather than out of the
        image, so a whole frame's worth of objects costs a single forward pass.

        Args:
            source: File path or BGR numpy array.
            xyxy: ``(K, 4)`` boxes in the original image's pixel coordinates —
                ``supervision.Detections.xyxy`` as it comes.

        Returns:
            ``(K, embedding_dim)`` float32, in box order.
        """
        from torchvision.ops import roi_align

        xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        if len(xyxy) == 0:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        image = self._read(source)
        tensor, scale, pad = preprocess(image, self.input_size)
        features = self._forward(tensor)

        # Original-image pixels → letterboxed canvas, the geometry the features are in.
        left, top = pad
        boxes = torch.from_numpy(xyxy.copy())
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + left
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + top
        rois = torch.cat([torch.zeros(len(boxes), 1), boxes], dim=1).to(self.device)

        pooled = []
        for name in self.layers:
            feature = features[name].float()
            stride = self.input_size / feature.shape[-1]
            grid = roi_align(
                feature,
                rois.float(),
                output_size=(_ROI_GRID, _ROI_GRID),
                spatial_scale=1.0 / stride,
                sampling_ratio=-1,
                aligned=True,
            )
            flat = grid.reshape(grid.shape[0], grid.shape[1], -1)
            pooled.append(flat.mean(dim=2) if self.pooling == "avg" else flat.amax(dim=2))

        return self._finalize(torch.cat(pooled, dim=1))
