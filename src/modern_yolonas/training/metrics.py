"""COCO-style detection metrics.

Two implementations, for two callers. :class:`DetectionMetrics` wraps torchmetrics and
works from tensors alone, which suits a validation loop with no annotation file.
:class:`COCOEvaluator` runs the reference `pycocotools` evaluation against the dataset's
own annotation JSON — the number that can be published and compared against other
papers' COCO results.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch
from torch import Tensor


class DetectionMetrics:
    """COCO-style object-detection metrics backed by :class:`torchmetrics.detection.MeanAveragePrecision`.

    Computes mAP @IoU[0.50:0.95], mAP@0.50, mAP@0.75, per-size mAP, and maximum
    average recall (mAR) for one, ten, and one-hundred max-detections per image —
    i.e. the same set reported by the COCO benchmark.

    Args:
        class_metrics: When ``True``, also produce per-class AP/AR in addition to
            the global averages.
        extended_summary: Forward the ``extended_summary`` flag to torchmetrics for
            extra diagnostic statistics.
        device: Device on which internal accumulation tensors are kept.  Use the
            training device to avoid cross-device copies during validation.

    Example::

        metrics = DetectionMetrics(device="cuda")
        for images, targets in val_loader:
            preds = decode_predictions(model(images))   # list of dicts
            gts   = convert_targets(targets)            # list of dicts
            metrics.update(preds, gts)

        scores = metrics.compute()
        # {"mAP": 0.423, "mAP_50": 0.612, "mAP_75": 0.448, ...}
        metrics.reset()
    """

    def __init__(
        self,
        class_metrics: bool = False,
        extended_summary: bool = False,
        device: str | torch.device = "cpu",
    ) -> None:
        from torchmetrics.detection import MeanAveragePrecision

        self._metric = MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox",
            class_metrics=class_metrics,
            extended_summary=extended_summary,
        ).to(torch.device(device))

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, preds: list[dict], targets: list[dict]) -> None:
        """Accumulate predictions and ground-truth for one batch.

        Args:
            preds: One dict per image, each containing:

                - ``"boxes"``: ``FloatTensor[N, 4]`` in *xyxy* pixel coords
                  (x1, y1, x2, y2).
                - ``"scores"``: ``FloatTensor[N]`` confidence values in ``[0, 1]``.
                - ``"labels"``: ``IntTensor[N]`` zero-based class indices.

            targets: One dict per image, each containing:

                - ``"boxes"``: ``FloatTensor[M, 4]`` in *xyxy* pixel coords.
                - ``"labels"``: ``IntTensor[M]`` zero-based class indices.
        """
        self._metric.update(preds, targets)

    def compute(self) -> dict[str, float]:
        """Compute metrics over all accumulated predictions and return a flat dict.

        Returns:
            A ``dict[str, float]`` with three keys:

            ``mAP``
                Mean AP averaged over IoU thresholds 0.50–0.95 (primary COCO metric).
            ``mAP_50``
                Mean AP at IoU = 0.50 (PASCAL VOC-style, a.k.a. the IoU metric).
            ``mAR_100``
                Max recall given at most 100 detections per image.
        """
        raw = self._metric.compute()

        def _scalar(key: str) -> float:
            v = raw.get(key, torch.tensor(0.0))
            return float(v.item() if isinstance(v, Tensor) else v)

        return {
            "mAP":    _scalar("map"),
            "mAP_50": _scalar("map_50"),
            "mAR_100": _scalar("mar_100"),
        }

    def reset(self) -> None:
        """Reset all accumulated state, ready for the next evaluation round."""
        self._metric.reset()


class COCOEvaluator:
    """Wrapper around pycocotools for mAP computation.

    Args:
        ann_file: Path to COCO annotations JSON.
        input_size: The square size the model was fed. Predictions come back in that
            letterboxed space while the ground truth is in original image pixels, so
            they have to be mapped back before they can be compared. Leave it ``None``
            only when the caller has already rescaled them.
    """

    def __init__(self, ann_file: str | Path, input_size: int | None = None):
        from pycocotools.coco import COCO

        self.coco_gt = COCO(str(ann_file))
        self.input_size = input_size
        self.results: list[dict] = []

        # Build reverse mapping: label → category_id
        cat_ids = sorted(self.coco_gt.getCatIds())
        self.label_to_cat_id = {i: cat_id for i, cat_id in enumerate(cat_ids)}

    def _letterbox_params(self, image_id: int) -> tuple[float, float, float]:
        """``(scale, pad_left, pad_top)`` LetterboxResize used for this image."""
        info = self.coco_gt.imgs[int(image_id)]
        h, w = info["height"], info["width"]
        scale = self.input_size / max(h, w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        return scale, (self.input_size - new_w) // 2, (self.input_size - new_h) // 2

    def reset(self):
        self.results = []

    def update(
        self,
        image_ids: list[int],
        boxes: list[Tensor],
        scores: list[Tensor],
        class_ids: list[Tensor],
    ):
        """Add batch of predictions.

        Args:
            image_ids: COCO image IDs.
            boxes: List of ``[D, 4]`` tensors (x1y1x2y2 pixel coords).
            scores: List of ``[D]`` confidence tensors.
            class_ids: List of ``[D]`` integer class ID tensors.
        """
        for img_id, bxs, scs, cids in zip(image_ids, boxes, scores, class_ids):
            if self.input_size is not None:
                scale, pad_left, pad_top = self._letterbox_params(img_id)
                info = self.coco_gt.imgs[int(img_id)]
                max_x, max_y = info["width"], info["height"]
            for box, score, cls_id in zip(bxs, scs, cids):
                x1, y1, x2, y2 = box.tolist()
                if self.input_size is not None:
                    x1 = min(max((x1 - pad_left) / scale, 0.0), max_x)
                    y1 = min(max((y1 - pad_top) / scale, 0.0), max_y)
                    x2 = min(max((x2 - pad_left) / scale, 0.0), max_x)
                    y2 = min(max((y2 - pad_top) / scale, 0.0), max_y)
                self.results.append({
                    "image_id": int(img_id),
                    "category_id": self.label_to_cat_id.get(int(cls_id), int(cls_id)),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],  # COCO format: x, y, w, h
                    "score": float(score),
                })

    def evaluate(self) -> dict[str, float]:
        """Compute COCO metrics.

        Returns:
            Dict with ``mAP``, ``mAP_50``, ``mAP_75``, etc.
        """
        from pycocotools.cocoeval import COCOeval

        if not self.results:
            return {"mAP": 0.0, "mAP_50": 0.0, "mAP_75": 0.0}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(self.results, f)
            tmp_path = f.name

        coco_dt = self.coco_gt.loadRes(tmp_path)
        coco_eval = COCOeval(self.coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        Path(tmp_path).unlink(missing_ok=True)

        return {
            "mAP": coco_eval.stats[0],
            "mAP_50": coco_eval.stats[1],
            "mAP_75": coco_eval.stats[2],
            "mAP_small": coco_eval.stats[3],
            "mAP_medium": coco_eval.stats[4],
            "mAP_large": coco_eval.stats[5],
        }
