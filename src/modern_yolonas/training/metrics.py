"""COCO-style detection metrics powered by torchmetrics."""

from __future__ import annotations

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
