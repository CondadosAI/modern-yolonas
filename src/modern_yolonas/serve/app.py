"""FastAPI application for YOLO-NAS object detection."""

from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import JSONResponse


def create_app(
    model: str = "yolo_nas_s",
    device: str = "cuda",
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.7,
) -> FastAPI:
    """Create a FastAPI app with a loaded detector.

    Args:
        model: Model variant name.
        device: Torch device string.
        conf_threshold: Default confidence threshold.
        iou_threshold: Default IoU threshold.

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(title="modern-yolonas", description="YOLO-NAS object detection API")

    detector = None

    @app.on_event("startup")
    async def _load_model():
        nonlocal detector
        from modern_yolonas.inference.detect import Detector

        detector = Detector(
            model=model,
            device=device,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "ok", "model": model, "device": device}

    @app.get("/models")
    async def models() -> dict[str, list[str]]:
        """List available model variants."""
        return {"models": ["yolo_nas_s", "yolo_nas_m", "yolo_nas_l"]}

    @app.post("/detect")
    async def detect(
        file: UploadFile = File(...),
        conf: float = Query(default=None, ge=0, le=1, description="Confidence threshold"),
        iou: float = Query(default=None, ge=0, le=1, description="IoU threshold"),
    ) -> JSONResponse:
        """Run object detection on an uploaded image.

        Returns JSON with detections: list of ``{box, score, class_id}``.
        """
        import cv2

        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image"})

        result = detector(image, conf_threshold=conf, iou_threshold=iou, retain_image=False)

        detections: list[dict[str, Any]] = []
        for box, score, cls_id in zip(result.boxes, result.scores, result.class_ids):
            detections.append({
                "box": [float(x) for x in box],
                "score": float(score),
                "class_id": int(cls_id),
            })

        return JSONResponse(content={"detections": detections, "count": len(detections)})

    return app
