"""Dump the fixtures the JS parity test compares against.

Run from a Python environment that has modern-yolonas installed:

    python spaces/static-demo/tests/make_fixtures.py

Writes into this directory:
  input.bin      the preprocessed [1,3,640,640] float32 tensor
  meta.json      letterbox scale/pad, original size, class names
  reference.json detections the Python pipeline produces from that tensor

Feeding the JS the *same tensor* is the point: it isolates the postprocessing port
from canvas-vs-OpenCV resampling differences, which are a separate, expected gap.
"""

from __future__ import annotations

import json

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch

from modern_yolonas.inference.postprocess import postprocess, rescale_boxes
from modern_yolonas.inference.preprocess import preprocess
from modern_yolonas.inference.visualize import COCO_NAMES

HERE = Path(__file__).parent
DEMO = HERE.parent
CONF_THRESHOLD = 0.4
IOU_THRESHOLD = 0.7


def main() -> None:
    image = cv2.imread(str(DEMO / "examples" / "street.jpg"))
    if image is None:
        raise SystemExit("examples/street.jpg not found")

    tensor, scale, pad = preprocess(image, 640)
    tensor.numpy().astype(np.float32).tofile(HERE / "input.bin")

    json.dump(
        {
            "scale": scale,
            "pad": list(pad),
            "orig": [image.shape[0], image.shape[1]],
            "conf": CONF_THRESHOLD,
            "iou": IOU_THRESHOLD,
            "names": COCO_NAMES,
        },
        (HERE / "meta.json").open("w"),
    )

    model = DEMO / "models" / "yolo-nas-s.onnx"
    if not model.exists():
        raise SystemExit(f"{model} not found — run `yolonas export` first (see spaces/README.md)")

    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    bboxes, scores = session.run(None, {"images": tensor.numpy()})

    boxes, confidences, class_ids = postprocess(
        torch.from_numpy(bboxes), torch.from_numpy(scores),
        CONF_THRESHOLD, IOU_THRESHOLD, multi_label=True,
    )[0]
    boxes = rescale_boxes(boxes, scale, pad, image.shape[:2])

    reference = [
        [COCO_NAMES[int(c)], round(float(s), 6), [round(float(v), 4) for v in b]]
        for b, s, c in zip(boxes, confidences, class_ids)
    ]
    json.dump(reference, (HERE / "reference.json").open("w"), indent=0)
    print(f"wrote fixtures for {len(reference)} detections")


if __name__ == "__main__":
    main()
