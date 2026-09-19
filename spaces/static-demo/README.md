---
title: modern-yolonas
emoji: 🔭
colorFrom: indigo
colorTo: blue
sdk: static
app_file: index.html
pinned: false
license: apache-2.0
short_description: YOLO-NAS object detection running in your browser, no server
tags:
  - object-detection
  - yolo
  - yolo-nas
  - computer-vision
  - onnx
---

# modern-yolonas

A browser demo of [modern-yolonas](https://github.com/CondadosAI/modern-yolonas) — YOLO-NAS object
detection rewritten so you can read the whole thing: a model variant is a function, a config is a
dataclass, and there is no layer of indirection that exists only to be configurable. `state_dict`
keys match [super-gradients](https://github.com/Deci-AI/super-gradients) exactly, so the original
pretrained COCO checkpoints load with `strict=True`.

**Everything runs on your machine.** This is a static Space: it serves files and executes no code
of its own. The ONNX graph is downloaded once, then inference runs in your browser through
[onnxruntime-web](https://onnxruntime.ai/docs/tutorials/web/). No image is ever uploaded.

## Using the library

```bash
pip install modern-yolonas
```

```python
import cv2
from modern_yolonas import YoloNASDetector

detector = YoloNASDetector("yolo_nas_s", device="cpu")
image = cv2.imread("street.jpg")
detections = detector(image)          # a supervision.Detections
cv2.imwrite("out.jpg", detector.annotate(image, detections))
```

[GitHub](https://github.com/CondadosAI/modern-yolonas) ·
[Documentation](https://condadosai.github.io/modern-yolonas/) ·
[PyPI](https://pypi.org/project/modern-yolonas/)

## What this runs, and how fast

YOLO-NAS-S at 640×640, exported to ONNX at opset 17 in fp32 (46.5 MB — a one-time download your
browser then caches).

A static Space cannot send the COOP/COEP headers that `SharedArrayBuffer` requires, so WASM runs
**single-threaded**. Measured that way, one forward pass takes roughly **0.75 s**. On a GPU the same
model runs in milliseconds; this figure is a property of single-threaded WASM, not of YOLO-NAS.

fp32 is deliberate. Dynamic int8 quantization shrinks the file to 12.5 MB but, measured on the
sample image, returned 17 detections instead of 19 **and was not faster** (148 ms vs 146 ms under
native onnxruntime) — it trades accuracy for download size alone.

On the bundled example at confidence 0.40 you should see **18** detections, where the Python
library reports 19. The chain is: Python end-to-end 19 → the same input tensor decoded by this
page's JavaScript 19 (identical classes, score delta 6e-7, box delta 0.000 px) → the browser doing
its own letterboxing 18. The gap is preprocessing, not the port: a canvas resamples differently
from OpenCV's `INTER_LINEAR`, which moves scores in the third decimal and drops one marginal box
below the threshold.

## Licence

The **source** of modern-yolonas is Apache-2.0.

The **pretrained COCO weights** in `models/` were converted from Deci AI's super-gradients releases
and remain under the [Super Gradients Model EULA](https://docs.deci.ai/super-gradients/latest/LICENSE.YOLONAS.html)
— **non-commercial use only**. For commercial deployment, train from scratch; no open-licensed COCO
pretrain is currently distributed.

Sample photograph by [Wilfredor](https://commons.wikimedia.org/wiki/User:Wilfredor),
[CC0](https://creativecommons.org/publicdomain/zero/1.0/).
