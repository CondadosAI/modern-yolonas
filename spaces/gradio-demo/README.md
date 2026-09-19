---
title: modern-yolonas
emoji: 🔭
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.28.0
python_version: "3.11"
app_file: app.py
pinned: false
license: apache-2.0
short_description: YOLO-NAS object detection — clean reimplementation, CPU demo
tags:
  - object-detection
  - yolo
  - yolo-nas
  - computer-vision
  - pytorch
---

# modern-yolonas

A CPU demo of [modern-yolonas](https://github.com/CondadosAI/modern-yolonas) — YOLO-NAS object
detection rewritten so you can read the whole thing: a model variant is a function, a config is a
dataclass, and there is no layer of indirection that exists only to be configurable. `state_dict`
keys match [super-gradients](https://github.com/Deci-AI/super-gradients) exactly, so the original
pretrained COCO checkpoints load with `strict=True`.

Detections are returned as [supervision](https://github.com/roboflow/supervision) `Detections`, so
every annotator, tracker and zone in that ecosystem works on them directly.

```bash
pip install modern-yolonas
```

```python
import cv2
from modern_yolonas import YoloNASDetector

detector = YoloNASDetector("yolo_nas_s", device="cpu")
image = cv2.imread("street.jpg")
detections = detector(image)
cv2.imwrite("out.jpg", detector.annotate(image, detections))
```

[GitHub](https://github.com/CondadosAI/modern-yolonas) ·
[Documentation](https://condadosai.github.io/modern-yolonas/) ·
[PyPI](https://pypi.org/project/modern-yolonas/)

## Performance note

This Space runs on free CPU hardware (2 vCPU). Measured on two cores, one 640×640 forward pass
takes about 0.5 s for YOLO-NAS-S, 1.0 s for YOLO-NAS-M and 1.4 s for YOLO-NAS-L. The same code runs
in milliseconds on a GPU — the latency shown in the UI is a property of this hardware, not of the
model.

Only YOLO-NAS-S is loaded at boot; M and L are downloaded and loaded the first time someone
selects them, so the first request for those is slower still.

## Licence

The **source** of modern-yolonas is Apache-2.0.

The **pretrained COCO weights** served here were converted from Deci AI's super-gradients releases
and remain under the [Super Gradients Model EULA](https://docs.deci.ai/super-gradients/latest/LICENSE.YOLONAS.html)
— **non-commercial use only**. For commercial deployment, train from scratch; no open-licensed COCO
pretrain is currently distributed.

Sample photograph by [Wilfredor](https://commons.wikimedia.org/wiki/User:Wilfredor),
[CC0](https://creativecommons.org/publicdomain/zero/1.0/).
