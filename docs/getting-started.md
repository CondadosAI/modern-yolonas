# Getting Started

## Installation

```bash
pip install modern-yolonas
# or with uv
uv add modern-yolonas
```

For ONNX export support:
```bash
pip install modern-yolonas[onnx]
```

For OpenVINO export:
```bash
pip install modern-yolonas[openvino]
```

## First detection

```python
import cv2
from modern_yolonas import Detector

det = Detector("yolo_nas_s", device="cuda")

image = cv2.imread("photo.jpg")
detections = det(image)

print(f"Found {len(detections)} objects")
cv2.imwrite("output.jpg", det.annotate(image, detections))
```

`detections` is a [`supervision.Detections`](https://supervision.roboflow.com/latest/detection/core/),
so it filters by slicing and works with every supervision annotator, tracker and zone:

```python
people = detections[detections.class_id == 0]
confident = detections[detections.confidence > 0.5]
```

## Using the CLI

```bash
# Detect objects in an image
yolonas detect --source photo.jpg --model yolo_nas_s

# Detect in a directory of images
yolonas detect --source images/ --output results/

# Detect in video
yolonas detect --source video.mp4 --conf 0.3

# Export to ONNX
yolonas export --model yolo_nas_s --format onnx

# Train on a custom dataset
yolonas train --model yolo_nas_s --data /path/to/dataset --format yolo
```

## Low-level model API

```python
import torch
from modern_yolonas import yolo_nas_s

model = yolo_nas_s(pretrained=True).eval().cuda()
x = torch.randn(1, 3, 640, 640).cuda()
pred_bboxes, pred_scores = model(x)
# pred_bboxes: [1, 8400, 4] — x1y1x2y2 pixel coordinates
# pred_scores: [1, 8400, 80] — class probabilities
```

## Model variants

| Model | Params | mAP (COCO val) |
|---|---|---|
| `yolo_nas_s` | 19.05M | 47.5 |
| `yolo_nas_m` | 51.18M | 51.5 |
| `yolo_nas_l` | 66.98M | 52.2 |
