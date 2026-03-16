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
from modern_yolonas import Detector

det = Detector("yolo_nas_s", device="cuda")
result = det("photo.jpg")

print(f"Found {len(result.boxes)} objects")
result.save("output.jpg")
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
| `yolo_nas_s` | ~12M | 47.5 |
| `yolo_nas_m` | ~31M | 51.5 |
| `yolo_nas_l` | ~44M | 52.2 |
