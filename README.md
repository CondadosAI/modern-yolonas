# modern-yolonas

[![PyPI](https://img.shields.io/pypi/v/modern-yolonas)](https://pypi.org/project/modern-yolonas/)
[![Tests](https://github.com/CondadosAI/modern-yolonas/actions/workflows/ci.yml/badge.svg)](https://github.com/CondadosAI/modern-yolonas/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/modern-yolonas)](https://pypi.org/project/modern-yolonas/)
[![License](https://img.shields.io/pypi/l/modern-yolonas)](https://github.com/CondadosAI/modern-yolonas/blob/main/LICENSE)

A clean, minimal Python reimplementation of [YOLO-NAS](https://github.com/Deci-AI/super-gradients) object detection. No factory patterns, no registries, no OmegaConf — just PyTorch.

Results come back as [`supervision`](https://github.com/roboflow/supervision) `Detections`, so every annotator, tracker, zone and metric in that ecosystem works on them out of the box.

## Install

```bash
uv add modern-yolonas
# or
pip install modern-yolonas
```

## Quick Start

### Detect objects in an image

```python
import cv2
from modern_yolonas import Detector

det = Detector("yolo_nas_s", device="cuda")

image = cv2.imread("image.jpg")
detections = det(image)  # an sv.Detections

for box, score, name in zip(detections.xyxy, detections.confidence, detections.data["class_name"]):
    x1, y1, x2, y2 = box
    print(f"{name}: {score:.2f} [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")

# Save annotated image
cv2.imwrite("output.jpg", det.annotate(image, detections))
```

Because the result is a `supervision` container, filtering is slicing:

```python
people = detections[detections.class_id == 0]
confident = detections[detections.confidence > 0.5]
big = detections[detections.box_area > 5000]
```

and it plugs straight into the rest of the ecosystem:

```python
import supervision as sv

zone = sv.PolygonZone(polygon=my_polygon)
inside = detections[zone.trigger(detections)]

heatmap_annotator = sv.HeatMapAnnotator()
frame = heatmap_annotator.annotate(frame, detections)
```

### Detect objects in a video

```python
from modern_yolonas import Detector

det = Detector("yolo_nas_s", device="cuda")

# Option 1: Write annotated video directly
stats = det.detect_video_to_file("input.mp4", "output.mp4")
print(f"{stats['total_detections']} detections across {stats['total_frames']} frames")

# Option 2: Iterate frames for custom logic
for frame_idx, frame, detections in det.detect_video("input.mp4"):
    print(f"Frame {frame_idx}: {len(detections)} objects")
    # detections.xyxy, .confidence, .class_id are numpy arrays
    # det.annotate(frame, detections) returns the annotated BGR frame
```

### Live webcam detection

```python
import cv2
from modern_yolonas import Detector

det = Detector("yolo_nas_s", device="cuda")

for frame_idx, frame, detections in det.detect_video(source=0):  # 0 = default camera
    cv2.imshow("YOLO-NAS", det.annotate(frame, detections))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cv2.destroyAllWindows()
```

### Low-level model API

```python
import torch
from modern_yolonas import yolo_nas_s

model = yolo_nas_s(pretrained=True).eval().cuda()
x = torch.randn(1, 3, 640, 640).cuda()
pred_bboxes, pred_scores = model(x)
# pred_bboxes: [1, 8400, 4] — x1y1x2y2 pixel coordinates
# pred_scores: [1, 8400, 80] — class probabilities
```

## CLI

```bash
# Detect in images
yolonas detect --model yolo_nas_s --source image.jpg --conf 0.25
yolonas detect --model yolo_nas_l --source images/ --output results/

# Detect in video
yolonas detect --model yolo_nas_s --source video.mp4 --output results/
yolonas detect --model yolo_nas_m --source video.mp4 --skip-frames 2 --conf 0.3

# Training
yolonas train --model yolo_nas_s --data /path/to/dataset --format yolo --epochs 100

# Evaluation
yolonas eval --model yolo_nas_s --data /path/to/coco --split val2017

# Export (needs the extras: pip install "modern-yolonas[onnx]" / [openvino])
yolonas export --model yolo_nas_s --format onnx --output model.onnx
yolonas export --model yolo_nas_s --format openvino --output model.xml

# Export for Frigate (embeds preprocessing + NMS in the graph)
yolonas export --model yolo_nas_s --format onnx --target frigate
yolonas export --model yolo_nas_s --format openvino --target frigate --input-size 320
```

### Frigate Integration

The `--target frigate` export produces a self-contained model that accepts raw `uint8` BGR
input and outputs a flat `[D, 7]` tensor with `[batch, x1, y1, x2, y2, confidence, class_id]`.

Example Frigate configuration:

```yaml
detectors:
  ov:
    type: openvino
    device: GPU

model:
  model_type: yolonas
  width: 320
  height: 320
  input_tensor: nchw
  input_pixel_format: bgr
  path: /config/model_frigate.xml
```

## Tutorials

Step-by-step notebooks in [`tutorials/`](tutorials/):

| Topic | Notebook | Description |
|---|---|---|
| **Roboflow** | [`roboflow/01_explore_dataset.ipynb`](tutorials/roboflow/01_explore_dataset.ipynb) | Download from Roboflow + explore |
| | [`roboflow/02_finetune.ipynb`](tutorials/roboflow/02_finetune.ipynb) | Fine-tune + evaluate + visualize |
| **FiftyOne** | [`fiftyone/01_explore_dataset.ipynb`](tutorials/fiftyone/01_explore_dataset.ipynb) | Load from FiftyOne Zoo + explore |
| | [`fiftyone/02_finetune.ipynb`](tutorials/fiftyone/02_finetune.ipynb) | Fine-tune + evaluate + visualize |
| **Export** | [`export_onnx.ipynb`](tutorials/export_onnx.ipynb) | ONNX export from any checkpoint |
| **Quantization** | [`quantization_ptq.ipynb`](tutorials/quantization_ptq.ipynb) | Post-Training Quantization |
| | [`quantization_qat.ipynb`](tutorials/quantization_qat.ipynb) | Quantization-Aware Training |
| **Inference** | [`inference_onnx.ipynb`](tutorials/inference_onnx.ipynb) | ONNX Runtime inference |

## Examples

Start with the notebook — install, detect, visualize, and inspect the raw model output in
one pass:

- [`notebooks/quickstart.ipynb`](https://github.com/CondadosAI/modern-yolonas/blob/main/notebooks/quickstart.ipynb)

Or the standalone scripts in [`examples/`](https://github.com/CondadosAI/modern-yolonas/blob/main/examples/):

- [`parity_check.py`](https://github.com/CondadosAI/modern-yolonas/blob/main/examples/parity_check.py) — verify this implementation matches
  super-gradients (writes [`docs/benchmarks/parity.md`](https://github.com/CondadosAI/modern-yolonas/blob/main/docs/benchmarks/parity.md))
- [`bench_devices.py`](https://github.com/CondadosAI/modern-yolonas/blob/main/examples/bench_devices.py) — latency across CPU / Intel iGPU /
  NVIDIA dGPU and FP32 / FP16 / INT8 (writes [`docs/benchmarks/latency_matrix.md`](https://github.com/CondadosAI/modern-yolonas/blob/main/docs/benchmarks/latency_matrix.md))
- [`detect_image.py`](https://github.com/CondadosAI/modern-yolonas/blob/main/examples/detect_image.py) — run detection on a single image
- [`detect_video.py`](https://github.com/CondadosAI/modern-yolonas/blob/main/examples/detect_video.py) — run detection on a video file
- [`detect_webcam.py`](https://github.com/CondadosAI/modern-yolonas/blob/main/examples/detect_webcam.py) — live webcam detection

## Variants

| Model | Params | Input | mAP (COCO val) |
|---|---|---|---|
| YOLO-NAS S | 19.05M | 640 | 47.5 |
| YOLO-NAS M | 51.18M | 640 | 51.5 |
| YOLO-NAS L | 66.98M | 640 | 52.2 |

Parameter counts are measured with `sum(p.numel() for p in model.parameters())`. The mAP
column is Deci's published figure, which this project has not independently re-measured.

## Development

```bash
uv sync --dev
uv run pytest tests/ -v
uv run ruff check src/
```

## Acknowledgments

This project is a clean-room reimplementation of the YOLO-NAS architecture originally developed by [Deci AI](https://deci.ai/) and published in their [super-gradients](https://github.com/Deci-AI/super-gradients) library (Apache-2.0). The model architecture, module structure, and state_dict key naming were derived from the super-gradients source code to enable pretrained weight compatibility.

**Pretrained weights notice:** The pretrained COCO weights downloaded by this library (via `pretrained=True`) are provided by Deci AI and are subject to [Deci's YOLO-NAS license](https://github.com/Deci-AI/super-gradients/blob/master/LICENSE.YOLONAS.md), which restricts commercial use and redistribution. The Apache-2.0 license of this repository applies only to the source code, **not** to the pretrained weights. If you train your own weights from scratch, those are entirely yours.

## Citation

If you use modern-yolonas in your research or project, please cite it:

```bibtex
@software{condados2025modernyolonas,
  author       = {Condados, Luis},
  title        = {modern-yolonas: A Clean Reimplementation of YOLO-NAS},
  year         = {2025},
  url          = {https://github.com/CondadosAI/modern-yolonas},
  license      = {Apache-2.0}
}
```

## License

Apache-2.0 — applies to the source code only. See [Acknowledgments](#acknowledgments) for pretrained weight licensing.
