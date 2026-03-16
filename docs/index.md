# modern-yolonas

A clean, minimal Python reimplementation of [YOLO-NAS](https://github.com/Deci-AI/super-gradients) object detection. No factory patterns, no registries, no OmegaConf — just PyTorch.

## Features

- **Drop-in pretrained weights** — loads super-gradients COCO checkpoints directly
- **Simple API** — `Detector("yolo_nas_s")` → call with an image → get boxes
- **CLI** — `yolonas detect`, `yolonas train`, `yolonas export`, `yolonas eval`
- **ONNX / OpenVINO export** — including Frigate-compatible graph surgery
- **Training** — full training loop with DDP, AMP, EMA, cosine LR
- **All 3 variants** — S (~12M), M (~31M), L (~44M)

## Quick install

```bash
pip install modern-yolonas
```

## Minimal example

```python
from modern_yolonas import Detector

det = Detector("yolo_nas_s", device="cuda")
result = det("image.jpg")
result.save("output.jpg")
```

## Next steps

- [Getting Started](getting-started.md) — install, first detection, CLI basics
- [API Reference](api/detector.md) — full Python API
- [Guides](guides/training.md) — training, export, extending
