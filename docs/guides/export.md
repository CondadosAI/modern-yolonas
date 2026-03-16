# Export Guide

## ONNX export

```bash
yolonas export --model yolo_nas_s --format onnx --output model.onnx
```

```python
import torch
from modern_yolonas import yolo_nas_s

model = yolo_nas_s(pretrained=True).eval()

# Fuse RepVGG branches for inference
for m in model.modules():
    if hasattr(m, "fuse_block_residual_branches"):
        m.fuse_block_residual_branches()

dummy = torch.randn(1, 3, 640, 640)
torch.onnx.export(
    model, dummy, "yolo_nas_s.onnx",
    input_names=["images"],
    output_names=["pred_bboxes", "pred_scores"],
    dynamic_axes={"images": {0: "batch"}},
    opset_version=17,
)
```

### Validate with ONNX Runtime

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("yolo_nas_s.onnx")
inputs = {"images": np.random.randn(1, 3, 640, 640).astype(np.float32)}
bboxes, scores = session.run(None, inputs)
print(f"bboxes: {bboxes.shape}, scores: {scores.shape}")
```

## OpenVINO export

```bash
yolonas export --model yolo_nas_s --format openvino --output model.xml
```

## Frigate integration

The `--target frigate` option bakes preprocessing (uint8 BGR input) and NMS
into the ONNX graph, producing a single self-contained model:

```bash
yolonas export --model yolo_nas_s --format openvino --target frigate --input-size 320
```

Output tensor shape: `[D, 7]` with columns `[batch, x1, y1, x2, y2, confidence, class_id]`.

Frigate config:

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

## Custom checkpoint export

```bash
yolonas export --model yolo_nas_s --checkpoint runs/train/last.pt --format onnx
```
