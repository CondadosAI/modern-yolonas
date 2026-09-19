# Export Guide

## Install the export extras

Exporting is not part of the base install. Pull in the runtime you are targeting:

```bash
pip install "modern-yolonas[onnx]"      # ONNX export and onnxruntime (CPU)
pip install "modern-yolonas[onnx-gpu]"  # the same, against onnxruntime's CUDA build
pip install "modern-yolonas[openvino]"  # OpenVINO IR export, with NNCF for INT8
pip install "modern-yolonas[tensorrt]"  # TensorRT engine building
```

`onnx` and `onnx-gpu` are mutually exclusive and declared as conflicting extras.
Both install a module named `onnxruntime`, and with both present you get whichever
wheel landed last — usually the CPU one, which drops `CUDAExecutionProvider` from
the provider list without warning. `modern_yolonas.export.onnx_session` asserts on
`session.get_providers()` for the same reason: the provider a build *supports* and
the provider a session *resolved to* are different questions, and only the second
one is the one you are about to benchmark.

## Picking an input size

The spatial dimensions are baked into the graph — the detection head builds its
anchor grid from them — so one file serves one resolution. At 640 the model produces
8400 anchors; at 320, 2100.

```bash
yolonas export --model yolo_nas_s --format onnx --input-size 320
```

The batch dimension stays symbolic for plain ONNX and is fixed at 1 everywhere else:
TensorRT would need an optimisation profile for a symbolic batch, OpenVINO's INT8
calibration is simpler without one, and the NMS surgery indexes the batch dimension.
`--static-batch` fixes it for ONNX too.

## ONNX export

```bash
yolonas export --model yolo_nas_s --format onnx --output model.onnx
```

The weights go into the file rather than into a `.onnx.data` sidecar, which is what
PyTorch's exporter does by default. A published artifact that silently needs a second
file beside it is a support burden, and TensorRT's byte-string parser cannot resolve
the sidecar at all.

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
yolonas export --model yolo_nas_s --format openvino --precision fp16
yolonas export --model yolo_nas_s --format openvino --precision int8 \
    --calibration-dir ~/datasets/coco/images/val2017
```

INT8 is the reason to reach for OpenVINO on a CPU or an integrated GPU, and it is
where the speedup is — see [the runtime matrix](../benchmarks/runtime_matrix.md).
Calibrate on representative images, not on noise: quantizing against random input
puts the activation ranges somewhere the model never goes.

The head's decode tail — DFL softmax, the anchor arithmetic, the stack back into
boxes — is excluded from quantization. It is elementwise work on a few thousand
values, so INT8 buys nothing there, and OpenVINO's low-precision transformations
throw on the dequantization it produces. The convolutions, which is where the compute
is, are quantized normally.

## TensorRT export

```bash
yolonas export --model yolo_nas_s --format tensorrt --precision fp16 --input-size 320
```

Three things about TensorRT 11 that shape this command:

- **It is always strongly typed.** `BuilderFlag.FP16` and `BuilderFlag.INT8` no
  longer exist, so the engine's precision is whatever the ONNX graph's is.
  `--precision fp16` therefore exports an FP16 ONNX first and builds from that.
- **INT8 needs a QDQ graph.** The implicit calibrator API is gone. Produce a graph
  carrying QuantizeLinear/DequantizeLinear nodes with `yolonas quantize`, and
  TensorRT will honour them.
- **An engine is not portable.** It is compiled for one GPU architecture, one
  TensorRT version and one shape profile, and refuses to deserialise anywhere else.
  A `.engine.json` is written beside it recording exactly what it was built against.

`--hardware-compatible` builds with `HardwareCompatibilityLevel.AMPERE_PLUS`, so the
engine loads on any Ampere-or-newer GPU rather than on one card. That is what makes
publishing a prebuilt engine possible at all, and it is not free: measured on an
RTX 3060 Laptop with YOLO-NAS-S, it costs 19% at 320 and 10% at 640. `--version-compatible`
is separate and lets the engine load under a later TensorRT release; it has its own
cost, which is why the two are separate flags rather than one.

Running an engine:

```python
from modern_yolonas.export import EngineRunner

runner = EngineRunner("yolo_nas_s_320_fp16.engine")
boxes, scores = runner(images)  # images: CUDA tensor [1, 3, 320, 320]
```

## NMS inside the graph

`--target end2end` appends NonMaxSuppression to the graph, so the model returns
`detections [D, 7]` — `[batch, x1, y1, x2, y2, confidence, class_id]` — instead of raw
`[N, 4]` + `[N, 80]` tensors.

```bash
yolonas export --model yolo_nas_s --format onnx --target end2end \
    --input-size 320 --conf-threshold 0.25 --iou-threshold 0.45
```

**This does not make inference faster, and the measurement says so.** On an RTX 3060
Laptop, YOLO-NAS-S at 320 in TensorRT FP16: 0.91 ms for the model alone, 1.28 ms for
the model plus torchvision's `batched_nms`, and 1.95 ms with NMS in the graph. ONNX's
NonMaxSuppression is simply slower than torchvision's. What `end2end` buys is
deployment shape: one file, one call, no Python in the inference path, and no
megabyte of raw tensors crossing back per frame in a pipeline that does copy them.

Pre-processing stays outside the graph deliberately. The letterbox scale and padding
are per-image and the caller needs them anyway to map boxes back to original pixels,
so baking them in would hide numbers the caller has to have. The Frigate target is
the one exception, because Frigate does the resize itself.

INT8 and `end2end` cannot be combined: NNCF cannot calibrate through a baked-in
NonMaxSuppression.

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

## Pre-exported artifacts

`examples/export_zoo.py` produces the whole cross product — every variant, at every
size, in every format — with a `manifest.json` recording each file's size, hash, and
what it will load on:

```bash
uv run examples/export_zoo.py --out build/zoo \
    --calibration-dir ~/datasets/coco/images/val2017
```

The ONNX and OpenVINO files are architecture-independent and cover ARM as they are.
TensorRT engines are built `--hardware-compatible`, because a natively built engine
is faster and loads on exactly one machine, which is no use to anyone downloading it.
