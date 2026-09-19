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

The result is one self-contained file. PyTorch's exporter defaults to putting the
weights in a sibling `model.onnx.data`; this turns that off, because an `.onnx`
shipped without its sidecar loads and then fails at the first inference, and
TensorRT's byte-string parser cannot resolve the sidecar at all.

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
    opset_version=18,
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

## Embedding export

Two targets export the feature embeddings rather than (or alongside) the detections.

```bash
# Embeddings only — the detection head is not in the graph
yolonas export --model yolo_nas_s --target embedding --output embedding.onnx

# Detections and an embedding, from one pass
yolonas export --model yolo_nas_s --target combined --output combined.onnx

# Pick the layers and pooling, exactly as FeaturePooler does
yolonas export --model yolo_nas_s --target embedding --embed-layers c4,c5 --embed-pooling max
yolonas export --model yolo_nas_s --target embedding --no-normalize
```

| Target | Inputs | Outputs |
|:---|:---|:---|
| `embedding` | `images`, `valid_region` | `embedding` |
| `combined` | `images`, `valid_region` | `pred_bboxes`, `pred_scores`, `embedding` |

`combined` is the deployment form of
[`predict(..., Task.DETECT | Task.EMBED)`](embeddings.md#one-pass-both-outputs): the
backbone and neck run once and both results are read off them, rather than shipping two
models over the same frames. `pred_bboxes` are in canvas coordinates with NMS not yet
applied — the same contract as the plain detection export, so existing postprocessing
applies unchanged.

### The `valid_region` input

Both graphs take a second input: `valid_region`, `[B, 4]` int64
`(left, top, right, bottom)` in canvas pixels, saying where each image's real pixels sit
inside the letterbox.

It is a separate input rather than a constant because it depends on the aspect ratio of
the image being embedded, which is not known at export time. And it is not optional:
pooling the gray letterbox padding in makes embeddings cluster by aspect ratio instead of
by content — [measured at 0.958 cosine between an unrelated noise image and a street
photo](embeddings.md#letterbox-padding-is-excluded), against 0.396 when the padding is
excluded.

`valid_region` is a public function that takes what `preprocess` already returns:

```python
import numpy as np
import onnxruntime as ort

from modern_yolonas.inference.embed import valid_region
from modern_yolonas.inference.preprocess import preprocess

session = ort.InferenceSession("embedding.onnx")

tensor, scale, pad = preprocess(image, 640)          # image is a BGR array
region = valid_region(image, scale, pad)             # (left, top, right, bottom)

embedding = session.run(None, {
    "images": tensor.numpy(),
    "valid_region": np.array([region], dtype=np.int64),
})[0]                                                 # (1, 768), L2-normalized
```

To pool the whole canvas anyway, pass `(0, 0, canvas, canvas)` — but that is the
behaviour the input exists to avoid.

`examples/embed_onnx.py` is this loop over a folder, ranking a gallery against a query.

### Per-object embeddings — the `objects` target

```bash
yolonas export --model yolo_nas_s --target objects --output objects.onnx \
    --conf-threshold 0.25 --iou-threshold 0.45 --max-detections 20
```

A self-contained graph: image in, detections **and** one vector per detection out, with
NMS and ROI pooling inside it.

| Output | Shape | Contents |
|:---|:---|:---|
| `detections` | `[D, 7]` | `batch, x1, y1, x2, y2, confidence, class_id` |
| `object_embedding` | `[D, E]` | one row per detection, same order |
| `embedding` | `[B, E]` | the whole-image vector, as in the other targets |

Row *i* of `object_embedding` describes row *i* of `detections`. That is the contract, and
it is what the tests pin.

This one could not be traced out of PyTorch. Which boxes exist depends on which survive
NMS — a data-dependent shape `torch.export` will not produce — so the graph is built by
exporting a base model whose feature maps are outputs, then adding `NonMaxSuppression`,
`RoiAlign` and the pooling as ONNX nodes. Same approach as the `frigate` target.

```python
detections, object_embedding, embedding = session.run(None, {
    "images": tensor.numpy(),
    "valid_region": np.array([region], dtype=np.int64),
})

people = detections[detections[:, 6] == COCOClass.PERSON]
```

**Boxes come back in canvas coordinates**, like the plain detection export.
`rescale_boxes(boxes, scale, pad, image.shape[:2])` maps them to the source image — the
same function the PyTorch path uses.

**Boxes are clipped to `valid_region` before they are embedded**, matching
`embed_boxes` and `predict(..., Task.EMBED_OBJECTS)`: outside it there is only letterbox
padding, so a detection running off the edge is described by the part of it that is real.
A box that clips to zero area samples nothing and comes back as a zero vector, never NaN.

#### Thresholds are not the same as `postprocess`'s

`--max-detections` is `max_output_boxes_per_class` — the ONNX NMS operator counts **per
class**, where `postprocess` caps the total per image. The defaults (`--iou-threshold
0.45`, `--max-detections 20`) are the Frigate-tuned ones this flag has always had. To get
closer to what `predict` does, pass `--iou-threshold 0.7 --max-detections 300`.

Exact detection parity with `postprocess` is **not** claimed: the graph uses the ONNX NMS
operator and `postprocess` uses `torchvision.ops.batched_nms` with a top-1024 prefilter,
so the surviving sets can differ at the margin. What *is* pinned is that for whatever
boxes the graph does emit, the vectors match PyTorch ROI pooling on those same boxes.

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
RTX 3060 Laptop with YOLO-NAS-S, it costs 16% at 320 and 8% at 640, and no accuracy
(47.31 AP against 47.33 for a natively built FP32 engine). `--version-compatible`
is separate and lets the engine load under a later TensorRT release; it has its own
cost, which is why the two are separate flags rather than one.

Running an engine:

```python
from modern_yolonas.export import EngineRunner

runner = EngineRunner("yolo_nas_s_320_fp16.engine")
boxes, scores = runner(images)  # images: CUDA tensor [1, 3, 320, 320]
```

`EngineRunner` binds torch's own CUDA tensors into TensorRT, which avoids a second
allocator and a second CUDA context. It runs on a stream of its own — enqueueing on
the default stream makes TensorRT insert extra synchronizations — and **orders that
stream against torch's** before and after the call. Skipping that ordering is not a
performance detail: the caching allocator associates a tensor with the stream that
produced it, so TensorRT can read an input whose copy has not landed. It does not
fail. It returns plausible numbers that cost 16 AP, which is how this was found.

An engine's accuracy is worth checking rather than assuming:

```bash
uv run examples/runtime_accuracy.py --coco ~/datasets/coco --input-size 640 \
    --tensorrt yolo_nas_s_640_fp16.engine
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
Laptop, YOLO-NAS-S at 320 in TensorRT FP16: 1.17 ms for the model alone, 1.55 ms for
the model plus torchvision's `batched_nms`, and 1.92 ms with NMS in the graph. ONNX's
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
