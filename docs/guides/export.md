# Export Guide

## Install the export extras

Exporting is not part of the base install. Pull in the runtime you are targeting:

```bash
pip install "modern-yolonas[onnx]"      # ONNX export and onnxruntime
pip install "modern-yolonas[openvino]"  # OpenVINO IR export
```

## ONNX export

```bash
yolonas export --model yolo_nas_s --format onnx --output model.onnx
```

The result is one self-contained file. (Torch's exporter defaults to putting the weights
in a sibling `model.onnx.data`; this command turns that off, because an `.onnx` shipped
without its sidecar loads and then fails at the first inference.)

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
