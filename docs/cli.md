# CLI Reference

All commands are available through the `yolonas` entry point.

## Global options

| Flag | Description |
|---|---|
| `--version` | Show version and exit |
| `--verbose / -v` | Enable debug logging |
| `--quiet / -q` | Suppress logging output |

## `yolonas detect`

Run object detection on images or video.

```bash
yolonas detect --source image.jpg --model yolo_nas_s --conf 0.25
yolonas detect --source images/ --output results/
yolonas detect --source video.mp4 --skip-frames 2
```

| Option | Default | Description |
|---|---|---|
| `--source` | *required* | Image file, directory, or video path |
| `--model` | `yolo_nas_s` | Model variant (s/m/l) |
| `--conf` | `0.25` | Confidence threshold |
| `--iou` | `0.7` | NMS IoU threshold |
| `--device` | `cuda` | Device (cuda or cpu) |
| `--output` | `results` | Output directory |
| `--input-size` | `640` | Model input size |
| `--skip-frames` | `0` | Process every N-th frame (video) |
| `--codec` | `mp4v` | Video output codec |

## `yolonas track`

Track objects across a video with Deep HM-SORT. See the
[tracking guide](guides/tracking.md) for what the algorithm does and what the defaults
assume about your footage.

```bash
yolonas track --source match.mp4 --classes 0
yolonas track --source match.mp4 --fusion min --max-lost 300   # the Deep-EIoU baseline
yolonas track --source match.mp4 --no-appearance               # motion-only, as a control
```

| Option | Default | Description |
|---|---|---|
| `--source` | *required* | Video file path |
| `--model` | `yolo_nas_s` | Model variant (s/m/l) |
| `--weights` | — | Custom checkpoint; `--model` then selects the architecture |
| `--conf` | `--track-low` | Detection threshold. Must stay at or below `--track-low`, or the low-score association round never sees anything |
| `--iou` | `0.7` | NMS IoU threshold |
| `--classes` | all | Comma-separated class ids to track, filtered before association |
| `--appearance` / `--no-appearance` | on | Associate on per-object embeddings as well as motion |
| `--fusion` | `harmonic` | `harmonic` is Deep HM-SORT; `min` is Deep-EIoU's original |
| `--track-high` | `0.6` | Score at or above which a detection enters the first round |
| `--track-low` | `0.4` | Score below which a detection is ignored |
| `--new-track` | `0.5` | Lowest score that may start a track |
| `--expansion` | `0.3` | Box growth for the first association round |
| `--max-lost` | unset | Frames a track may go unmatched. Unset keeps every tracklet — the paper's default, and a closed-environment assumption |
| `--class-aware` | off | Refuse to associate across classes |
| `--output` | `results` | Output directory |
| `--codec` | `mp4v` | Video output codec |
| `--show-fps` | off | Burn the per-frame time into the output |

Boxes are coloured by track id, not by class, so an ID-swap shows as a colour change.
The summary reports `unique_ids`: far above the true object count means ids are
fragmenting.

## `yolonas train`

Train a YOLO-NAS model.

```bash
yolonas train --data /path/to/dataset --format yolo --epochs 100
```

| Option | Default | Description |
|---|---|---|
| `--data` | *required* | Path to dataset root |
| `--model` | `yolo_nas_s` | Model variant |
| `--format` | `yolo` | Dataset format (yolo/coco) |
| `--epochs` | `300` | Training epochs |
| `--batch-size` | `32` | Batch size per GPU |
| `--lr` | `2e-4` | Learning rate |
| `--device` | `cuda` | Device |
| `--output` | `runs/train` | Output directory |
| `--resume` | `None` | Checkpoint to resume from |
| `--input-size` | `640` | Input size |
| `--workers` | `8` | DataLoader workers |
| `--pretrained/--no-pretrained` | `True` | Use COCO pretrained weights |

## `yolonas export`

Export model to ONNX or OpenVINO format.

```bash
yolonas export --model yolo_nas_s --format onnx
yolonas export --model yolo_nas_s --format openvino --target frigate

# Feature embeddings, alone or beside the detections
yolonas export --model yolo_nas_s --target embedding --output embedding.onnx
yolonas export --model yolo_nas_s --target combined --output combined.onnx

# Detections + a vector per detection, NMS inside the graph
yolonas export --model yolo_nas_s --target objects --output objects.onnx
```

| Option | Default | Description |
|---|---|---|
| `--model` | `yolo_nas_s` | Model variant |
| `--format` | `onnx` | Export format (onnx/openvino) |
| `--output` | auto | Output file path |
| `--input-size` | `640` | Model input size |
| `--opset` | `18` | ONNX opset version |
| `--checkpoint` | `None` | Custom checkpoint path |
| `--target` | `generic` | Export target (generic/frigate/embedding/combined/objects) |
| `--embed-layers` | `c5` | Feature maps to pool, comma-separated (embedding/combined) |
| `--embed-pooling` | `avg` | Spatial pooling, `avg` or `max` (embedding/combined) |
| `--normalize` / `--no-normalize` | normalize | L2-normalize the embedding |

The `embedding` and `combined` targets take a second input, `valid_region` — see the
[export guide](guides/export.md#the-valid_region-input).

## `yolonas eval`

Evaluate model on COCO dataset.

```bash
yolonas eval --data /path/to/coco --split val2017
```

| Option | Default | Description |
|---|---|---|
| `--data` | *required* | Path to COCO dataset root |
| `--model` | `yolo_nas_s` | Model variant |
| `--split` | `val2017` | Split name |
| `--batch-size` | `32` | Batch size |
| `--device` | `cuda` | Device |
| `--conf` | `0.001` | Confidence threshold |
| `--iou` | `0.65` | NMS IoU threshold |
