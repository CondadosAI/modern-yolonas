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

Track objects across a video, with ByteTrack (the default) or Deep HM-SORT. ByteTrack needs
the `tracking` extra. See the [tracking guide](guides/tracking.md) for how the two differ and
the [tracking benchmark](benchmarks/tracking.md) for why ByteTrack is the default.

```bash
yolonas track --source match.mp4 --classes 0                              # ByteTrack
yolonas track --source match.mp4 --tracker deep-hm-sort                   # Deep HM-SORT
yolonas track --source match.mp4 --tracker deep-hm-sort --fusion min      # the Deep-EIoU baseline
yolonas track --source match.mp4 --tracker deep-hm-sort --keep-all-tracks # the paper's memory
yolonas track --source match.mp4 --tracker deep-hm-sort --no-appearance   # motion-only
```

The options marked *deep-hm-sort* only apply with `--tracker deep-hm-sort`; passing one
with ByteTrack is an error rather than a silent no-op.

| Option | Default | Description |
|---|---|---|
| `--source` | *required* | Video file path |
| `--tracker` | `bytetrack` | `bytetrack` or `deep-hm-sort` |
| `--model` | `yolo_nas_s` | Model variant (s/m/l) |
| `--weights` | — | Custom checkpoint; `--model` then selects the architecture |
| `--conf` | the tracker's floor | Detection threshold: 0.1 for ByteTrack, `--track-low` for Deep HM-SORT. Keep it at or below that floor, or the low-score association round never sees anything |
| `--iou` | `0.7` | NMS IoU threshold |
| `--classes` | all | Comma-separated class ids to track, filtered before association. ByteTrack is not class-aware, so this is how to keep classes apart |
| `--appearance` / `--no-appearance` | on | *deep-hm-sort.* Associate on per-object embeddings as well as motion |
| `--fusion` | `harmonic` | *deep-hm-sort.* `harmonic` is Deep HM-SORT; `min` is Deep-EIoU's original |
| `--track-high` | `0.6` | *deep-hm-sort.* Score at or above which a detection enters the first round |
| `--track-low` | `0.4` | *deep-hm-sort.* Score below which a detection is ignored |
| `--new-track` | `0.5` | *deep-hm-sort.* Lowest score that may start a track |
| `--expansion` | `0.3` | *deep-hm-sort.* Box growth for the first association round |
| `--max-lost-seconds` | `2.0` | *deep-hm-sort.* How long a track may go unmatched, in seconds of video |
| `--keep-all-tracks` | off | *deep-hm-sort.* Never drop a track. The paper's setting, and a closed-environment assumption |
| `--class-aware` | off | *deep-hm-sort.* Refuse to associate across classes |
| `--output` | `results` | Output directory |
| `--codec` | `mp4v` | Video output codec |
| `--show-fps` | off | Burn the per-frame time into the output |

Boxes are coloured by track id, not by class, so an ID-swap shows as a colour change.
The summary reports `unique_ids`: far above the true object count means ids are
fragmenting.

## `yolonas benchmark-tracking`

Measure the tracker on a MOT-format dataset. Split in two, because the detector pass
takes minutes and a tracker pass takes seconds — an ablation should not pay for the
first one every time, and every configuration then sees byte-identical detections, so
a difference in the metrics can only have come from the association.

```bash
# Once per dataset: detect (or read the ground truth) and embed every frame.
yolonas benchmark-tracking cache --data ~/datasets/sportsmot/val --source oracle
yolonas benchmark-tracking cache --data ~/datasets/sportsmot/val --source detector

# As often as you like: replay every tracker configuration and score it.
yolonas benchmark-tracking evaluate --data ~/datasets/sportsmot/val --source oracle
```

Needs the `mot` extra for the metrics: `uv sync --extra mot`, or
`pip install "modern-yolonas[mot]"`. It is separate from `benchmark` because TrackEval
depends on `opencv-python` where this project depends on `opencv-python-headless`;
installing it replaces the headless build, which then wants libGL at import time.

### `cache`

| Option | Default | Description |
|---|---|---|
| `--data` | *required* | Split directory, one subdirectory per sequence |
| `--source` | `detector` | `detector` runs detection; `oracle` embeds the ground-truth boxes, leaving association as the only source of error |
| `--model` | `yolo_nas_l` | Model variant |
| `--conf` | `0.1` | Detector threshold. Low on purpose — the tracker filters on replay, so a score sweep needs no re-detection |
| `--output` | `runs/mot-cache` | Where the per-sequence `.npz` caches go |
| `--sequences` | all | A split file, or a comma-separated list of names |
| `--limit` | — | Only the first N sequences, for a smoke run |
| `--overwrite` | off | Rebuild caches that already exist |

### `evaluate`

| Option | Default | Description |
|---|---|---|
| `--data` | *required* | The same split directory |
| `--cache` | `runs/mot-cache` | Where `cache` wrote its output |
| `--source` | `detector` | Which cache to replay |
| `--benchmark` / `--split` | `sportsmot` / `val` | TrackEval names; also name the output folder |
| `--preproc` | off | TrackEval's MOT17 preprocessing. Needed for MOT17, a no-op on SportsMOT |
| `--configs` | all | Comma-separated subset of `harmonic`, `min`, `motion`, `harmonic-keepall`, `motion-keepall` |
| `--output` | `runs/mot-eval` | Results table, `results.json` and the TrackEval tree |

Ground truth is symlinked into the TrackEval layout, never copied — SportsMOT and
MOT17 are both non-redistributable, and a copy inside the repo is the accident worth
designing out.

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
