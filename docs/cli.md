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
```

| Option | Default | Description |
|---|---|---|
| `--model` | `yolo_nas_s` | Model variant |
| `--format` | `onnx` | Export format (onnx/openvino) |
| `--output` | auto | Output file path |
| `--input-size` | `640` | Model input size |
| `--opset` | `17` | ONNX opset version |
| `--checkpoint` | `None` | Custom checkpoint path |
| `--target` | `generic` | Export target (generic/frigate) |

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
