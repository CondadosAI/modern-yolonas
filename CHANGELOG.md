# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

> **Note on maintenance.** Releases between `v0.1.0` and `v0.3.0` were tagged
> automatically on every push to `main`, so this file does not describe them
> individually. See the [releases page](https://github.com/CondadosAI/modern-yolonas/releases)
> for the commit-level history of that range.

## [Unreleased]

## [0.5.0] - 2026-09-18

### Changed — breaking
- **License: MIT → Apache-2.0.** The source is Apache-2.0 from this branch on; `v0.4.0`
  and everything before it shipped under MIT. The pretrained COCO weights are unaffected
  either way — they come from Deci and carry their own non-commercial terms, which
  `weights.py` prints at download time.
- Training runs on [PyTorch Lightning](https://lightning.ai/). The hand-written `Trainer`,
  `ModelEMA` and the manual callback system (`CSVLoggerCallback`, `EarlyStoppingCallback`,
  `RichProgressCallback`, `TensorBoardCallback`, `WandbCallback`) are gone; Lightning's own
  loggers and callbacks replace them. `yolonas train` keeps its full flag surface — `--amp`
  now picks the precision, `--num-gpus` the device count and ddp strategy, `--val-freq` the
  validation interval.
- `Mixup(dataset, p=...)` is now `Mixup(dataset, prob=...)`, matching `Mosaic`.

### Added
- **`YoloNASDetector`, the new name for `Detector`.** `Detector` said nothing about which
  library it came from, and the name was about to become a problem: the planned
  segmentation, pose and feature-extraction entry points need task siblings, and `YoloNAS*`
  has to stay free for the `nn.Module` classes those tasks will bring (`YoloNASPose` and
  friends, matching super-gradients). So the ergonomic layer carries a role suffix —
  `YoloNASDetector` today, `YoloNASSegmenter` and `YoloNASPoseEstimator` later — and the
  module layer keeps the bare architecture names.
- Demo media in `docs/assets/`: an annotated street-scene still (CC0) and a 3-second
  annotated clip of the Shibuya crossing. The clip's source is CC BY-SA 4.0, so those two
  media files are CC BY-SA 4.0 rather than Apache-2.0; `docs/assets/README.md` records the
  provenance and the scope of that obligation. The source code is unaffected.
- Conventional Commits are now enforced, and drive the release bump. A `commitizen`
  `commit-msg` hook checks the message as you write it, `pr-title.yml` checks the pull
  request title (which is what a squash merge actually records), and release-drafter's
  autolabeler turns that title into the `minor`/`patch` label its version-resolver reads.
  Previously no rule applied those labels, so every release drafted as a patch — including
  `v0.4.0`, which was a `feat!`.
- Quantization: `yolonas quantize` (post-training) and `yolonas qat`
  (quantization-aware training), built on `torch.ao.quantization` FX graph mode.
- `yolonas benchmark-dataset coco` and `yolonas benchmark-dataset rf100vl` — train and
  report mAP. Distinct from `yolonas benchmark`, which measures inference latency.
- `--close-mosaic-epochs`: train the final N epochs without Mosaic/Mixup, as the
  super-gradients recipes do.
- `Detector.annotate(..., show_fps=True)` and `Detector.last_inference_ms` for an inference
  speed overlay; `detect_video_to_file(show_fps=True)` burns it into the output.
- Dataset-aware augmentation: `Mosaic`, `Mixup` with inner transforms, `VerticalFlip`,
  `RandomCrop`, `RandomChannelShuffle`, and a recipe-driven `build_transforms`.
- `data/dataset_config.py` reads YOLO/Roboflow `data.yaml`; `data/fiftyone.py` and
  `data/download.py` fetch COCO and RF100-VL.
- Tutorial notebooks for ONNX export/inference, quantization (PTQ and QAT), FiftyOne and
  Roboflow workflows.

### Changed
- `extract_model_state_dict` moved to `weights.py` and is what every consumer now uses, so
  inference, evaluation and export all read Lightning `.ckpt` files as well as the legacy
  trainer's format and plain state-dicts.
- Gradient clipping is on by default at norm 10 (`--grad-clip`), which the training review
  calls the usual cause of a from-scratch run that NaNs in its first epoch.
- Accuracy is now measured rather than quoted. The README table carries this project's own
  COCO val2017 numbers — 47.3 / 51.3 / 52.0 AP for S / M / L — alongside parameters, FLOPs
  and latency, all regenerable with `uv run examples/model_table.py`.
- Example scripts are invoked as `uv run <script>`; the previous `python examples/...`
  only worked with a virtualenv already activated.
- CI runs on `dev` as well as `main`, and `main` accepts pull requests only from `dev`.

### Deprecated
- `Detector`, in favour of `YoloNASDetector`. Nothing breaks yet: the old spelling still
  resolves from `modern_yolonas`, `modern_yolonas.inference` and
  `modern_yolonas.inference.detect`, raising `DeprecationWarning` on access. It is removed
  in 0.7.0, which is where the actual break lands.

### Fixed
- **Mosaic emitted every box at half size.** Cropping the 2s×2s canvas to s×s renormalises
  coordinates by 2; the centres were scaled and the widths and heights were not. With
  mosaic at probability 1.0 in the COCO recipe, every training sample taught the model to
  predict boxes half as large as the objects. Boxes are now clipped to the crop window as
  well, rather than only filtered by centre.
- **Validation overwrote BatchNorm running statistics.** The validation step flipped the
  model to `train()` to get the raw predictions the loss needs; `torch.no_grad` stops
  gradients but not buffer updates, so validation-set statistics were written into the
  checkpoint. `NDFLHeads.return_raw_outputs` returns them without touching any module's
  training flag.
- **`COCOEvaluator` was missing**, so every path that computes mAP — validation with
  annotations, `yolonas eval`, both dataset benchmarks — raised `ImportError` on its lazy
  import.
- **Predictions were never mapped out of letterboxed space** before being compared with
  ground truth in original image pixels, which put every mAP this project could produce
  near zero. Measured on a fixed checkpoint: 0.008 → 0.588.
- `yolonas train` letterboxes for validation instead of centre-cropping. A centre crop
  deletes objects at the edges and raises outright on any image smaller than the input
  size, which is most of COCO val2017 at 640.
- The checkpoint callback monitors `val/mAP` when annotations are present; it previously
  hardcoded `val/loss`, which is not logged in that case, so `--format coco` died at the
  first validation.

## [0.4.0] - 2026-09-18

### Changed — breaking
- `Detector` now returns [`supervision.Detections`](https://supervision.roboflow.com/latest/detection/core/)
  instead of the project's own `Detection` dataclass. Field names follow supervision:
  `boxes` → `xyxy`, `scores` → `confidence`, `class_ids` → `class_id`. Results filter by
  slicing (`detections[detections.confidence > 0.5]`) and work directly with supervision's
  annotators, trackers, zones and metrics.
- `Detector.detect_video` yields `(frame_index, frame, detections)`. `sv.Detections` does
  not carry the source image, so the frame is yielded alongside it; the `retain_image`
  argument is gone.
- `result.save(path)` and `result.visualize()` are replaced by
  `Detector.annotate(image, detections)`, which returns the annotated frame for the caller
  to display or write.
- `Detection` is no longer exported from `modern_yolonas`.
- `Detector` accepts `class_names`; names default to COCO only when `num_classes` is 80.

### Added
- `supervision>=0.30` as a core dependency. Releases before 0.30 required
  `opencv-python`, which conflicts with this project's `opencv-python-headless`; 0.30
  dropped that dependency, which is what makes this practical.
- Package metadata for PyPI: `readme`, `keywords`, classifiers, and project URLs
- Community docs: `CONTRIBUTING.md` (with API design principles),
  `CODE_OF_CONDUCT.md`, issue forms, `CITATION.cff`, Dependabot config
- Benchmark artifacts are now published on the docs site under `docs/benchmarks/`

### Changed
- CI now tests both ends of the supported Python range (3.10 and 3.13) instead of
  3.13 only; `ruff` and `mypy` target 3.10 to match `requires-python`
- Development and docs tooling moved to PEP 735 dependency groups, removing a
  duplicate `dev` definition that declared two different `pytest` floors

### Removed
- The `auto-tag` CI job, which tagged a patch bump on every push to `main`. Releases are
  now cut by pushing a `v*` tag, which `publish.yml` already listens for. Every merge was
  a release, minor and major bumps were impossible without intervention, and this
  changelog went stale at `0.1.0` while tags reached `v0.3.0` as a result.
- `onnxscript` from the core dependencies; it is only needed by the ONNX exporter
  and remains in the `onnx` extra

## [0.1.0] - 2025-06-01

### Added
- YOLO-NAS S/M/L model architectures (QARepVGG backbone, PAN neck, DFL heads)
- Pretrained weight loading from super-gradients checkpoints (`strict=True` compatible)
- Inference pipeline: preprocessing, postprocessing with NMS, visualization
- Image detection with bounding box overlay and class labels
- Video detection with frame iteration and direct file output
- COCO and YOLO format dataset loaders with mosaic/mixup augmentations
- Training with DDP, AMP, EMA, cosine LR scheduler, PPYoloE loss
- COCO mAP evaluation
- ONNX export with dynamic batch axes
- Click-based CLI: `detect`, `train`, `eval`, `export`
- CI with GitHub Actions (lint + test)
