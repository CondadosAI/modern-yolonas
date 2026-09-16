# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

> **Note on maintenance.** Releases between `v0.1.0` and `v0.3.0` were tagged
> automatically on every push to `main`, so this file does not describe them
> individually. See the [releases page](https://github.com/CondadosAI/modern-yolonas/releases)
> for the commit-level history of that range.

## [Unreleased]

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
