"""Publish the export zoo to the Hugging Face Hub.

Dry by default. It prints exactly what would be uploaded and to where, and does
nothing until `--push` is passed:

    uv run examples/publish_zoo.py --zoo build/zoo                  # plan only
    uv run examples/publish_zoo.py --zoo build/zoo --push           # upload

The weights these artifacts are derived from are Deci's pretrained COCO checkpoints,
under the Super Gradients Model EULA — research use only. The model card says so,
because a `.onnx` file downloaded from a model page carries no such warning the way
`yolo_nas_s(pretrained=True)` does when it prints one.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

CARD = """---
license: other
license_name: super-gradients-model-eula
license_link: https://docs.deci.ai/super-gradients/latest/LICENSE.YOLONAS.html
library_name: modern-yolonas
pipeline_tag: object-detection
tags:
  - object-detection
  - yolo-nas
  - onnx
  - openvino
  - tensorrt
  - edge-ai
---

# modern-yolonas — pre-exported runtimes

Every pretrained YOLO-NAS variant, at every input size this project measures, in
every runtime it supports. Produced by
[`examples/export_zoo.py`](https://github.com/CondadosAI/modern-yolonas/blob/main/examples/export_zoo.py)
from [modern-yolonas](https://github.com/CondadosAI/modern-yolonas); `manifest.json`
records every file's size, SHA-256 and provenance.

## ⚠️ License: these weights are not open

The parameters in every file here come from Deci AI's pretrained COCO checkpoints,
which remain under the **Super Gradients Model EULA — non-commercial use only**. The
`modern-yolonas` *source* is Apache-2.0; these artifacts are not. For a commercial
deployment, train from scratch — COCO's annotations are CC-BY 4.0 and the training
code is in the repository.

## What is here

{inventory}

## Which file do you want

| You are running on | Take |
|---|---|
| Anything, as a starting point | `<model>_<size>.onnx` — FP32, dynamic batch, architecture-independent |
| CPU or an Intel iGPU | `<model>_<size>_int8.xml` + `.bin` — the INT8 is where the speedup is |
| An NVIDIA GPU with TensorRT | `<model>_<size>_fp16.onnx`, and build the engine locally |
| An NVIDIA GPU, no build step | `<model>_<size>_fp16_ampere_plus.engine` — see the caveat below |
| A single-file deployment, no Python | `<model>_<size>_end2end.onnx` — NMS in the graph |

## The TensorRT engines, and why you probably want the ONNX instead

A TensorRT engine is compiled for one GPU architecture, one TensorRT version and one
shape profile. The engines here are built with `HardwareCompatibilityLevel.AMPERE_PLUS`
so they load on any Ampere-or-newer card, but they still require **the exact TensorRT
build recorded in `manifest.json`**, and hardware compatibility costs performance:
measured on an RTX 3060 Laptop with YOLO-NAS-S, 19% at 320 and 10% at 640.

Building from the FP16 ONNX on your own machine takes a couple of minutes and gives
you the faster engine:

```bash
pip install "modern-yolonas[tensorrt]"
yolonas export --model yolo_nas_s --format tensorrt --precision fp16 --input-size 320
```

## Input and output

Input is `float32 [B, 3, S, S]`, RGB, `[0, 1]`, letterboxed with aspect preserved and
centre-padded at 114. `modern_yolonas.inference.preprocess.preprocess` does this and
returns the scale and padding you need to map boxes back.

Output is `pred_bboxes [B, N, 4]` in `x1y1x2y2` letterboxed pixels and
`pred_scores [B, N, 80]`, except for the `_end2end` files, which return
`detections [D, 7]` as `[batch, x1, y1, x2, y2, confidence, class_id]` with NMS
already applied.

`N` follows the input size: 2100 at 320, 8400 at 640.

## Accuracy and latency

Measured, not quoted, and regenerable:
[model table](https://github.com/CondadosAI/modern-yolonas/blob/main/docs/benchmarks/model_table.md)
and [runtime matrix](https://github.com/CondadosAI/modern-yolonas/blob/main/docs/benchmarks/runtime_matrix.md).
"""


def inventory(artifacts: list[dict]) -> str:
    groups = defaultdict(list)
    for entry in artifacts:
        groups[(entry["model"], entry["input_size"])].append(entry)

    lines = ["| model | input | file | format | precision | NMS | size |", "|---|---:|---|---|---|---|---:|"]
    for (model, size), entries in sorted(groups.items()):
        for entry in entries:
            if entry["path"].endswith(".bin"):
                continue  # listed with its .xml
            lines.append(
                f"| {model} | {size} | `{entry['path']}` | {entry['format']} | "
                f"{entry['precision']} | {entry['nms']} | {entry['bytes'] / 1e6:.0f} MB |"
            )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zoo", default="build/zoo")
    parser.add_argument("--repo", default="CondadosAI/modern-yolonas-export")
    parser.add_argument("--push", action="store_true", help="Actually create the repo and upload.")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    zoo = Path(args.zoo)
    manifest = json.loads((zoo / "manifest.json").read_text())
    artifacts = manifest["artifacts"]
    total = sum(entry["bytes"] for entry in artifacts)

    card = CARD.format(inventory=inventory(artifacts))
    (zoo / "README.md").write_text(card)

    print(f"repo:      {args.repo}{' (private)' if args.private else ''}")
    print(f"artifacts: {len(artifacts)} files, {total / 1e9:.2f} GB")
    print(f"card:      {zoo / 'README.md'}")
    if not args.push:
        print("\nDry run. Nothing was uploaded. Re-run with --push.")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", exist_ok=True, private=args.private)
    api.upload_folder(
        folder_path=str(zoo),
        repo_id=args.repo,
        repo_type="model",
        commit_message="Pre-exported ONNX, OpenVINO IR and TensorRT engines",
    )
    print(f"\nhttps://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
