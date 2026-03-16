"""Export YOLO-NAS to ONNX and validate with ONNX Runtime.

Usage:
    python examples/export_onnx.py --model yolo_nas_s --output model.onnx
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from modern_yolonas import yolo_nas_s, yolo_nas_m, yolo_nas_l


def main():
    parser = argparse.ArgumentParser(description="Export YOLO-NAS to ONNX")
    parser.add_argument("--model", default="yolo_nas_s", choices=["yolo_nas_s", "yolo_nas_m", "yolo_nas_l"])
    parser.add_argument("--output", default="yolo_nas_s.onnx")
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    builders = {"yolo_nas_s": yolo_nas_s, "yolo_nas_m": yolo_nas_m, "yolo_nas_l": yolo_nas_l}
    model = builders[args.model](pretrained=True).eval()

    # Fuse RepVGG branches for inference
    for m in model.modules():
        if hasattr(m, "fuse_block_residual_branches"):
            m.fuse_block_residual_branches()

    dummy = torch.randn(1, 3, args.input_size, args.input_size)

    print(f"Exporting {args.model} to ONNX (opset {args.opset})...")
    torch.onnx.export(
        model,
        dummy,
        args.output,
        input_names=["images"],
        output_names=["pred_bboxes", "pred_scores"],
        dynamic_axes={
            "images": {0: "batch"},
            "pred_bboxes": {0: "batch"},
            "pred_scores": {0: "batch"},
        },
        opset_version=args.opset,
    )
    print(f"Saved to {args.output}")

    # Validate with ONNX Runtime
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(args.output)
        inputs = {"images": np.random.randn(1, 3, args.input_size, args.input_size).astype(np.float32)}
        bboxes, scores = session.run(None, inputs)
        print(f"ORT validation passed: bboxes={bboxes.shape}, scores={scores.shape}")
    except ImportError:
        print("Install onnxruntime to validate: pip install onnxruntime")


if __name__ == "__main__":
    main()
