"""COCO AP for an exported artifact, not just for the PyTorch model.

Publishing an INT8 or FP16 file without an AP behind it asks the reader to assume
quantization was free. It is not obviously free, and this is what settles it:

    uv run examples/runtime_accuracy.py --coco ~/datasets/coco \
        --openvino build/ov/yolo_nas_s_320_int8_external.xml --input-size 320

The evaluation path is `examples/model_table.py`'s, unchanged — the adapters below
present a compiled model or a TensorRT engine as something with `__call__`, `.to()`
and `.eval()`, which is all `evaluate_coco` asks of a model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


class _Adapter:
    """Enough of `nn.Module` for the COCO evaluation loop."""

    def to(self, *_args, **_kwargs):
        return self

    def eval(self):
        return self


class OpenVINOAdapter(_Adapter):
    def __init__(self, xml: str, device: str = "CPU"):
        import openvino as ov

        core = ov.Core()
        self.compiled = core.compile_model(core.read_model(xml), device)
        self.request = self.compiled.create_infer_request()

    def __call__(self, images: torch.Tensor):
        # The IR is exported at batch 1; the evaluator batches, so unroll.
        boxes, scores = [], []
        for image in images.cpu().numpy():
            result = self.request.infer({0: image[None]})
            outputs = [result[out] for out in self.compiled.outputs]
            boxes.append(torch.from_numpy(outputs[0]))
            scores.append(torch.from_numpy(outputs[1]))
        return torch.cat(boxes).to(images.device), torch.cat(scores).to(images.device)


class OnnxAdapter(_Adapter):
    def __init__(self, path: str, provider: str = "cpu"):
        from modern_yolonas.export import onnx_session

        self.session = onnx_session(path, provider)
        spec = self.session.get_inputs()[0]
        # An FP16 graph wants FP16 input; ORT refuses a float32 feed rather than casting.
        self.dtype = "float16" if spec.type == "tensor(float16)" else "float32"
        # A graph exported with a static batch has to be unrolled, as the evaluator
        # batches by 16 and the artifacts for TensorRT are all fixed at 1.
        self.static_batch = isinstance(spec.shape[0], int)

    def _run(self, batch):
        return self.session.run(None, {"images": batch.astype(self.dtype)})

    def __call__(self, images: torch.Tensor):
        array = images.cpu().numpy()
        if self.static_batch:
            outputs = [self._run(array[i : i + 1]) for i in range(array.shape[0])]
            boxes = np.concatenate([o[0] for o in outputs])
            scores = np.concatenate([o[1] for o in outputs])
        else:
            boxes, scores = self._run(array)
        return (torch.from_numpy(boxes).float().to(images.device),
                torch.from_numpy(scores).float().to(images.device))


class HalfAdapter(_Adapter):
    """The PyTorch model with an FP16 graph, fed FP16 input.

    This is the control for a TensorRT FP16 engine: TensorRT 11 takes its precision
    from the ONNX, so an engine built from a half graph and this run the same
    arithmetic. If both lose the same accuracy, the graph is the cause and the
    runtime is not.
    """

    def __init__(self, model, device: str):
        self.model = model.to(device).eval().half()
        self.device = device

    def __call__(self, images: torch.Tensor):
        with torch.no_grad():
            boxes, scores = self.model(images.to(self.device).half())
        return boxes.float(), scores.float()


class TensorRTAdapter(_Adapter):
    def __init__(self, path: str, half: bool = True):
        from modern_yolonas.export import EngineRunner

        self.runner = EngineRunner(path)
        self.dtype = torch.float16 if self.runner._torch_dtype(self.runner.input_name) == torch.float16 else torch.float32

    def __call__(self, images: torch.Tensor):
        # Built at batch 1, so the evaluator's batch is unrolled the same way.
        boxes, scores = [], []
        for image in images:
            b, s = self.runner(image[None].to(self.dtype))
            boxes.append(b.float())
            scores.append(s.float())
        return torch.cat(boxes), torch.cat(scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco", required=True, help="COCO root (images/val2017 + annotations/).")
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--openvino", default=None, help="Path to an OpenVINO .xml.")
    parser.add_argument("--openvino-device", default="CPU")
    parser.add_argument("--onnx", default=None, help="Path to an .onnx.")
    parser.add_argument("--onnx-provider", default="cpu")
    parser.add_argument("--tensorrt", default=None, help="Path to a .engine.")
    parser.add_argument("--torch-model", default=None, help="Variant name, as the reference.")
    parser.add_argument("--torch-half", action="store_true",
                        help="Run the reference in FP16, to separate the numerics of a half graph "
                             "from the runtime that executes it.")
    parser.add_argument("--output", default=None, help="Append the result to this JSON.")
    args = parser.parse_args()

    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from model_table import evaluate_coco

    if args.openvino:
        model, label = OpenVINOAdapter(args.openvino, args.openvino_device), f"openvino:{args.openvino_device}"
        device = "cpu"
    elif args.onnx:
        model, label = OnnxAdapter(args.onnx, args.onnx_provider), f"onnx:{args.onnx_provider}"
        device = "cpu"
    elif args.tensorrt:
        model, label = TensorRTAdapter(args.tensorrt), "tensorrt"
        device = "cuda"
    elif args.torch_model:
        import modern_yolonas
        from modern_yolonas.export import fuse_for_inference

        model = fuse_for_inference(getattr(modern_yolonas, args.torch_model)(pretrained=True))
        label, device = f"pytorch:{args.torch_model}", args.device
        if args.torch_half:
            model, label = HalfAdapter(model, device), f"pytorch-fp16:{args.torch_model}"
    else:
        raise SystemExit("pass one of --openvino / --onnx / --tensorrt / --torch-model")

    metrics = evaluate_coco(model, Path(args.coco).expanduser(), args.input_size,
                            device, args.batch_size, args.nms_iou)
    row = {"artifact": args.openvino or args.onnx or args.tensorrt or args.torch_model,
           "runtime": label, "input_size": args.input_size, "nms_iou": args.nms_iou,
           **{k: round(float(v) * 100, 2) for k, v in metrics.items()}}
    print(json.dumps(row, indent=2))

    if args.output:
        out = Path(args.output)
        rows = json.loads(out.read_text()) if out.exists() else []
        rows.append(row)
        out.write_text(json.dumps(rows, indent=2) + "\n")
        print(f"appended to {out}")


if __name__ == "__main__":
    main()
