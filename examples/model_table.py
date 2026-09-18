"""Measure the model table published in the README.

Every column is measured here rather than quoted, so the table can be regenerated
on any machine and disagreements are reproducible:

    uv run examples/model_table.py --coco ~/datasets/coco

Without `--coco` the accuracy columns are skipped and only the static columns
(parameters, FLOPs) and latency are produced.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from modern_yolonas import yolo_nas_l, yolo_nas_m, yolo_nas_s

BUILDERS = {"YOLO-NAS-S": yolo_nas_s, "YOLO-NAS-M": yolo_nas_m, "YOLO-NAS-L": yolo_nas_l}


def count_params(model) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_flops(model, input_size: int) -> float:
    """Total FLOPs for one forward pass, multiplies and adds counted separately."""
    from torch.utils.flop_counter import FlopCounterMode

    counter = FlopCounterMode(display=False)
    with counter, torch.no_grad():
        model(torch.randn(1, 3, input_size, input_size))
    return counter.get_total_flops() / 1e9


def measure_latency(model, input_size: int, device: str, half: bool, runs: int = 30, warmup: int = 8) -> float:
    """Median wall-clock of one forward pass, in milliseconds.

    Model only: no preprocessing, no NMS, single stream, synchronous. An async
    multi-stream pipeline reports higher throughput on the same hardware.
    """
    model = model.to(device).eval()
    x = torch.randn(1, 3, input_size, input_size, device=device)
    if half:
        model, x = model.half(), x.half()

    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if device.startswith("cuda"):
            torch.cuda.synchronize()

        times = []
        for _ in range(runs):
            start = time.perf_counter()
            model(x)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)

    times.sort()
    return times[len(times) // 2]


def evaluate_coco(
    model, coco_root: Path, input_size: int, device: str, batch_size: int, nms_iou: float = 0.70
) -> dict[str, float]:
    """Full COCO val2017 mAP through the same evaluator `yolonas eval` uses."""
    from torch.utils.data import DataLoader

    from modern_yolonas.data.coco import COCODetectionDataset
    from modern_yolonas.data.collate import detection_collate_fn
    from modern_yolonas.data.transforms import Compose, LetterboxResize, Normalize
    from modern_yolonas.inference.postprocess import postprocess
    from modern_yolonas.training.metrics import COCOEvaluator

    ann_file = coco_root / "annotations" / "instances_val2017.json"
    dataset = COCODetectionDataset(
        coco_root / "images" / "val2017",
        ann_file,
        transforms=Compose([LetterboxResize(target_size=input_size), Normalize()]),
        input_size=input_size,
        ignore_empty_annotations=False,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4,
        collate_fn=detection_collate_fn, pin_memory=True,
    )

    model = model.to(device).eval()
    evaluator = COCOEvaluator(ann_file, input_size=input_size)

    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            pred_bboxes, pred_scores = model(images)
            results = postprocess(pred_bboxes, pred_scores, conf_threshold=0.001, iou_threshold=nms_iou)

            start = batch_idx * batch_size
            image_ids = [dataset.ids[i] for i in range(start, min(start + batch_size, len(dataset)))]
            evaluator.update(
                image_ids,
                [r[0] for r in results],
                [r[1] for r in results],
                [r[2] for r in results],
            )
            if (batch_idx + 1) % 20 == 0:
                print(f"  [{batch_idx + 1}/{len(loader)}]", flush=True)

    return evaluator.evaluate()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco", default=None, help="COCO root (images/val2017 + annotations/). Skipped when omitted.")
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--nms-iou", type=float, default=0.70,
                        help="NMS IoU for evaluation. 0.70 measured ~0.1 AP better than the 0.65 "
                             "inference default across a sweep; 0.80 is worse.")
    parser.add_argument("--half", action="store_true", help="Measure latency in FP16.")
    parser.add_argument("--output", default="docs/benchmarks/model_table.json")
    args = parser.parse_args()

    rows = []
    for name, builder in BUILDERS.items():
        print(f"== {name}")
        model = builder(pretrained=True).eval()
        row = {
            "model": name,
            "params_m": round(count_params(model), 2),
            "gflops": round(count_flops(model, args.input_size), 1),
            "input_size": args.input_size,
            "nms_iou": args.nms_iou,
        }
        # Accuracy first: `--half` mutates the model in place, and evaluating a
        # half-precision model would quietly report a different number.
        if args.coco:
            metrics = evaluate_coco(
                model, Path(args.coco).expanduser(), args.input_size, args.device, args.batch_size, args.nms_iou
            )
            row.update({k: round(float(v) * 100, 1) for k, v in metrics.items()})
            print(f"   mAP {row['mAP']}  mAP50 {row['mAP_50']}")

        row["latency_ms"] = round(measure_latency(model, args.input_size, args.device, args.half), 2)
        row["latency_precision"] = "fp16" if args.half else "fp32"
        print(f"   {row['params_m']}M params, {row['gflops']} GFLOPs, {row['latency_ms']} ms")

        rows.append(row)
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
