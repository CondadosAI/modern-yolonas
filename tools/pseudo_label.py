"""Pseudo-label images with an open-licensed detector.

Deci warm-started YOLO-NAS's heads on pseudo-labelled COCO images; this is that
step, with a teacher whose licence lets us publish the result. D-FINE-X reaches
55.8 AP on COCO under Apache-2.0, and because it was trained on COCO its boxes
follow COCO's conventions by construction — which is why it is here and SAM 3,
a stronger segmenter with its own notion of an instance, is not.

Running it is cheap: one inference pass, offline, once. Keeping a teacher in the
training loop instead costs 36% of every step forever (see
``tools/bench_training.py --teacher``).

Two modes:

    label      write a COCO-format annotation file for a directory of images
    validate   label a set that *has* ground truth and score against it

**Run validate before trusting label.** Pseudo-labels fail silently: a wrong
confidence threshold or a broken class mapping produces a plausible-looking file
that quietly poisons every epoch trained on it. Validate reproduces the teacher's
published AP or tells you something is wrong while it is still cheap to fix.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from modern_yolonas import COCO_NAMES

# D-FINE ships VOC-style spellings for six classes at COCO's own indices. The
# order is identical to COCO's, so the mapping is positional; matching on names
# would silently drop these six.
KNOWN_SYNONYMS = {
    "motorcycle": "motorbike",
    "airplane": "aeroplane",
    "couch": "sofa",
    "potted plant": "pottedplant",
    "dining table": "diningtable",
    "tv": "tvmonitor",
}

TEACHERS = {
    "dfine-x": "ustc-community/dfine-xlarge-coco",
    "dfine-l": "ustc-community/dfine-large-coco",
    "rtdetrv2-x": "PekingU/rtdetr_v2_r101vd",
}


def check_class_order(id2label: dict) -> None:
    """Fail loudly if the teacher's class order is not COCO's.

    Every pseudo-label depends on this. A reordered checkpoint would relabel the
    whole dataset consistently and produce a model that trains perfectly well and
    detects the wrong things.
    """
    teacher = [id2label[i] for i in range(len(id2label))]
    if len(teacher) != len(COCO_NAMES):
        raise SystemExit(f"teacher has {len(teacher)} classes, COCO has {len(COCO_NAMES)}")
    for index, (ours, theirs) in enumerate(zip(COCO_NAMES, teacher)):
        if ours == theirs or KNOWN_SYNONYMS.get(ours) == theirs:
            continue
        raise SystemExit(
            f"class order mismatch at index {index}: ours={ours!r}, teacher={theirs!r}.\n"
            "The positional mapping this tool relies on does not hold for this checkpoint."
        )


def load_categories(reference: Path) -> list[dict]:
    """COCO's real category ids, in COCO's order, from an annotation file.

    The teacher emits contiguous 0-79; COCO's own ids run 1-90 with gaps. Reading
    them from a real annotation file rather than hardcoding keeps the output
    loadable by pycocotools.
    """
    with reference.open() as handle:
        data = json.load(handle)
    categories = sorted(data["categories"], key=lambda c: c["id"])
    if len(categories) != 80:
        raise SystemExit(f"{reference} has {len(categories)} categories, expected 80")
    return categories


@torch.no_grad()
def run(args) -> dict:
    try:
        from transformers import AutoImageProcessor, AutoModelForObjectDetection
    except ImportError:
        raise SystemExit("needs transformers: uv sync --extra distill")

    from PIL import Image

    device = torch.device(args.device)
    repo = TEACHERS[args.teacher]
    processor = AutoImageProcessor.from_pretrained(repo)
    model = AutoModelForObjectDetection.from_pretrained(repo).to(device).eval()
    check_class_order(model.config.id2label)

    categories = load_categories(args.categories_from)
    category_id = [c["id"] for c in categories]

    files = sorted(p for p in args.images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"no images in {args.images}")

    images_out: list[dict] = []
    annotations: list[dict] = []
    next_id = 1
    started = time.perf_counter()

    for start in range(0, len(files), args.batch):
        chunk = files[start : start + args.batch]
        pil = [Image.open(p).convert("RGB") for p in chunk]
        inputs = processor(images=pil, return_tensors="pt").to(device)

        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            outputs = model(**inputs)

        sizes = [(im.height, im.width) for im in pil]
        results = processor.post_process_object_detection(
            outputs, target_sizes=sizes, threshold=args.threshold
        )

        for path, (height, width), result in zip(chunk, sizes, results):
            image_id = int(path.stem)
            images_out.append(
                {"id": image_id, "file_name": path.name, "height": height, "width": width}
            )
            for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
                x1, y1, x2, y2 = (float(v) for v in box)
                w, h = x2 - x1, y2 - y1
                if w < 1 or h < 1:
                    continue
                annotations.append({
                    "id": next_id,
                    "image_id": image_id,
                    "category_id": category_id[int(label)],
                    "bbox": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
                    "area": round(w * h, 2),
                    "iscrowd": 0,
                    "score": round(float(score), 4),
                })
                next_id += 1

        done = start + len(chunk)
        if done % (args.batch * 20) == 0 or done == len(files):
            rate = done / (time.perf_counter() - started)
            eta = (len(files) - done) / rate / 60
            print(f"  {done}/{len(files)}  {rate:.1f} img/s  eta {eta:.1f} min", flush=True)

    return {
        "info": {
            "description": f"Pseudo-labels from {repo}",
            "teacher": repo,
            "threshold": args.threshold,
        },
        "images": images_out,
        "annotations": annotations,
        "categories": categories,
    }


def cmd_label(args) -> None:
    data = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        json.dump(data, handle)
    per_image = len(data["annotations"]) / max(1, len(data["images"]))
    print(f"\n{len(data['images'])} images, {len(data['annotations'])} boxes "
          f"({per_image:.1f} per image) -> {args.out}")
    print("COCO train2017 averages 7.3 boxes per image; a wildly different number "
          "means the threshold is wrong.")


def cmd_validate(args) -> None:
    """Score the teacher's own output against real annotations.

    This is the gate. If the teacher does not roughly reproduce its published AP
    here, the threshold, the class mapping or the postprocessing is wrong, and
    123k bad labels are worse than none.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    args.categories_from = args.ann
    data = run(args)
    if not data["annotations"]:
        raise SystemExit("no detections at all — threshold far too high, or the model is broken")

    coco = COCO(str(args.ann))
    scored_ids = [im["id"] for im in data["images"]]
    detections = [
        {k: a[k] for k in ("image_id", "category_id", "bbox", "score")}
        for a in data["annotations"]
    ]
    evaluation = COCOeval(coco, coco.loadRes(detections), iouType="bbox")
    evaluation.params.imgIds = scored_ids
    evaluation.evaluate()
    evaluation.accumulate()
    evaluation.summarize()

    ap = evaluation.stats[0]
    print(f"\nthreshold {args.threshold} -> AP {ap:.3f} over {len(scored_ids)} images")
    print("A detection AP is maximised by keeping low-confidence boxes, so this number "
          "is a sanity check on the plumbing, not the threshold to train with.\n"
          "For that, compare precision and recall across thresholds: pseudo-labels want "
          "precision, because a false box teaches a wrong object every epoch.")


def _iou(a: list[float], b: list[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return inter / (aw * ah + bw * bh - inter + 1e-9)


def cmd_sweep(args) -> None:
    """Precision and recall per threshold, which is what picks the threshold.

    AP is maximised by keeping low-confidence boxes, so it cannot answer this.
    Pseudo-labelling has a two-sided cost that classification does not: a false
    box teaches a wrong object every epoch, and a *missed* box teaches the model
    that a real object is background. Neither end of the curve is safe.
    """
    from pycocotools.coco import COCO

    args.categories_from = args.ann
    args.threshold = min(args.thresholds)
    data = run(args)

    coco = COCO(str(args.ann))
    image_ids = [im["id"] for im in data["images"]]
    truth = {
        i: [(a["category_id"], a["bbox"])
            for a in coco.loadAnns(coco.getAnnIds(imgIds=i)) if not a.get("iscrowd")]
        for i in image_ids
    }
    total_truth = sum(len(v) for v in truth.values())

    print(f"\n{'threshold':>10} {'precision':>10} {'recall':>8} {'boxes/img':>10}")
    for threshold in sorted(args.thresholds):
        # Highest score first: a greedy match in arbitrary order lets a weak
        # detection consume the ground truth a strong one would have matched.
        kept: dict[int, list] = {}
        for a in sorted(data["annotations"], key=lambda a: -a["score"]):
            if a["score"] >= threshold:
                kept.setdefault(a["image_id"], []).append((a["category_id"], a["bbox"]))

        true_positive = predicted = 0
        for image_id in image_ids:
            matched: set[int] = set()
            for category, box in kept.get(image_id, []):
                predicted += 1
                for index, (gt_category, gt_box) in enumerate(truth[image_id]):
                    if index in matched or gt_category != category:
                        continue
                    if _iou(box, gt_box) >= args.iou:
                        true_positive += 1
                        matched.add(index)
                        break
        print(f"{threshold:>10.2f} {true_positive / max(1, predicted):>10.3f} "
              f"{true_positive / max(1, total_truth):>8.3f} {predicted / len(image_ids):>10.1f}")

    print(f"\nground truth: {total_truth / len(image_ids):.1f} boxes per image "
          f"over {len(image_ids)} images, at IoU {args.iou}")
    print("Precision here is measured against COCO, which is itself incompletely "
          "annotated, so it understates the teacher: a correct detection of an "
          "unannotated object counts as a false positive.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--images", type=Path, required=True)
        p.add_argument("--teacher", default="dfine-x", choices=sorted(TEACHERS))
        p.add_argument("--threshold", type=float, default=0.5)
        p.add_argument("--batch", type=int, default=8)
        p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        p.add_argument("--limit", type=int, default=0, help="stop after N images (smoke tests)")

    p = sub.add_parser("label", help="write pseudo-labels for a directory")
    common(p)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--categories-from", type=Path, required=True,
                   help="a real COCO annotation file, for the category ids")
    p.set_defaults(func=cmd_label)

    p = sub.add_parser("validate", help="score the teacher against real annotations")
    common(p)
    p.add_argument("--ann", type=Path, required=True)
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("sweep", help="precision and recall per threshold")
    common(p)
    p.add_argument("--ann", type=Path, required=True)
    p.add_argument("--thresholds", type=float, nargs="+",
                   default=[0.3, 0.4, 0.5, 0.6, 0.7])
    p.add_argument("--iou", type=float, default=0.5)
    p.set_defaults(func=cmd_sweep)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
