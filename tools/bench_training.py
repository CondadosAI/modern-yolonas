"""Find out what a training step spends its time on, before renting a GPU for it.

Three measurements that discriminate between the two things people guess about:

    loader   iterate the training dataloader with NO model attached. Pure CPU.
    stages   single process, per-stage breakdown inside the transform pipeline.
    model    synthetic batch on the GPU: forward, loss, backward and step, timed apart.

``loader`` faster than ``model`` in img/s means the GPU is the bottleneck and no
amount of dataloader work will help; slower means the opposite. The ratio between
them is the whole budget any optimisation can recover, so run both before changing
anything. ``stages`` then says which transform to spend that budget on.

The per-sample cost is per image, so val2017 measures it as well as train2017 and
downloads in a fraction of the time.

Examples::

    uv run tools/bench_training.py loader --data ~/datasets/coco --workers 4 8 16
    uv run tools/bench_training.py stages --data ~/datasets/coco
    uv run tools/bench_training.py model --batch 16 --channels-last

Every run reports the machine it measured, because none of these numbers transfer
between machines — a figure from a laptop says nothing about a rented pod.
"""

from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from modern_yolonas.data.coco import COCODetectionDataset
from modern_yolonas.data.collate import detection_collate_fn
from modern_yolonas.data.transforms import (
    Compose,
    HSVAugment,
    LetterboxResize,
    Mixup,
    Mosaic,
    Normalize,
    RandomAffine,
    RandomChannelSwap,
    RandomResizedCropFlipAffine,
    TrainTransformPipeline,
)


def pin_threads(worker_id: int) -> None:
    """One compute thread per worker.

    OpenCV keeps its own thread pool, and N dataloader workers each spinning one up
    oversubscribes the cores. Worth 3% at 8 workers and 24% at 20 on a 20-core box:
    a guard against a wrong worker count rather than a speedup.
    """
    cv2.setNumThreads(0)
    torch.set_num_threads(1)


def machine() -> str:
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no CUDA"
    import os

    return f"{platform.node()} | {os.cpu_count()} vCPU | {gpu} | torch {torch.__version__}"


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------

def build_pipeline(kind: str, dataset, size: int):
    """``cli`` is what ``yolonas train`` builds; ``mosaic`` is what ``training/run.py`` does.

    They differ more than the names suggest — ``yolonas train`` uses no Mosaic at all —
    so a number measured on one says nothing about the other.
    """
    if kind == "cli":
        pipe = Compose([
            HSVAugment(p=0.5),
            RandomResizedCropFlipAffine(size=size, scale=(0.05, 0.8), ratio=(0.75, 1.33),
                                        flip_prob=0.5, translate=0.25, affine_scale=(0.5, 1.5)),
            RandomChannelSwap(p=0.5),
            Normalize(dtype="uint8"),
        ])
        pipe.transforms.insert(-1, Mixup(dataset, prob=0.5))
        return pipe

    per_image = Compose([
        RandomAffine(degrees=0.0, translate=0.1, scale=(0.5, 1.5)),
        HSVAugment(p=1.0),
    ])
    mixup = Mixup(dataset, prob=0.5)
    mixup.inner_transforms = per_image
    return TrainTransformPipeline(
        mosaic=Mosaic(dataset, input_size=size, prob=1.0),
        per_image_transforms=per_image,
        mixup=mixup,
        final_transforms=Compose([LetterboxResize(target_size=size), Normalize(dtype="uint8")]),
    )


def make_dataset(data: Path, kind: str, size: int):
    images = data / "images" / "val2017"
    anns = data / "annotations" / "instances_val2017.json"
    for p in (images, anns):
        if not p.exists():
            raise SystemExit(f"not found: {p}\nPass --data pointing at a COCO root.")
    ds = COCODetectionDataset(images, anns, transforms=None, input_size=size)
    ds.transforms = build_pipeline(kind, ds, size)
    return ds


# ---------------------------------------------------------------------------
# loader
# ---------------------------------------------------------------------------

def bench_loader(args) -> None:
    ds = make_dataset(args.data, args.pipeline, args.size)
    print(machine())
    print(f"{len(ds)} images | pipeline={args.pipeline} | batch={args.batch}\n")
    print(f"{'workers':>8} {'img/s':>9} {'MB/batch':>9}")

    for workers in args.workers:
        dl = DataLoader(
            ds, batch_size=args.batch, shuffle=True, num_workers=workers,
            collate_fn=detection_collate_fn, pin_memory=True, drop_last=True,
            worker_init_fn=pin_threads if workers > 0 else None,
            persistent_workers=args.persistent and workers > 0,
            prefetch_factor=args.prefetch if workers > 0 else None,
        )
        it = iter(dl)
        mb = 0.0
        for _ in range(args.warmup):          # worker spawn and page cache
            images, _ = next(it)
            mb = images.element_size() * images.nelement() / 1e6

        t0 = time.perf_counter()
        for _ in range(args.batches):
            next(it)
        dt = time.perf_counter() - t0
        print(f"{workers:>8} {args.batches * args.batch / dt:>9.1f} {mb:>9.1f}")
        del it, dl


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------

def bench_stages(args) -> None:
    pin_threads(0)
    ds = make_dataset(args.data, args.pipeline, args.size)
    pipe = ds.transforms
    idxs = np.random.default_rng(0).integers(0, len(ds), args.samples)
    acc: dict[str, float] = {}

    def tick(name: str, t0: float) -> float:
        now = time.perf_counter()
        acc[name] = acc.get(name, 0.0) + (now - t0)
        return now

    for i in idxs:
        i = int(i)
        if args.pipeline == "mosaic":
            t = time.perf_counter()
            image, targets = pipe.mosaic(i)
            t = tick("mosaic", t)
            image, targets = pipe.per_image_transforms(image, targets)
            t = tick("per_image", t)
            image, targets = pipe.mixup(image, targets)
            t = tick("mixup", t)
            pipe.final_transforms(image, targets)
            tick("final", t)
        else:
            t = time.perf_counter()
            image, targets = ds.load_raw(i)
            t = tick("load_raw (decode)", t)
            for step in pipe.transforms:
                image, targets = step(image, targets)
                t = tick(type(step).__name__, t)

    total = sum(acc.values())
    ms = total / args.samples * 1000
    print(machine())
    print(f"\npipeline={args.pipeline}  n={args.samples}  "
          f"{ms:.2f} ms/sample single process -> {1000 / ms:.1f} img/s per core\n")
    print(f"{'stage':<36} {'ms/sample':>10} {'%':>6}")
    for name, v in sorted(acc.items(), key=lambda kv: -kv[1]):
        print(f"{name:<36} {v / args.samples * 1000:>10.2f} {v / total * 100:>5.1f}%")


# ---------------------------------------------------------------------------
# frozen teachers
# ---------------------------------------------------------------------------

# A distillation step pays for a frozen teacher forward on top of the student's
# own forward and backward. On a small card that is the difference between a
# recipe that fits and one that does not, so it has to be measured rather than
# reasoned about. Only the teacher's *cost* is measured here -- what the
# distillation loss should be is a separate question.
TEACHERS = {
    # DINOv3 emits one patch token per 16x16 cell, which is exactly the stride of
    # the backbone's c4, so the student feature needs a 1x1 projection and nothing else.
    "dinov3-s": {"repo": "facebook/dinov3-vits16-pretrain-lvd1689m", "dim": 384, "spatial": True},
    "dinov3-b": {"repo": "facebook/dinov3-vitb16-pretrain-lvd1689m", "dim": 768, "spatial": True},
    # D-FINE is query-based: there is no spatial map to align c4 against, so this
    # measures the forward alone. Distilling from it means matching queries or
    # logits, which is a different design.
    "dfine-l": {"repo": "ustc-community/dfine-large-coco", "dim": 256, "spatial": False},
    "dfine-x": {"repo": "ustc-community/dfine-xlarge-coco", "dim": 256, "spatial": False},
}


def load_teacher(name: str, device: torch.device, channels_last: bool):
    """Load a frozen teacher. Requires the ``distill`` extra."""
    try:
        from transformers import AutoModel, AutoModelForObjectDetection
    except ImportError:
        raise SystemExit(
            "--teacher needs transformers: uv sync --extra distill\n"
            "DINOv3 is a gated repo, so export HF_TOKEN as well."
        )

    spec = TEACHERS[name]
    # A detector checkpoint is a ForObjectDetection model; loading it through
    # AutoModel silently drops the head and randomly initialises what is left.
    loader = AutoModel if spec["spatial"] else AutoModelForObjectDetection
    model = loader.from_pretrained(spec["repo"]).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if channels_last and not spec["spatial"]:
        model = model.to(memory_format=torch.channels_last)
    return model, spec


def teacher_spatial_features(outputs, batch: int, side: int, dim: int) -> torch.Tensor:
    """Patch tokens as ``[B, dim, side, side]``, dropping CLS and register tokens."""
    tokens = outputs.last_hidden_state
    tokens = tokens[:, tokens.shape[1] - side * side:, :]
    return tokens.transpose(1, 2).reshape(batch, dim, side, side)


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

def bench_model(args) -> None:
    from modern_yolonas import yolo_nas_l, yolo_nas_m, yolo_nas_s
    from modern_yolonas.training.loss import PPYoloELoss

    if not torch.cuda.is_available():
        raise SystemExit("model needs a CUDA device.")

    device = torch.device("cuda")
    builder = {"yolo_nas_s": yolo_nas_s, "yolo_nas_m": yolo_nas_m, "yolo_nas_l": yolo_nas_l}[args.model]
    model = builder(pretrained=False, num_classes=80).to(device).train()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    if args.compile:
        # Model only. Compiling the loss would recompile on every change in the
        # number of ground-truth boxes, which varies per batch by construction.
        model = torch.compile(model)
    criterion = PPYoloELoss(num_classes=80)

    teacher = teacher_spec = projection = None
    if args.teacher:
        teacher, teacher_spec = load_teacher(args.teacher, device, args.channels_last)
        if teacher_spec["spatial"]:
            # c4 is stride 16 with 384 channels on yolo_nas_s; a 1x1 conv is the
            # whole distillation head.
            c4_channels = model.backbone.out_channels[2] if hasattr(model, "backbone") else 384
            projection = torch.nn.Conv2d(c4_channels, teacher_spec["dim"], 1).to(device)
            if args.channels_last:
                projection = projection.to(memory_format=torch.channels_last)

    trainable = list(model.parameters()) + (list(projection.parameters()) if projection else [])
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)

    images = torch.randn(args.batch, 3, args.size, args.size, device=device)
    if args.channels_last:
        images = images.to(memory_format=torch.channels_last)

    # 7 objects per image is the COCO train average.
    rows = [[b, np.random.randint(80), 0.5, 0.5, 0.2, 0.2] for b in range(args.batch) for _ in range(7)]
    targets = torch.tensor(rows, dtype=torch.float32, device=device)

    amp = {"bf16": torch.bfloat16, "fp16": torch.float16, "off": None}[args.amp]
    torch.backends.cudnn.benchmark = True

    def step(timed: bool) -> dict[str, float]:
        marks: dict[str, float] = {}

        def mark(name: str, t0: float) -> float:
            if timed:
                torch.cuda.synchronize()
                marks[name] = time.perf_counter() - t0
            return time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        t = time.perf_counter()
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            if teacher is None:
                predictions = model(images)
                t = mark("forward", t)
            else:
                # One student pass serving both heads, as a real distillation step
                # would do -- running the model twice would double the measurement.
                features = model.forward_features(images)
                predictions = model.heads((features["p3"], features["p4"], features["p5"]))
                t = mark("forward", t)
                with torch.no_grad():
                    teacher_out = teacher(images)
                t = mark("teacher forward (frozen)", t)

            loss, _ = criterion(predictions, targets, input_size=(args.size, args.size), epoch=1)

            if teacher is not None and teacher_spec["spatial"]:
                side = args.size // 16
                target_feature = teacher_spatial_features(
                    teacher_out, args.batch, side, teacher_spec["dim"]
                ).float()
                student_feature = projection(features["c4"]).float()
                loss = loss + (1 - torch.nn.functional.cosine_similarity(
                    student_feature, target_feature, dim=1
                ).mean())
            t = mark("loss (with assigner)", t)
        loss.backward()
        t = mark("backward", t)
        optimizer.step()
        mark("optimizer.step", t)
        return marks

    for _ in range(args.warmup):
        step(False)

    agg: dict[str, float] = {}
    t0 = time.perf_counter()
    for _ in range(args.iters):
        for name, v in step(True).items():
            agg[name] = agg.get(name, 0.0) + v
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    print(machine())
    print(f"\n{args.model} | batch={args.batch} size={args.size} amp={args.amp} "
          f"channels_last={args.channels_last} compile={args.compile} "
          f"teacher={args.teacher or 'none'}")
    print(f"T_model = {args.batch * args.iters / wall:.1f} img/s "
          f"({wall / args.iters * 1000:.1f} ms/step)")
    print(f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB\n")
    total = sum(agg.values())
    print(f"{'phase':<24} {'ms/step':>9} {'%':>6}")
    for name, v in sorted(agg.items(), key=lambda kv: -kv[1]):
        print(f"{name:<24} {v / args.iters * 1000:>9.1f} {v / total * 100:>5.1f}%")


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_data(p):
        p.add_argument("--data", type=Path, default=Path.home() / "datasets" / "coco",
                       help="COCO root holding images/val2017 and annotations/.")
        p.add_argument("--pipeline", default="cli", choices=["cli", "mosaic"])
        p.add_argument("--size", type=int, default=640)

    p = sub.add_parser("loader", help="dataloader throughput, no model")
    add_data(p)
    p.add_argument("--workers", type=int, nargs="+", default=[0, 4, 8])
    p.add_argument("--persistent", action="store_true")
    p.add_argument("--prefetch", type=int, default=2)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--batches", type=int, default=30)
    p.add_argument("--warmup", type=int, default=3)
    p.set_defaults(func=bench_loader)

    p = sub.add_parser("stages", help="per-transform breakdown, single process")
    add_data(p)
    p.add_argument("--samples", type=int, default=250)
    p.set_defaults(func=bench_stages)

    p = sub.add_parser("model", help="GPU step, synthetic batch")
    p.add_argument("--model", default="yolo_nas_s",
                   choices=["yolo_nas_s", "yolo_nas_m", "yolo_nas_l"])
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--size", type=int, default=640)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--amp", default="fp16", choices=["fp16", "bf16", "off"])
    p.add_argument("--channels-last", action="store_true")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile the model. Warmup is excluded from the timing, "
                        "so raise --warmup: the first steps pay for compilation.")
    p.add_argument("--teacher", default=None, choices=sorted(TEACHERS),
                   help="Add a frozen teacher forward to the step, as distillation "
                        "training would. Needs the 'distill' extra.")
    p.set_defaults(func=bench_model)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
