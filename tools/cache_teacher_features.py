"""Precompute DINOv3 features once, so distillation training never pays for them.

A frozen teacher inside the training loop costs 36% of every step (measured,
``tools/bench_training.py --teacher``). Run once offline it costs 434 img/s on a
4090, which is nine minutes for all of COCO. The trade is that a cached feature
belongs to one view of the image, so training may only use augmentations that a
cached feature survives:

* **photometric** — HSV, channel shuffle. These move no pixels, and matching an
  augmented image to the clean image's features is *invariance training*, which
  is what self-supervised methods do deliberately.
* **horizontal flip** — only with ``--with-flip``, which caches a second set of
  features for the mirrored image. Mirroring the cached grid instead does *not*
  work: a ViT carries positional embeddings and is not flip-equivariant. Measured
  on val2017, a mirrored cache matches the teacher's own output for the mirrored
  image at cosine 0.867, against 0.9999 for the unmirrored pair. A CNN would have
  been close to exact here; this one is not, and the error is 3 orders of
  magnitude above the quantiser's.

Random crops and affine warps are not available at all. If the distilled backbone
turns out to want geometric diversity, the fused affine matrix could be applied to
the feature grid too (see ``RandomResizedCropFlipAffine``), at stride-16 resolution
and with the same caveat about what a ViT does under geometry.

Storage: int8 directions plus fp16 norms, 0.61 MB per image, 148 GB for COCO's
241k. Measured against fp32, int8 per-token preserves cosine similarity to
0.99989 on average and 0.99976 at worst -- the distillation loss is cosine, which
ignores scale entirely, so only direction has to survive.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

TEACHERS = {
    "dinov3-s": ("facebook/dinov3-vits16-pretrain-lvd1689m", 384),
    "dinov3-b": ("facebook/dinov3-vitb16-pretrain-lvd1689m", 768),
}


def letterbox(image: np.ndarray, size: int, pad_value: int = 114) -> np.ndarray:
    """Exactly the geometry ``LetterboxResize`` gives the student.

    The teacher's patch grid has to line up with the student's feature map, so
    both sides must see the same pixels in the same places.
    """
    h, w = image.shape[:2]
    scale = size / max(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), pad_value, np.uint8)
    top, left = (size - new_h) // 2, (size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas


def quantize(features: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """int8 directions plus fp16 norms.

    The cosine loss ignores magnitude, so only the direction has to survive the
    round trip; the norms are kept at 0.5% overhead so a future MSE-style loss is
    not locked out.
    """
    norms = features.norm(dim=-1, keepdim=True)
    directions = features / norms.clamp_min(1e-6)
    scale = directions.abs().amax(dim=-1, keepdim=True) / 127.0
    quantized = torch.round(directions / scale.clamp_min(1e-12)).clamp(-127, 127)
    return (
        quantized.to(torch.int8).cpu().numpy(),
        (norms.squeeze(-1) * scale.squeeze(-1)).to(torch.float16).cpu().numpy(),
    )


def dequantize(quantized: np.ndarray, scaled_norms: np.ndarray) -> np.ndarray:
    """Inverse of :func:`quantize`, up to the direction the cosine loss reads."""
    return quantized.astype(np.float32) * scaled_norms.astype(np.float32)[..., None]


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=Path, nargs="+", required=True,
                        help="one or more image directories; all are cached into one index")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--teacher", default="dinov3-s", choices=sorted(TEACHERS))
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--shard-size", type=int, default=2000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--with-flip", action="store_true",
                        help="also cache the mirrored image's features, doubling the "
                             "size and enabling the horizontal flip in training")
    args = parser.parse_args()

    try:
        from transformers import AutoModel
    except ImportError:
        raise SystemExit("needs transformers: uv sync --extra distill")

    repo, dim = TEACHERS[args.teacher]
    grid = args.size // 16
    tokens = grid * grid

    device = torch.device(args.device)
    model = AutoModel.from_pretrained(repo).to(device).eval()
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    files: list[Path] = []
    for directory in args.images:
        files.extend(sorted(p for p in directory.iterdir()
                            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}))
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit("no images found")

    args.out.mkdir(parents=True, exist_ok=True)
    per_image_mb = (tokens * dim + tokens * 2) / 1e6 * (2 if args.with_flip else 1)
    print(f"{len(files)} images | {args.teacher} | {grid}x{grid} grid x {dim} dims")
    print(f"{per_image_mb:.2f} MB/image -> {per_image_mb * len(files) / 1000:.1f} GB total\n")

    index = {"teacher": repo, "size": args.size, "patch": 16, "grid": grid, "dim": dim,
             "shard_size": args.shard_size, "files": [], "shards": [],
             "has_flip": bool(args.with_flip)}
    started = time.perf_counter()
    shard_q: list[np.ndarray] = []
    shard_n: list[np.ndarray] = []
    flip_q: list[np.ndarray] = []
    flip_n: list[np.ndarray] = []

    def flush(shard_index: int) -> None:
        if not shard_q:
            return
        name = f"shard_{shard_index:05d}"
        np.save(args.out / f"{name}.npy", np.concatenate(shard_q))
        np.save(args.out / f"{name}_norm.npy", np.concatenate(shard_n))
        if args.with_flip:
            np.save(args.out / f"{name}_flip.npy", np.concatenate(flip_q))
            np.save(args.out / f"{name}_flip_norm.npy", np.concatenate(flip_n))
        index["shards"].append(name)
        shard_q.clear()
        shard_n.clear()
        flip_q.clear()
        flip_n.clear()

    def encode(pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        batch = torch.from_numpy(pixels).permute(0, 3, 1, 2).to(device).float().div_(255)
        batch = (batch - mean) / std
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            output = model(batch).last_hidden_state
        # Drop CLS and register tokens: the patch grid is the trailing `tokens` rows.
        return quantize(output[:, output.shape[1] - tokens :, :].float())

    written = 0
    for start in range(0, len(files), args.batch):
        chunk = files[start : start + args.batch]
        images = np.stack([letterbox(cv2.imread(str(p))[:, :, ::-1], args.size) for p in chunk])

        quantized, norms = encode(images)
        shard_q.append(quantized)
        shard_n.append(norms)
        if args.with_flip:
            # A second forward, not a mirrored grid: see the module docstring.
            fq, fn = encode(np.ascontiguousarray(images[:, :, ::-1]))
            flip_q.append(fq)
            flip_n.append(fn)
        index["files"].extend(p.name for p in chunk)
        written += len(chunk)

        if sum(len(q) for q in shard_q) >= args.shard_size:
            flush(len(index["shards"]))

        if written % (args.batch * 20) == 0 or written == len(files):
            rate = written / (time.perf_counter() - started)
            print(f"  {written}/{len(files)}  {rate:.1f} img/s  "
                  f"eta {(len(files) - written) / rate / 60:.1f} min", flush=True)

    flush(len(index["shards"]))
    (args.out / "index.json").write_text(json.dumps(index))
    print(f"\n{written} images in {len(index['shards'])} shards -> {args.out}")


if __name__ == "__main__":
    main()
