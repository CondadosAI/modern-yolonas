"""Example: image retrieval with YOLO-NAS feature embeddings.

Embeds a folder of images with the backbone (the detection head is never run),
then ranks them against a query image by cosine similarity.

Usage:
    uv run examples/embed_image.py query.jpg images/
    uv run examples/embed_image.py query.jpg images/ --model yolo_nas_l --layers c4 c5
    uv run examples/embed_image.py query.jpg images/ --device cpu --top-k 10
"""

import argparse

from pathlib import Path

import numpy as np

from modern_yolonas import YoloNASEmbedder

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def main():
    parser = argparse.ArgumentParser(description="YOLO-NAS embedding-based image retrieval")
    parser.add_argument("query", help="Query image")
    parser.add_argument("gallery", help="Directory of images to search")
    parser.add_argument("--model", default="yolo_nas_s", choices=["yolo_nas_s", "yolo_nas_m", "yolo_nas_l"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layers", nargs="+", default=["c5"], help="Feature maps to pool (default: c5)")
    parser.add_argument("--pooling", default="avg", choices=["avg", "max"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    gallery = sorted(p for p in Path(args.gallery).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not gallery:
        raise SystemExit(f"No images found under {args.gallery}")

    embedder = YoloNASEmbedder(args.model, device=args.device, layers=args.layers, pooling=args.pooling)
    print(f"{args.model}: {embedder.embedding_dim}-d embeddings from {'+'.join(args.layers)}")

    # Embed in batches so a large gallery does not have to fit in memory at once.
    vectors = np.concatenate(
        [embedder.embed_batch(gallery[i : i + args.batch_size]) for i in range(0, len(gallery), args.batch_size)]
    )
    print(f"Embedded {len(gallery)} gallery images")

    # Both sides are L2-normalized, so the dot product is the cosine similarity.
    similarity = vectors @ embedder(args.query)
    ranking = np.argsort(-similarity)[: args.top_k]

    print(f"\nMost similar to {args.query}:")
    for rank, index in enumerate(ranking, start=1):
        print(f"  {rank}. {similarity[index]:.4f}  {gallery[index]}")


if __name__ == "__main__":
    main()
