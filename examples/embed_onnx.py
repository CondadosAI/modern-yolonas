"""Example: run an exported embedding graph with ONNX Runtime.

Export one first:

    yolonas export --model yolo_nas_s --target embedding --output embedding.onnx
    yolonas export --model yolo_nas_s --target combined  --output combined.onnx

Then:

    uv run examples/embed_onnx.py embedding.onnx query.jpg gallery/
    uv run examples/embed_onnx.py combined.onnx query.jpg gallery/ --top-k 10

The graph takes two inputs. `images` is the usual letterboxed NCHW batch;
`valid_region` is [B, 4] int64 `(left, top, right, bottom)` saying where the real
pixels sit inside that letterbox. It cannot be baked into the graph because it
depends on the image's aspect ratio, and dropping it makes embeddings cluster by
aspect ratio instead of by content — so `preprocess` and `valid_region` are used
together here, exactly as the PyTorch path does internally.
"""

import argparse

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from modern_yolonas.inference.embed import valid_region
from modern_yolonas.inference.preprocess import preprocess

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def embed(session: ort.InferenceSession, paths: list[Path], input_size: int) -> np.ndarray:
    """Embed a batch of image files with an exported graph."""
    tensors, regions = [], []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            raise SystemExit(f"Cannot read image: {path}")
        tensor, scale, pad = preprocess(image, input_size)
        tensors.append(tensor.numpy())
        regions.append(valid_region(image, scale, pad))

    feeds = {
        "images": np.concatenate(tensors),
        "valid_region": np.array(regions, dtype=np.int64),
    }
    outputs = session.run(None, feeds)

    # The embedding is the only output of the `embedding` target and the last of
    # the `combined` one, so naming it beats indexing by position.
    index = [o.name for o in session.get_outputs()].index("embedding")
    return outputs[index]


def main():
    parser = argparse.ArgumentParser(description="Image retrieval with an exported ONNX embedding graph")
    parser.add_argument("model", help="Exported .onnx (target embedding or combined)")
    parser.add_argument("query", help="Query image")
    parser.add_argument("gallery", help="Directory of images to search")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])

    # The graph is exported at a fixed canvas; only the batch axis is dynamic.
    input_size = session.get_inputs()[0].shape[2]
    print(f"{Path(args.model).name}: {input_size}x{input_size} canvas")

    gallery = sorted(p for p in Path(args.gallery).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not gallery:
        raise SystemExit(f"No images found under {args.gallery}")

    vectors = np.concatenate(
        [embed(session, gallery[i : i + args.batch_size], input_size) for i in range(0, len(gallery), args.batch_size)]
    )
    print(f"Embedded {len(gallery)} gallery images -> {vectors.shape[1]}-d")

    # The graph L2-normalizes unless it was exported with --no-normalize, so the
    # dot product is already the cosine similarity.
    similarity = vectors @ embed(session, [Path(args.query)], input_size)[0]
    ranking = np.argsort(-similarity)[: args.top_k]

    print(f"\nMost similar to {args.query}:")
    for rank, index in enumerate(ranking, start=1):
        print(f"  {rank}. {similarity[index]:.4f}  {gallery[index]}")


if __name__ == "__main__":
    main()
