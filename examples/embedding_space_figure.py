"""Generate the README's embedding-space figure.

One query and two gallery images, embedded with the YOLO-NAS backbone, plotted in
a 2D space whose distances are the *true* angular distances between the vectors.

With three points that is exact, not an approximation: any three points define a
triangle, and a triangle embeds in the plane without distortion. So the picture
is the geometry, not a projection of it — no UMAP, no PCA, nothing to mislead.

Needs matplotlib, which the dev group carries:

    uv sync --dev

Usage:
    uv run examples/embedding_space_figure.py
    uv run examples/embedding_space_figure.py --model yolo_nas_l --output out.png
"""

import argparse

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from modern_yolonas import YoloNASEmbedder

ASSETS = Path(__file__).resolve().parent.parent / "docs" / "assets"

QUERY = ASSETS / "street.jpg"
GALLERY = [ASSETS / "street_nyc.jpg", ASSETS / "pancakes.jpg"]
CAPTIONS = {
    "street.jpg": "query\nstreet in Porlamar",
    "street_nyc.jpg": "street in New York",
    "pancakes.jpg": "pancakes",
}

INK = "#1b1b1f"
MUTED = "#6b6b76"
HIT = "#5b3fd6"
MISS = "#b9b9c4"


def _thumb(path: Path, width: int = 260) -> np.ndarray:
    """RGB thumbnail, for drawing at a point in the plot."""
    image = cv2.imread(str(path))
    height = int(round(image.shape[0] * width / image.shape[1]))
    return cv2.cvtColor(cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)


def _thumb_zoom(ax, points: np.ndarray, thumb_px: int, cap: float = 0.62) -> float:
    """Thumbnail scale that keeps the closest pair apart, capped.

    The layout is fixed by the data, so the *pictures* are what has to give: this
    measures the closest pair in rendered pixels and sizes the thumbnails to a
    fraction of it, instead of guessing a zoom that collides whenever two images
    happen to be similar. The cap stops far-apart points from inflating the
    pictures until they dominate the figure.

    ``OffsetImage`` applies its own ``dpi / 72`` correction, so a zoom of 1 draws
    one image pixel per *point*, not per pixel — hence the factor here and the
    matching one where labels are offset.
    """
    ax.figure.canvas.draw()
    display = ax.transData.transform(points)
    gaps = [
        np.hypot(*(display[i] - display[j]))
        for i in range(len(points))
        for j in range(i + 1, len(points))
    ]
    drawn_px_per_zoom = thumb_px * ax.figure.dpi / 72.0
    return min(min(gaps) * 0.42 / drawn_px_per_zoom, cap)


def _triangle(similarity: np.ndarray) -> np.ndarray:
    """Place three vectors in 2D so every pairwise distance is exact.

    Uses angular distance, ``arccos`` of the cosine similarity, which is a proper
    metric on the sphere — so the triangle inequality holds and the layout below
    always closes. The query goes at the origin, the first gallery image on the
    x-axis, and the third point is where the two remaining distances intersect.
    """
    distance = np.arccos(np.clip(similarity, -1.0, 1.0))
    a, b, c = distance[0, 1], distance[0, 2], distance[1, 2]

    x = (a**2 + b**2 - c**2) / (2 * a)
    y = np.sqrt(max(b**2 - x**2, 0.0))
    return _landscape(np.array([[0.0, 0.0], [a, 0.0], [x, y]]))


def _landscape(points: np.ndarray) -> np.ndarray:
    """Rotate the layout so its longest axis is horizontal, query on the left.

    The embedding is only defined up to rotation and reflection — turning it
    changes no distance — so it may as well be turned to fit a page instead of
    leaving half the figure empty.
    """
    centred = points - points.mean(axis=0)
    # Principal axis of the three points; rotate it onto x.
    _, _, basis = np.linalg.svd(centred, full_matrices=False)
    rotated = centred @ basis.T

    if rotated[0, 0] > rotated[1:, 0].mean():   # keep the query on the left
        rotated[:, 0] *= -1
    if rotated[2, 1] < 0:                        # and the odd one out on top
        rotated[:, 1] *= -1
    return rotated


def main():
    parser = argparse.ArgumentParser(description="Render the embedding-space figure")
    parser.add_argument("--model", default="yolo_nas_s", choices=["yolo_nas_s", "yolo_nas_m", "yolo_nas_l"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default=str(ASSETS / "embedding_space.png"))
    args = parser.parse_args()

    paths = [QUERY, *GALLERY]
    embedder = YoloNASEmbedder(args.model, device=args.device)
    vectors = embedder.embed_batch(paths)

    # Both sides are L2-normalized, so this is the cosine similarity.
    similarity = vectors @ vectors.T
    points = _triangle(similarity)
    to_query = similarity[0, 1:]
    nearest = int(np.argmax(to_query)) + 1

    fig = plt.figure(figsize=(12.0, 5.6))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.01, 0.03, 0.98, 0.70])
    ax.set_facecolor("white")

    # A 1:1 aspect is not negotiable — the distances would otherwise be a lie —
    # so the limits are padded to the figure's shape instead.
    span = points.max(axis=0) - points.min(axis=0)
    pad_x = 0.12 * span[0]
    pad_y = 0.45 * span[1]
    ax.set_xlim(points[:, 0].min() - pad_x, points[:, 0].max() + pad_x)
    ax.set_ylim(points[:, 1].min() - pad_y, points[:, 1].max() + pad_y * 0.6)
    ax.set_aspect("equal")
    ax.set_axis_off()

    thumbs = [_thumb(path) for path in paths]
    zoom = _thumb_zoom(ax, points, thumbs[0].shape[1])

    # ---- edges from the query, labelled with the cosine --------------------
    for i in (1, 2):
        is_hit = i == nearest
        ax.plot(
            [points[0, 0], points[i, 0]], [points[0, 1], points[i, 1]],
            color=HIT if is_hit else MISS, lw=3.0 if is_hit else 1.6,
            zorder=1, solid_capstyle="round",
        )
        ax.annotate(
            f"{to_query[i - 1]:.2f}", (points[0] + points[i]) / 2,
            textcoords="offset points", xytext=(0, 0),
            ha="center", va="center", fontsize=14 if is_hit else 12,
            fontweight="bold" if is_hit else "normal",
            color=HIT if is_hit else MUTED, zorder=5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.95),
        )

    # ---- the images, at their true relative positions ----------------------
    for i, (path, thumb) in enumerate(zip(paths, thumbs)):
        # Half the drawn height, in points — the unit annotation offsets use.
        half = 0.5 * thumb.shape[0] * zoom
        highlighted = i in (0, nearest)
        box = AnnotationBbox(
            OffsetImage(thumb, zoom=zoom), points[i], frameon=True, pad=0.0, zorder=4,
        )
        box.patch.set_edgecolor(HIT if highlighted else MISS)
        box.patch.set_linewidth(3.5 if highlighted else 1.5)
        ax.add_artist(box)

        # Offsets in points, measured off the thumbnail's rendered half-height, so
        # a label can never land on a picture whatever the layout turns out to be.
        ax.annotate(
            CAPTIONS[path.name], points[i], textcoords="offset points", xytext=(0, -half - 10),
            ha="center", va="top", fontsize=11.5,
            color=INK if highlighted else MUTED,
            fontweight="bold" if i == 0 else "normal", zorder=5, linespacing=1.45,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.92),
        )

    ax.annotate(
        "nearest neighbour", points[nearest], textcoords="offset points",
        xytext=(0, 0.5 * thumbs[nearest].shape[0] * zoom + 10),
        ha="center", va="bottom", fontsize=11, color=HIT, fontweight="bold", zorder=5,
    )

    fig.text(
        0.015, 0.97, "One query, two gallery images, and the space they land in",
        fontsize=15, fontweight="bold", color=INK, va="top",
    )
    fig.text(
        0.015, 0.90,
        f"{embedder.embedding_dim}-d vectors from the {args.model.replace('_', '-').upper()} backbone — the detection head never runs.\n"
        "Distances are proportional to the true angular distances between those vectors: with three points\n"
        "that is exact, not a projection. Numbers are cosine similarity to the query.",
        fontsize=10, color=MUTED, va="top", linespacing=1.7,
    )

    # `tight` trims whatever slack the equal-aspect axes leaves at the sides,
    # which varies with the layout and is not worth predicting.
    fig.savefig(args.output, dpi=150, facecolor="white", bbox_inches="tight", pad_inches=0.25)
    print(f"wrote {args.output}")
    for path, score in zip(GALLERY, to_query):
        print(f"  cos(query, {path.name}) = {score:.4f}")


if __name__ == "__main__":
    main()
