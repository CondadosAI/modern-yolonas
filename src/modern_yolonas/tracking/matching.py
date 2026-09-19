"""Cost matrices and assignment for :mod:`modern_yolonas.tracking`.

Every function here is pure numpy over ``(x1, y1, x2, y2)`` boxes and ``(N, D)``
feature blocks, so they can be tested — and reused — without a tracker around them.
"""

from __future__ import annotations

import numpy as np

from scipy.optimize import linear_sum_assignment

#: Cost assigned to a pair that a gate has ruled out. Anything at or above this is
#: refused by :func:`linear_assignment` for every threshold the trackers use.
GATED = 1.0


def expand_boxes(boxes: np.ndarray, expansion: float) -> np.ndarray:
    """Grow each box outwards by ``expansion`` times its own width and height.

    Each side moves out by ``expansion * w`` (or ``* h``), so the expanded box is
    ``(1 + 2 * expansion)`` times as wide and as tall, centred on the original.
    ``expansion=0`` returns the box unchanged.

    This is ExpansionIoU's remedy for fast motion: two boxes of the same object in
    consecutive frames can have no overlap at all, which a plain IoU cost reads as
    "different objects". Expanding both before intersecting restores a usable
    signal without expanding the detector's own boxes.

    .. note::
       The reference Deep-EIoU implementation's ``expand()`` adds half the
       *expanded* width to each side rather than half the *increase*, so for the
       same ``e`` its boxes end up ``2 * (1 + e)`` times as wide — larger than the
       ``(1 + 2e)`` the name implies. This uses the plain reading. It matters when
       porting a tuned ``expansion`` value across the two.

    Args:
        boxes: ``(N, 4)`` in ``(x1, y1, x2, y2)``.
        expansion: Non-negative growth factor.

    Returns:
        ``(N, 4)``, a new array.
    """
    if expansion < 0:
        raise ValueError(f"expansion must be >= 0, got {expansion}")
    boxes = np.asarray(boxes, dtype=np.float32)
    if len(boxes) == 0:
        return boxes.reshape(0, 4).copy()

    margin_x = (boxes[:, 2] - boxes[:, 0]) * expansion
    margin_y = (boxes[:, 3] - boxes[:, 1]) * expansion
    grown = boxes.copy()
    grown[:, 0] -= margin_x
    grown[:, 2] += margin_x
    grown[:, 1] -= margin_y
    grown[:, 3] += margin_y
    return grown


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of boxes.

    Args:
        a: ``(M, 4)`` in ``(x1, y1, x2, y2)``.
        b: ``(N, 4)`` in ``(x1, y1, x2, y2)``.

    Returns:
        ``(M, N)`` in ``[0, 1]``.
    """
    a = np.asarray(a, dtype=np.float32).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float32).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)

    top_left = np.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(bottom_right - top_left, 0.0, None)
    overlap = wh[..., 0] * wh[..., 1]

    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - overlap

    return np.where(union > 0, overlap / np.maximum(union, 1e-12), 0.0).astype(np.float32)


def expansion_iou_distance(a: np.ndarray, b: np.ndarray, expansion: float) -> np.ndarray:
    """``1 - IoU`` between the two box sets after both are expanded.

    Args:
        a: ``(M, 4)`` boxes, usually the tracks' last known positions.
        b: ``(N, 4)`` boxes, usually this frame's detections.
        expansion: Passed to :func:`expand_boxes`; both sides are expanded.

    Returns:
        ``(M, N)`` in ``[0, 1]``, 0 for a perfect overlap.
    """
    return (1.0 - box_iou(expand_boxes(a, expansion), expand_boxes(b, expansion))).astype(np.float32)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise cosine distance, rescaled to ``[0, 1]``.

    Cosine distance ``1 - cos`` lives in ``[0, 2]`` for vectors that may point
    opposite ways. Halving it puts it on the same scale as the IoU distance it is
    about to be fused with, which is what makes a single threshold — and a
    harmonic mean — meaningful across both.

    Rows are L2-normalized here, so a caller that skipped normalization, or an EMA
    that drifted off the unit sphere, still gets a true cosine. A zero row gives
    distance 0.5 against everything (no information), not a NaN.

    Args:
        a: ``(M, D)``.
        b: ``(N, D)``.

    Returns:
        ``(M, N)`` in ``[0, 1]``.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"embedding width mismatch: {a.shape[1]} vs {b.shape[1]}")

    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return np.clip((1.0 - a @ b.T) / 2.0, 0.0, 1.0).astype(np.float32)


def harmonic_mean(d1: np.ndarray, d2: np.ndarray) -> np.ndarray:
    """Element-wise harmonic mean of two cost matrices.

    ``2 * d1 * d2 / (d1 + d2)``, the form Deep HM-SORT puts in place of
    Deep-EIoU's ``min(d1, d2)``.

    The harmonic mean leans towards the smaller of the two, so a strong cue still
    carries the pair — but unlike ``min`` it does not *ignore* the other one. Two
    teammates in the same kit have near-identical embeddings; ``min`` will happily
    match the wrong one on appearance alone, while the harmonic mean lets the
    motion cost veto it. That is the paper's whole argument for the swap, and its
    ablation is where the ID-switch count drops.

    Both inputs are expected in ``[0, 1]``. A zero anywhere makes the result zero,
    including ``0`` against ``0``, which the plain quotient would make a NaN.

    Args:
        d1: ``(M, N)`` cost matrix, e.g. expansion-IoU distance.
        d2: ``(M, N)`` cost matrix, e.g. cosine distance.

    Returns:
        ``(M, N)``.
    """
    d1 = np.asarray(d1, dtype=np.float32)
    d2 = np.asarray(d2, dtype=np.float32)
    total = d1 + d2
    return np.where(total > 0, 2.0 * d1 * d2 / np.maximum(total, 1e-12), 0.0).astype(np.float32)


def linear_assignment(
    cost: np.ndarray, threshold: float
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Hungarian assignment, with pairs costlier than ``threshold`` refused.

    Uses ``scipy.optimize.linear_sum_assignment`` — an optimal solver over the
    *whole* matrix — and then drops the pairs that came back above ``threshold``.
    Gated entries are raised above every candidate first so the solver routes
    around them where it can; a gated pair that survives that is still dropped.

    Args:
        cost: ``(M, N)``.
        threshold: Highest cost still considered a match.

    Returns:
        ``(matches, unmatched_rows, unmatched_cols)``, where ``matches`` is a list
        of ``(row, col)``.
    """
    cost = np.asarray(cost, dtype=np.float32)
    rows, cols = cost.shape if cost.ndim == 2 else (0, 0)
    if rows == 0 or cols == 0:
        return [], list(range(rows)), list(range(cols))

    # A cost above the threshold can never become a match, so make it maximally
    # unattractive rather than merely expensive — otherwise the optimal total can
    # be reached by taking one impossible pair to cheapen another.
    solvable = np.where(cost > threshold, threshold + 1.0 + np.max(cost), cost)
    row_idx, col_idx = linear_sum_assignment(solvable)

    matches = [(int(r), int(c)) for r, c in zip(row_idx, col_idx) if cost[r, c] <= threshold]
    matched_rows = {r for r, _ in matches}
    matched_cols = {c for _, c in matches}
    return (
        matches,
        [r for r in range(rows) if r not in matched_rows],
        [c for c in range(cols) if c not in matched_cols],
    )


def fuse_costs(
    iou_distance: np.ndarray,
    embedding_distance: np.ndarray | None,
    *,
    proximity_threshold: float,
    appearance_threshold: float,
    fusion: str = "harmonic",
) -> np.ndarray:
    """Combine a motion cost and an appearance cost into the association cost.

    Two gates run before the fusion, both from BoT-SORT by way of Deep-EIoU:

    * an appearance cost above ``appearance_threshold`` is not evidence of
      anything — same-team players sit well inside it — so it is discarded;
    * an appearance cost is not *trusted* for a pair that is nowhere near each
      other geometrically, so an IoU distance above ``proximity_threshold``
      discards it too. Without this, a track can be re-attached across the frame
      by appearance alone.

    A pair whose appearance cost is discarded falls back to the motion cost
    **alone**, rather than being fused with a sentinel. That is a decision the
    paper does not make for us, and it is not cosmetic: ``harmonic_mean(d, 1)`` is
    ``2d / (d + 1)``, which is *larger* than ``d`` for every ``d < 1``. Fusing
    against a placeholder would therefore charge a pair for evidence that was
    never available — and since the gate fires on exactly the distant pairs the
    expansion scale-up exists to reach, it would quietly cancel the scale-up
    (``match_threshold`` 0.8 would start refusing at an IoU distance of 0.67).
    Deep-EIoU's ``min`` has no such effect, so falling back also keeps the two
    fusions comparable where only one cue exists, which is what makes an ablation
    between them mean anything.

    Args:
        iou_distance: ``(M, N)`` expansion-IoU distance.
        embedding_distance: ``(M, N)`` cosine distance, or ``None`` to associate
            on motion alone. Individual columns may be all-``NaN`` to mark
            detections that arrived without an embedding; those fall back to
            motion too.
        proximity_threshold: IoU-distance gate on the appearance cost.
        appearance_threshold: Cosine-distance gate on the appearance cost.
        fusion: ``"harmonic"`` for Deep HM-SORT, ``"min"`` for Deep-EIoU's
            original ``min(d1, d2)``. The second exists so the difference can be
            measured rather than asserted.

    Returns:
        ``(M, N)`` association cost.
    """
    if fusion not in ("harmonic", "min"):
        raise ValueError(f"fusion must be 'harmonic' or 'min', got {fusion!r}")
    if embedding_distance is None or embedding_distance.size == 0:
        return iou_distance.astype(np.float32)

    appearance = np.array(embedding_distance, dtype=np.float32, copy=True)
    usable = (
        np.isfinite(appearance)
        & (appearance <= appearance_threshold)
        & (iou_distance <= proximity_threshold)
    )
    appearance[~usable] = GATED

    fused = (
        np.minimum(iou_distance, appearance)
        if fusion == "min"
        else harmonic_mean(iou_distance, appearance)
    )
    return np.where(usable, fused, iou_distance).astype(np.float32)
