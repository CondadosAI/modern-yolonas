"""Can the detector's per-box embedding tell two players apart?

Reads the oracle caches written by ``yolonas benchmark-tracking cache --source
oracle``, recovers the ground-truth id of every box from ``gt/gt.txt``, and
collects cosine distances for

* the same player 1 frame apart and 25 frames apart, and
* two different players in the same frame,

then reports how far the two distributions overlap. Deep HM-SORT treats an
appearance distance under ``appearance_threshold`` (0.3) as evidence of identity,
so the fraction of different-player pairs under 0.3 is the fraction that gate
lets through.

    uv run examples/tracking_separability.py --data ~/datasets/sportsmot/extracted/val \\
        --cache runs/mot-cache/oracle --splits ~/datasets/sportsmot/splits_txt \\
        --output docs/benchmarks/tracking_separability.json

``--splits`` is optional; with it, results are also broken down by the sport
split files SportsMOT ships (basketball.txt, football.txt, volleyball.txt).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

GATE = 0.3
SAME_GAPS = {"same_1f": 1, "same_1s": 25}


def gt_ids(gt_path: Path, frames: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    """Ground-truth id for each oracle box, by frame and top-left corner; -1 if none."""
    gt = np.loadtxt(gt_path, delimiter=",", ndmin=2)
    key = {(int(f), round(x, 1), round(y, 1)): int(i) for f, i, x, y in gt[:, :4]}
    return np.array([key.get((int(f), round(float(b[0]), 1), round(float(b[1]), 1)), -1)
                     for f, b in zip(frames, xyxy)])


def cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a /= np.linalg.norm(a, axis=-1, keepdims=True)
    b /= np.linalg.norm(b, axis=-1, keepdims=True)
    return 1.0 - (a * b).sum(-1)


def sequence_distances(npz: Path, gt_path: Path) -> tuple[dict[str, list[float]], int, int]:
    data = np.load(npz)
    frames, xyxy, emb = data["frame"], data["xyxy"], data["embedding"]
    ids = gt_ids(gt_path, frames, xyxy)

    rows_by_frame: dict[int, dict[int, int]] = defaultdict(dict)
    for row, (frame, identity) in enumerate(zip(frames, ids)):
        if identity >= 0:
            rows_by_frame[int(frame)][int(identity)] = row

    out: dict[str, list[float]] = defaultdict(list)
    for frame, rows in rows_by_frame.items():
        for kind, gap in SAME_GAPS.items():
            later = rows_by_frame.get(frame + gap)
            if later:
                common = [i for i in rows if i in later]
                if common:
                    out[kind] += cosine_distance(emb[[rows[i] for i in common]],
                                                 emb[[later[i] for i in common]]).tolist()
        r = np.array(list(rows.values()))
        if len(r) > 1:
            a, b = np.triu_indices(len(r), 1)
            out["diff"] += cosine_distance(emb[r[a]], emb[r[b]]).tolist()
    return out, int((ids < 0).sum()), len(ids)


def summarise(pool: dict[str, list[float]], rng: np.random.Generator) -> dict:
    summary = {}
    for kind, values in pool.items():
        v = np.asarray(values)
        summary[kind] = {
            "n": int(len(v)),
            "median": float(np.median(v)),
            "p10": float(np.percentile(v, 10)),
            "p90": float(np.percentile(v, 90)),
            f"frac_below_{GATE}": float((v < GATE).mean()),
        }
    # Probability that a same-player (1 s) distance is smaller than a different-player
    # one: 0.5 is chance, 1.0 is perfect separation. Estimated on 20k draws of each.
    same, diff = np.asarray(pool["same_1s"]), np.asarray(pool["diff"])
    same = rng.choice(same, min(len(same), 20000), replace=False)
    diff = rng.choice(diff, min(len(diff), 20000), replace=False)
    summary["auc_same_1s_vs_diff"] = float((same[:, None] < diff[None, :]).mean())
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, type=Path, help="Split directory, one subdirectory per sequence.")
    parser.add_argument("--cache", required=True, type=Path, help="The oracle cache directory.")
    parser.add_argument("--splits", type=Path, help="Directory of per-group name lists, e.g. SportsMOT's splits_txt.")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    groups: dict[str, set[str]] = {}
    if args.splits:
        for name in ("basketball", "football", "volleyball"):
            path = args.splits / f"{name}.txt"
            if path.exists():
                groups[name] = set(path.read_text().split())

    rng = np.random.default_rng(0)
    pools: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    unmatched = total = 0
    for npz in sorted(args.cache.glob("*.npz")):
        distances, missed, count = sequence_distances(npz, args.data / npz.stem / "gt" / "gt.txt")
        unmatched += missed
        total += count
        group = next((g for g, names in groups.items() if npz.stem in names), None)
        for kind, values in distances.items():
            pools["all"][kind] += values
            if group:
                pools[group][kind] += values

    result = {
        "boxes": total,
        "matched_fraction": 1 - unmatched / total if total else 0.0,
        "gate": GATE,
        **{group: summarise(pool, rng) for group, pool in pools.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    for group in pools:
        s = result[group]
        print(f"{group:10}  AUC {s['auc_same_1s_vs_diff']:.3f}  "
              f"diff median {s['diff']['median']:.3f}  same_1s median {s['same_1s']['median']:.3f}  "
              f"diff below {GATE}: {s['diff'][f'frac_below_{GATE}']:.1%}")


if __name__ == "__main__":
    main()
