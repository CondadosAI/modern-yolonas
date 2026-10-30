"""Collect `yolonas benchmark-tracking evaluate` results into one JSON and markdown tables.

Expects one evaluate output directory per group, each holding
``sportsmot-val-{oracle,detector}/results.json``:

    uv run examples/render_tracking_table.py --runs runs/mot-final \\
        --groups all,basketball,football,volleyball --output docs/benchmarks/tracking.json

Prints the tables that docs/benchmarks/tracking.md carries, so the page and the JSON
cannot drift apart.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ORDER = ["bytetrack", "motion-matched", "motion", "harmonic", "min",
         "motion-keepall", "harmonic-keepall", "ocsort"]
LABELS = {
    "bytetrack": "ByteTrack (roboflow/trackers)",
    "motion-matched": "Deep HM-SORT, motion only, ByteTrack's thresholds",
    "motion": "Deep HM-SORT, motion only",
    "harmonic": "Deep HM-SORT, harmonic (its default)",
    "min": "Deep HM-SORT, min fusion (Deep-EIoU)",
    "motion-keepall": "Deep HM-SORT, motion only, keep every tracklet",
    "harmonic-keepall": "Deep HM-SORT, harmonic, keep every tracklet",
    "ocsort": "OC-SORT (roboflow/trackers)",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--groups", default="all,basketball,football,volleyball")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    groups = [g.strip() for g in args.groups.split(",")]
    collected: dict = {}
    for group in groups:
        for source in ("oracle", "detector"):
            data = json.loads((args.runs / group / f"sportsmot-val-{source}" / "results.json").read_text())
            collected.setdefault(source, {})[group] = {"sequences": len(data["sequences"]),
                                                      "results": data["results"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(collected, indent=2))

    for source in ("detector", "oracle"):
        full = collected[source]["all"]
        print(f"\n### {source}, all {full['sequences']} sequences\n")
        print("| Tracker | HOTA | AssA | DetA | IDF1 | MOTA | ID switches |")
        print("|:---|---:|---:|---:|---:|---:|---:|")
        for name in ORDER:
            r = full["results"][name]
            print(f"| {LABELS[name]} | {r['HOTA']:.1f} | {r['AssA']:.1f} | {r['DetA']:.1f} | "
                  f"{r['IDF1']:.1f} | {r['MOTA']:.1f} | {r['IDSW']:.0f} |")
        print(f"\n### {source}, HOTA by sport\n")
        sports = [g for g in groups if g != "all"]
        print("| Tracker | " + " | ".join(f"{s} ({collected[source][s]['sequences']})" for s in sports) + " |")
        print("|:---|" + "---:|" * len(sports))
        for name in ORDER:
            print(f"| {LABELS[name]} | " + " | ".join(
                f"{collected[source][s]['results'][name]['HOTA']:.1f}" for s in sports) + " |")


if __name__ == "__main__":
    main()
