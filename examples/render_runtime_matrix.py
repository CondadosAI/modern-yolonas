"""Render `runtime_matrix.json` as a page someone can read in ten seconds.

Split from the benchmark so the presentation can change without re-running hours
of measurement, and so a merged JSON (the ONNX Runtime CUDA rows come from a second
environment) renders in one pass.

    uv run examples/render_runtime_matrix.py

The shape of the page follows the order the questions actually get asked: what is
the fastest I can go on the hardware I have, what does each choice cost me, and only
then the full grid. A flat fifty-row table answers the third question and buries the
first two.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

# The `device` field mixes what each runtime calls itself: PyTorch reports "CPU" and
# "dGPU", OpenVINO reports a marketing name, TensorRT appends its build variant.
# Three buckets is what a reader is choosing between.
HARDWARE = [
    (re.compile(r"iris|iGPU", re.I), "iGPU"),
    (re.compile(r"dGPU|NVIDIA|GeForce", re.I), "dGPU"),
    (re.compile(r"CPU|Intel\(R\) Core|Ryzen|i7-", re.I), "CPU"),
]

# Two series: input size. Slots 1 and 2 of the reference categorical palette,
# validated with scripts/validate_palette.js in both modes.
LIGHT = {"bg": "none", "ink": "#0b0b0b", "muted": "#52514e", "grid": "#e3e2df",
         "s1": "#2a78d6", "s2": "#eb6834"}
DARK = {"bg": "none", "ink": "#ffffff", "muted": "#c3c2b7", "grid": "#3a3a38",
        "s1": "#3987e5", "s2": "#d95926"}


def hardware(device: str) -> str:
    for pattern, label in HARDWARE:
        if pattern.search(device):
            return label
    return device


def config(row: dict) -> str:
    """How a reader would name this leg: runtime, precision, and any build variant."""
    variant = row["device"].split()[-1] if row["runtime"] == "TensorRT" else ""
    variant = f" {variant}" if variant in {"ampere_plus"} else ""
    return f"{row['runtime']}{variant} {row['precision'].upper()}"


# --------------------------------------------------------------------------------
# Chart: the fastest achievable on each piece of hardware, at each input size.
#
# Horizontal grouped bars on a linear scale from zero. The full matrix spans 0.9 ms
# to 200 ms, which no honest bar chart can hold — but the *best per device* spans
# well under an order of magnitude, which is exactly why this is the cut that gets
# drawn and the rest stays a table.
# --------------------------------------------------------------------------------
def svg(groups: list[tuple[str, dict]], sizes: list[int], colors: dict, model: str) -> str:
    bar_h, bar_gap, group_gap = 26, 4, 22
    # The right margin holds the value and the configuration that produced it. Both
    # go outside the bar: a label inside overflows as soon as a bar is short, and
    # white-on-background is invisible.
    left, right, top = 92, 340, 46
    width = 760
    plot_w = width - left - right

    height = top + sum(len(sizes) * bar_h + (len(sizes) - 1) * bar_gap + group_gap for _, _ in groups) + 16
    peak = max(entry["median_ms"] for _, by_size in groups for entry in by_size.values())
    scale = plot_w / (peak * 1.02)

    parts = [
        # The viewBox starts at -4 so the leftmost glyph is not clipped by the edge.
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="-4 0 {width + 4} {height + 6}" '
        f'width="{width + 4}" height="{height + 6}" font-family="system-ui,-apple-system,Segoe UI,sans-serif">',
        f'<style>text{{fill:{colors["ink"]}}} .m{{fill:{colors["muted"]}}}</style>',
        f'<text x="0" y="16" font-size="14" font-weight="600">Fastest achievable latency, {model}</text>',
        '<text x="0" y="34" font-size="11" class="m">milliseconds per frame, batch 1, single stream — FPS is its reciprocal, not throughput</text>',
    ]

    # Legend, top right. Two series always get one.
    for i, size in enumerate(sizes):
        parts.append(f'<rect x="{width - 108}" y="{8 + i * 18}" width="10" height="10" rx="2" '
                     f'fill="{colors["s1"] if i == 0 else colors["s2"]}"/>')
        parts.append(f'<text x="{width - 92}" y="{17 + i * 18}" font-size="11">{size}x{size}</text>')

    y = top
    for name, by_size in groups:
        block = len(sizes) * bar_h + (len(sizes) - 1) * bar_gap
        parts.append(f'<text x="0" y="{y + block / 2 + 4}" font-size="12" font-weight="600">{name}</text>')
        for i, size in enumerate(sizes):
            entry = by_size.get(size)
            if entry is None:
                y += bar_h + bar_gap
                continue
            w = max(entry["median_ms"] * scale, 3)
            parts.append(
                f'<rect x="{left}" y="{y}" width="{w:.1f}" height="{bar_h}" rx="4" '
                f'fill="{colors["s1"] if i == 0 else colors["s2"]}"/>'
            )
            value = f'{entry["median_ms"]:.2f} ms · {entry["fps"]:.0f} FPS'
            parts.append(
                f'<text x="{left + w + 8:.1f}" y="{y + bar_h / 2 + 4}" font-size="11" '
                f'font-weight="600">{value}</text>'
            )
            # Offset by the value's own width — a fixed gap collides as soon as the
            # number gains a digit.
            parts.append(
                f'<text x="{left + w + 26 + len(value) * 6.6:.1f}" y="{y + bar_h / 2 + 4}" '
                f'font-size="10.5" class="m">{config(entry)}</text>'
            )
            y += bar_h + bar_gap
        y += group_gap - bar_gap

    parts.append("</svg>")
    return "\n".join(parts)


def best_per_hardware(rows, model, nms="external"):
    """The quickest measured leg on each piece of hardware, per input size."""
    best = defaultdict(dict)
    for row in rows:
        if row["median_ms"] is None or row["model"] != model or row["nms"] != nms:
            continue
        slot = best[hardware(row["device"])]
        size = row["input"]
        if size not in slot or row["median_ms"] < slot[size]["median_ms"]:
            slot[size] = row
    return best


def pct(new: float, old: float) -> str:
    return f"{(new / old - 1) * 100:+.0f}%"


def cost_table(rows, model, sizes) -> list[str]:
    """What the choices that are not purely "go faster" actually cost."""
    index = {}
    for row in rows:
        if row["median_ms"] is not None and row["model"] == model:
            index[(row["runtime"], row["device"], row["precision"], row["input"], row["nms"])] = row["median_ms"]

    def get(runtime, device, precision, size, nms):
        return index.get((runtime, device, precision, size, nms))

    comparisons = [
        ("TensorRT `--hardware-compatible`", "an engine that loads on any sm_80+ GPU instead of only this one",
         lambda s: (get("TensorRT", "dGPU ampere_plus", "fp16", s, "external"),
                    get("TensorRT", "dGPU native", "fp16", s, "external"))),
        ("`--target end2end` vs torchvision NMS", "one self-contained file, no Python in the inference path",
         lambda s: (get("TensorRT", "dGPU native", "fp16", s, "graph"),
                    get("TensorRT", "dGPU native", "fp16", s, "torch"))),
        ("TensorRT FP16 vs FP32", "a speedup, not a trade — the AP cost is in the model table",
         lambda s: (get("TensorRT", "dGPU native", "fp16", s, "external"),
                    get("TensorRT", "dGPU native", "fp32", s, "external"))),
    ]

    cpu = next((r["device"] for r in rows if r["runtime"] == "OpenVINO" and hardware(r["device"]) == "CPU"), None)
    if cpu:
        comparisons.append(
            ("OpenVINO INT8 vs FP32, CPU", "also a speedup; whether it costs AP is measured separately",
             lambda s: (get("OpenVINO", cpu, "int8", s, "external"),
                        get("OpenVINO", cpu, "fp32", s, "external")))
        )

    lines = ["| choice | " + " | ".join(str(s) for s in sizes) + " | what it buys |",
             "|---|" + "---:|" * len(sizes) + "---|"]
    for label, why, pair in comparisons:
        cells = []
        for size in sizes:
            new, old = pair(size)
            cells.append(pct(new, old) if new and old else "—")
        if all(c == "—" for c in cells):
            continue
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {why} |")
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="docs/benchmarks/runtime_matrix.json")
    parser.add_argument("--output", default="docs/benchmarks/runtime_matrix.md")
    parser.add_argument("--chart-model", default="yolo_nas_s", help="Variant drawn in the headline chart.")
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text())
    env, rows = data["environment"], data["rows"]
    measured = [r for r in rows if r["median_ms"] is not None]
    skipped = [r for r in rows if r["median_ms"] is None]
    sizes = sorted({r["input"] for r in measured})
    models = sorted({r["model"] for r in measured})

    out = Path(args.output)
    best = best_per_hardware(measured, args.chart_model)
    order = [h for h in ("dGPU", "iGPU", "CPU") if h in best] + [h for h in best if h not in ("dGPU", "iGPU", "CPU")]
    groups = [(h, best[h]) for h in order]

    for suffix, colors in (("", LIGHT), ("-dark", DARK)):
        (out.parent / f"runtime_matrix{suffix}.svg").write_text(
            svg(groups, sizes, colors, args.chart_model.replace("yolo_nas_", "YOLO-NAS-").upper()) + "\n"
        )

    lines = [
        "# Latency: which runtime, on which device, at which size",
        "",
        "Generated by `examples/runtime_matrix.py`, rendered by `examples/render_runtime_matrix.py`.",
        "Every number is measured on the machine below. Nothing here is quoted.",
        "",
        "## The short answer",
        "",
        '<picture>',
        '  <source media="(prefers-color-scheme: dark)" srcset="runtime_matrix-dark.svg">',
        '  <img src="runtime_matrix.svg" alt="Fastest achievable latency per device, at 320 and 640">',
        '</picture>',
        "",
    ]

    lines.append("| hardware | " + " | ".join(f"fastest at {s} | FPS" for s in sizes) + " |")
    lines.append("|---|" + "---|---:|" * len(sizes))
    for name, by_size in groups:
        cells = []
        for size in sizes:
            entry = by_size.get(size)
            if entry:
                cells += [f"**{entry['median_ms']:.2f} ms** — {config(entry)}", f"**{entry['fps']:.0f}**"]
            else:
                cells += ["—", "—"]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    lines += [
        "",
        f"{args.chart_model.replace('yolo_nas_', 'YOLO-NAS-').upper()}, model inference only, NMS left to the caller.",
        "The other variants are in the per-device tables below.",
        "",
        "**FPS here is `1000 / latency` on a single synchronous stream, which is not**",
        "**throughput.** A pipeline that overlaps decode, transfer and inference across",
        "streams reports a higher number on the same hardware; one that does none of that",
        "reports a lower one, because these figures exclude preprocessing.",
        "",
        "## What each choice costs",
        "",
        "Each percentage is against the alternative named in the row, on the same",
        "hardware. **A positive number is a slowdown you are choosing to accept** for",
        "the reason in the last column; a negative one is a straight win.",
        "",
    ]
    lines += cost_table(measured, args.chart_model, sizes)

    lines += ["", "## Per device", ""]
    by_hw = defaultdict(list)
    for row in measured:
        by_hw[hardware(row["device"])].append(row)

    for name in order:
        lines += [f"### {name}", ""]
        device_names = sorted({r["device"] for r in by_hw[name]})
        if len(device_names) > 1 or device_names[0] != name:
            lines += ["<sub>" + " · ".join(f"`{d}`" for d in device_names) + "</sub>", ""]
        lines.append("| runtime | precision | NMS | " + " | ".join(f"{s} ms | {s} FPS" for s in sizes) + " |")
        lines.append("|---|---|---|" + "---:|---:|" * len(sizes))

        cells = defaultdict(dict)
        for row in by_hw[name]:
            if row["model"] != args.chart_model:
                continue
            cells[(config(row), row["nms"])][row["input"]] = row["median_ms"]
        for (label, nms), by_size in sorted(cells.items(), key=lambda kv: min(kv[1].values())):
            runtime, precision = label.rsplit(" ", 1)
            values = " | ".join(
                f"{by_size[s]:.2f} | {1000 / by_size[s]:.0f}" if s in by_size else "— | —" for s in sizes
            )
            lines.append(f"| {runtime} | {precision} | {nms} | {values} |")
        lines.append("")

    lines += [
        "## How this was measured",
        "",
        "Batch 1, single stream, synchronous. Median of 30 runs after 8 warmup runs",
        "discarded whole. GPU work is synchronised before the clock is read. Latency,",
        "not throughput — an async multi-stream pipeline reports higher FPS on the same",
        "hardware.",
        "",
        "The **NMS** column says what is included:",
        "",
        "- `external` — the model only. Returns raw `[N, 4]` + `[N, 80]` tensors and",
        "  leaves NMS to the caller, so this number is not a whole detection.",
        "- `torch` — the model plus `postprocess`, torchvision's batched NMS, on the",
        "  tensors where they already are. On a CUDA leg nothing crosses to the host.",
        "- `graph` — `--target end2end`, with NonMaxSuppression inside the graph.",
        "",
        "`torch` and `graph` are the two comparable ones. `external` is there because it",
        "is what every published latency figure for this kind of model actually reports.",
        "",
        "The input is a real COCO frame with 62 annotated objects, not random noise:",
        "nothing in noise clears a 0.25 score, so every NMS leg would sort an empty list",
        "and report itself free.",
        "",
        "**Machine.**",
        "",
    ]
    for key in ("cpu", "gpu", "driver", "platform", "power_source", "python", "torch",
                "onnxruntime", "openvino", "tensorrt", "input"):
        if key in env:
            lines.append(f"- `{key}`: {env[key]}")
    lines += [
        "",
        "The power source is recorded because it changes the answer: this laptop GPU is",
        "power-capped on battery, and the same pass runs materially slower there. The SM",
        "clock is sampled at the end of each timed run rather than at startup, where an",
        "idle card reports a clock it was not running at.",
        "",
        "<details>",
        "<summary><b>Everything measured</b> — all variants, all legs</summary>",
        "",
        "| model | runtime | device | precision | NMS | input | median ms | FPS |",
        "|---|---|---|---|---|---:|---:|---:|",
    ]
    for row in sorted(measured, key=lambda r: (models.index(r["model"]), r["input"], r["median_ms"])):
        lines.append(
            f"| {row['model']} | {row['runtime']} | {row['device']} | {row['precision'].upper()} | "
            f"{row['nms']} | {row['input']} | {row['median_ms']:.2f} | {row['fps']:.1f} |"
        )
    lines += ["", "</details>", ""]

    if skipped:
        lines += [
            "<details>",
            "<summary><b>Not measured</b> — recorded rather than dropped</summary>",
            "",
            "A runtime missing from the environment is information; a silent gap is not.",
            "",
        ]
        for row in sorted(skipped, key=lambda r: (r["runtime"], r["device"], r["precision"], r["input"])):
            lines.append(
                f"- **{row['runtime']} / {row['device']} / {row['precision'].upper()} / "
                f"{row['input']} / nms={row['nms']}** — {row['error']}"
            )
        lines += ["", "</details>", ""]

    lines += [
        "## Reproducing",
        "",
        "```bash",
        "uv sync --dev --extra onnx --extra openvino --extra tensorrt",
        "uv run examples/runtime_matrix.py --models yolo_nas_s --sizes 320,640 \\",
        "    --nms external,torch,graph --calibration-dir ~/datasets/coco/images/val2017 \\",
        "    --sample-image ~/datasets/coco/images/val2017/000000435081.jpg",
        "",
        "# ONNX Runtime's CUDA provider needs the other wheel, which cannot coexist with",
        "# the CPU one, so its rows come from a second pass that merges into the same JSON:",
        "uv sync --dev --extra onnx-gpu --extra tensorrt",
        "uv run examples/runtime_matrix.py --runtimes ort --sizes 320,640",
        "",
        "uv run examples/render_runtime_matrix.py",
        "```",
        "",
        "## On comparing this with published leaderboards",
        "",
        "Detection leaderboards usually report TensorRT latency on an NVIDIA T4 at batch 1.",
        "That is not the hardware above, and a latency figure means nothing without its",
        "protocol, so the two should not be read side by side.",
    ]

    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out} ({len(measured)} measured, {len(skipped)} not) and 2 SVGs")


if __name__ == "__main__":
    main()
