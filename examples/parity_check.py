"""Verify modern-yolonas reproduces super-gradients' YOLO-NAS, layer for layer.

Loads ONE set of safetensors weights into both implementations and compares their
forward passes on identical input tensors. Because both sides get byte-identical
weights, the comparison isolates the architecture from weight loading.

Writes `output/parity.md` and `output/parity_summary.csv`.

Why the weights are side-loaded rather than downloaded
------------------------------------------------------
`models.get(..., pretrained_weights="coco")` fetches from `sghub.deci.ai`, which no
longer resolves in DNS. So we build the super-gradients architecture with
`num_classes=80` (no download) and push our safetensors into it.

Environment
-----------
super-gradients 3.7.1 does not install on Python 3.11+ (it pins onnxruntime==1.15.0,
whose cp311 wheel needs numpy>=1.24.2 while super-gradients pins numpy<=1.23), and it
does not import under setuptools>=81 (it uses `pkg_resources`). So:

    uv venv --python 3.10 .venv-parity
    VIRTUAL_ENV=.venv-parity uv pip install super-gradients==3.7.1 'setuptools<81' \
        safetensors huggingface-hub
    VIRTUAL_ENV=.venv-parity uv pip install --no-deps -e .
    .venv-parity/bin/python examples/parity_check.py

modern-yolonas requires Python 3.10+, so it imports and runs fine in that env.

Note: the pretrained COCO weights are under the Super Gradients Model EULA
(non-commercial). See src/modern_yolonas/weights.py.
"""

from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

WEIGHT_FILES = {
    "yolo_nas_s": "yolo-nas-s.safetensors",
    "yolo_nas_m": "yolo-nas-m.safetensors",
    "yolo_nas_l": "yolo-nas-l.safetensors",
}


def build_pair(variant: str, repo_id: str):
    """Return (ours, theirs, n_tensors, load_report) sharing one set of weights."""
    from super_gradients.training import models

    from modern_yolonas.model import YoloNAS

    sd = load_file(hf_hub_download(repo_id, filename=WEIGHT_FILES[variant]))

    ours = YoloNAS.from_config(variant, num_classes=80)
    o_missing, o_unexpected = ours.load_state_dict(sd, strict=False)
    ours.eval()

    theirs = models.get(variant, num_classes=80)  # architecture only, no download
    t_missing, t_unexpected = theirs.load_state_dict(sd, strict=False)
    theirs.eval()

    report = {
        "ours_missing": len(o_missing),
        "ours_unexpected": len(o_unexpected),
        "theirs_missing": len(t_missing),
        "theirs_unexpected": len(t_unexpected),
    }
    return ours, theirs, len(sd), report


def forward_pair(ours, theirs, x):
    """Run both and return (boxes, scores) from each, unwrapping SG's nested output."""
    with torch.no_grad():
        a_boxes, a_scores = ours(x)
        out = theirs(x)
        head = out[0] if isinstance(out[0], (list, tuple)) else out
        b_boxes, b_scores = head[0], head[1]
    return (a_boxes, a_scores), (b_boxes, b_scores)


def fusion_report(model) -> tuple[int, int]:
    """(number of RepVGG blocks, number currently fused) — both sides must match."""
    blocks = [m for m in model.modules() if hasattr(m, "fully_fused")]
    fused = sum(1 for m in blocks if getattr(m, "fully_fused", False))
    return len(blocks), fused


def compare(variant: str, repo_id: str, n_random: int, seed: int, dtype: torch.dtype):
    ours, theirs, n_tensors, load_report = build_pair(variant, repo_id)
    ours, theirs = ours.to(dtype), theirs.to(dtype)

    torch.manual_seed(seed)
    worst_box = worst_score = 0.0
    scale = 0.0
    for _ in range(n_random):
        x = torch.randn(1, 3, 640, 640).to(dtype)
        (ab, asc), (bb, bsc) = forward_pair(ours, theirs, x)
        worst_box = max(worst_box, (ab - bb).abs().max().item())
        worst_score = max(worst_score, (asc - bsc).abs().max().item())
        scale = max(scale, ab.abs().max().item())

    return {
        "variant": variant,
        "dtype": str(dtype).replace("torch.", ""),
        "inputs": n_random,
        "checkpoint_tensors": n_tensors,
        "repvgg_blocks": fusion_report(ours)[0],
        "repvgg_blocks_sg": fusion_report(theirs)[0],
        "worst_box_abs": worst_box,
        "worst_box_rel": worst_box / scale if scale else 0.0,
        "worst_score_abs": worst_score,
        "box_scale_px": scale,
        **load_report,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", default="yolo_nas_s,yolo_nas_m,yolo_nas_l")
    ap.add_argument("--repo-id", default="CondadosAI/detectors")
    ap.add_argument("--inputs", type=int, default=15, help="random tensors per variant")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--precision-sweep", action="store_true",
                    help="also run yolo_nas_s in float64 (slow, but distinguishes "
                         "rounding from a genuine formula difference)")
    ap.add_argument("--out", default="output")
    args = ap.parse_args()

    rows = []
    for variant in [v.strip() for v in args.variants.split(",")]:
        print(f"comparing {variant} (float32) ...", flush=True)
        rows.append(compare(variant, args.repo_id, args.inputs, args.seed, torch.float32))

    if args.precision_sweep:
        # Matched input counts, so the float32/float64 ratio is directly comparable.
        # float64 at 640x640 is slow, hence the smaller count on both sides.
        n = min(3, args.inputs)
        for dt in (torch.float32, torch.float64):
            print(f"precision sweep: yolo_nas_s ({dt}) over {n} inputs ...", flush=True)
            rows.append(compare("yolo_nas_s", args.repo_id, n, args.seed, dt))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with (out / "parity_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    lines = [
        "# Parity: modern-yolonas vs super-gradients",
        "",
        "Generated by `examples/parity_check.py`. Both implementations receive the same",
        "safetensors weights and the same input tensors, in one process on one torch build.",
        "",
        f"- torch: `{torch.__version__}`",
        f"- weights repo: `{args.repo_id}` (Super Gradients Model EULA, non-commercial)",
        f"- random inputs per variant: {args.inputs} (see the `inputs` column), seed {args.seed}, device CPU",
        "",
        "| variant | dtype | inputs | ckpt tensors | miss/unexp (ours) | miss/unexp (SG) | RepVGG blocks | worst box abs | worst box rel | worst score abs |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['variant']}` | {r['dtype']} | {r['inputs']} | {r['checkpoint_tensors']} | "
            f"{r['ours_missing']}/{r['ours_unexpected']} | {r['theirs_missing']}/{r['theirs_unexpected']} | "
            f"{r['repvgg_blocks']} vs {r['repvgg_blocks_sg']} | "
            f"{r['worst_box_abs']:.3e} | {r['worst_box_rel']:.3e} | {r['worst_score_abs']:.3e} |"
        )
    lines += [
        "",
        "`worst score abs` of exactly `0.000e+00` means the class scores are **bit-identical**,",
        "so the backbone, neck and classification branch agree exactly. Any remaining box",
        "difference is confined to the distance decode.",
        "",
        "If you ran `--precision-sweep`, compare the float32 and float64 rows for",
        "`yolo_nas_s` at the same input count: a relative error that collapses by ~2^28 is the signature of",
        "floating-point rounding rather than a genuine difference in the arithmetic, since",
        "extra precision does not repair a wrong formula.",
        "",
    ]
    (out / "parity.md").write_text("\n".join(lines))
    print(f"wrote {out/'parity.md'} and {out/'parity_summary.csv'}")


if __name__ == "__main__":
    main()
