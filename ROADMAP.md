# Roadmap

What is planned and what is deliberately not. Items are ordered by what blocks the most
downstream work, not by size. Dates are when the item was raised.

Current state: `main` is released (v0.4.0, MIT). `dev` carries Lightning training,
quantization, dataset benchmarks and the Apache-2.0 relicence, and will land on `main` as
v0.5.0.

## Now

### ~~Publish our own COCO numbers~~ — done 2026-09-18

Measured 47.2 / 51.2 / 51.9 AP for S / M / L on full COCO val2017, against Deci's
published 47.5 / 51.5 / 52.2. `examples/model_table.py` measures every column, so the
table regenerates rather than being quoted. See
[the model table](docs/benchmarks/model_table.md).

The constant 0.3 shortfall is the `iscrowd` item below; closing it should close the gap.

Out of scope: a leaderboard against other detectors. Their published latency uses an
NVIDIA T4 with TensorRT at batch 1 — hardware RunPod does not offer and we do not have —
and quoting their accuracy beside latency measured elsewhere would compare two different
protocols.

### The two blockers on any published mAP — 2026-09-13

Both are described in `docs/guides/training-review-2026-09-13.md`.

- **`iscrowd` ground truth is dropped rather than ignored.** COCO marks crowd regions
  *ignore*; a correct detection inside one currently scores as a false positive, costing
  roughly a point of AP — **measured at 0.3**, identically for all three variants, which
  is the entire gap between our numbers and Deci's. The clean fix widens the target tensor
  from `[N, 5]` to `[N, 6]` and ripples through the collate function and the loss.
- **Train, validation and deployment see three different geometries.** Training crops to
  5–80% of image area; validation and inference letterbox the whole frame. Train and eval
  therefore disagree on object-size prior. Rewriting the recipe toward the
  super-gradients shape is a design decision, not a bug fix.

### Port the training-audit fixes that remain — 2026-09-13

The fixes that applied to the Lightning path have landed. What the audit lists under
"Still open" has not, beyond the two blockers above: `RandomChannelSwap` shuffles all
three channels rather than reversing them, `_BBOX_PARAMS` has no `min_visibility`, and
the warmup is counted in batches rather than optimizer steps.

## Next

### Land `dev` on `main` as v0.5.0

Breaking, and the PR has to say so plainly: the licence changes from MIT to Apache-2.0,
`Mixup(p=)` becomes `Mixup(prob=)`, and `Detector(weights=...)` now loads checkpoints with
`weights_only=False` because Lightning `.ckpt` files require it.

### Documentation that describes the code that exists

`docs/api/training.md` and `docs/guides/training.md` still describe the manual `Trainer`,
which Lightning replaced. `docs/cli.md` documents four of the ten commands — `quantize`,
`qat`, `benchmark-dataset`, `serve`, `benchmark` and `demo` are missing, and there is no
quantization guide outside the notebooks.

### Train our own COCO weights

The pretrained weights this project downloads are Deci's, under the Super Gradients Model
EULA — research use only. Training from scratch on COCO (whose annotations are CC-BY 4.0)
would remove that restriction.

Budget honestly: YOLO-NAS-S is ~34 GFLOPs forward at 640, so ~100 GFLOPs per training
image. 118k images × 300 epochs is ~3.5 EFLOP, which at a realistic 20–40 achieved TFLOPS
is 25–50 A100-hours for the *smallest* variant — $40–80 on a spot 4090-class GPU, several
times that for L. Expect low-40s mAP rather than Deci's 47.5: that figure depends on
Objects365 pretraining, pseudo-labelling and distillation, and Objects365's own licence
makes it unavailable to us.

Nothing should be rented until the two blockers above are closed, or the resulting number
cannot be published.

## Later

- A `[train]` extra. `albumentations`, `torchmetrics` and `pycocotools` are core
  dependencies today, so inference-only users pay for the training stack. `albumentations`
  is also pinned to an exact version, which is hostile in a library.
- `ruff format` across the codebase — it rewrites 28 files, so it wants its own PR.
- `mypy` and `--doctest-modules` in CI.
- Coverage sits just above its 70% gate; the CLI command bodies are the thin part.

## Not planned

Adopting what larger projects do because they do it: a vendored OpenCV reimplementation,
VLM output parsers, versioned documentation, analytics widgets, or a two-branch release
flow. This project is small enough that they cost more than they return.
