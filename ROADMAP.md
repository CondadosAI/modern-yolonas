# Roadmap

What is planned and what is deliberately not. Items are ordered by what blocks the most
downstream work, not by size. Dates are when the item was raised.

Current state: `main` is released as **v0.5.0, Apache-2.0**, and carries Lightning
training, quantization and the dataset benchmarks. `dev` is ahead again; PR #38 is the
open `dev → main` release PR and should be merged as a merge commit, not a squash.

## Now

### Make a training hour worth renting — 2026-09-19

Measured 2026-09-19 on this machine (20 vCPU, RTX 3060 Laptop 6 GB, val2017 as the
per-sample cost is per image). An earlier note asserted training is CPU-bound in the
dataloader; that assertion was never measured, and the numbers below are what replaced it.

**The `yolonas train` pipeline has no Mosaic.** `train_cmd.py:134` builds a plain `Compose`;
`Mosaic` is reachable only through `training/run.py`, which the dataset-benchmark commands
use. Any claim about mosaic cost does not apply to the COCO run this roadmap plans.

Single process, per sample, `yolonas train` pipeline — 19.2 ms total, 52 img/s per core:

| stage | ms | share |
|---|---:|---:|
| `HSVAugment` | 3.93 | 20% |
| `Mixup` | 3.46 | 18% |
| `RandomAffine` | 3.32 | 17% |
| `Normalize` | 3.30 | 17% |
| decode (`load_raw`) | 3.10 | 16% |
| `RandomResizedCrop` | 1.50 | 8% |
| flip + channel swap | 0.64 | 3% |

JPEG decode is 16%, not the bottleneck — so a pre-resized disk cache, nvJPEG or DALI would
buy almost nothing and are dropped from this plan. The two largest entries are both pure
waste and were prototyped:

- `HSVAugment` copies the whole image twice to swap BGR/RGB around an Albumentations call
  that converts colour space again internally. A three-LUT `cv2` implementation:
  **5.32 → 2.32 ms**.
- `Normalize` emits float32 CHW in the worker — 4.92 MB per sample through collate,
  `pin_memory` and PCIe. Emitting uint8 and doing `.float().div_(255)` on the GPU:
  **3.23 → 0.28 ms and 4.92 → 1.23 MB**, with the reconstructed float32 bit-identical to
  today's output (max abs diff 0.0 — identical *input* to the model, which says nothing
  about the loss).

  Blast radius, measured: `Normalize` is constructed in `train_cmd.py`, `eval_cmd.py`,
  `quantize_cmd.py`, `training/run.py`, `benchmarks/rf100vl.py`, two `examples/` scripts and
  four test modules. Only the Lightning path gets an `on_after_batch_transfer` hook for
  free; `eval_cmd` and `quantize_cmd` consume batches directly and each need the cast made
  explicit. `inference/preprocess.py` is a separate path and stays independent.

Together that is 31% of loader time, which would move 52 → ~75 img/s per core.

`T_loader`, batch 16, `yolonas train` pipeline:

| workers | with `cv2.setNumThreads(0)` | without |
|---:|---:|---:|
| 4 | 259 img/s | 221 |
| 8 | **383 img/s** | 371 |
| 20 | 259 img/s | 209 |

Throughput *falls* past 8 workers: the workers oversubscribe the cores. Pinning threads is
worth 3% at 8 workers and 24% at 20 — it matters most exactly where the config is wrong,
so it is a guard, not a speedup. Scaling is already sub-linear by 8 workers (4 → 8 buys
1.48×, not 2×) and collapses at 20, so the default of 8 is close to right on 20 cores.
A pod's **vCPU count** is still the largest loader lever available, and it costs no code.

`T_model`, YOLO-NAS-S, 640, synthetic batch:

| config | img/s |
|---|---:|
| fp16 AMP, batch 8 | 40.8 |
| fp16 AMP + `channels_last` | **47.1** |
| bf16 AMP | 38.6 |
| no AMP | 26.1 (5.17 GiB) |

`channels_last` is +15% for a one-line change and should land. bf16 measured *slower* than
fp16 here, so it is a `GradScaler` stability argument, not a throughput one — do not sell
it as a speedup. Backward is 57% of the step and the loss, assigner included, is only 4%,
so the `TaskAlignedAssigner` is not worth optimising.

**On this hardware training is GPU-bound with roughly 10× headroom** (383 vs 41 img/s), so
none of the loader work pays off locally.

Whether it pays off on a rented 4090 is **not answered here, and should not be guessed**.
One data point exists — a 3060 Laptop reaching ~3.7 achieved TFLOPS, about 28% of its fp16
peak — and extrapolating it to another card at an assumed identical utilisation is the same
kind of unmeasured claim this item was written to retract. Run `T_loader` and `T_model` on
the first pod, before the first real epoch. It costs one pod-hour and it decides whether
any of the transform work below is worth doing at all.

`DetectionDataModule` sets neither `persistent_workers` nor `prefetch_factor`, so workers
respawn every epoch. Turning `persistent_workers` on **silently breaks**
`CloseMosaicCallback`, which mutates the transform object in the main process and today
reaches the workers only because they are respawned: mosaic would never close, and only the
final AP would show it. The two changes are one change — but note that since `yolonas train`
builds no Mosaic, the callback has nothing to close on that path today, so this bites the
`training/run.py` benchmark path and any future recipe that enables mosaic, not the COCO run.

Harness: `stages`, `loader` and `model` subcommands; it patches nothing, so the
`cv2.setNumThreads` question is measured rather than assumed.

### ~~Publish our own COCO numbers~~ — done 2026-09-18

Measured 47.3 / 51.3 / 52.0 AP for S / M / L on full COCO val2017 at the NMS IoU 0.70
optimum, against Deci's published 47.5 / 51.5 / 52.2. (The same measurement at IoU 0.65
gives 47.2 / 51.2 / 51.9; the table publishes 0.70.) `examples/model_table.py` measures
every column, so the table regenerates rather than being quoted. See
[the model table](docs/benchmarks/model_table.md).

A small residual to Deci's figures remains and is **not** explained by `iscrowd`, which an
experiment ruled out — see the model table for what was tested and what is still open.

Out of scope: a leaderboard against other detectors. Their published latency uses an
NVIDIA T4 with TensorRT at batch 1 — hardware RunPod does not offer and we do not have —
and quoting their accuracy beside latency measured elsewhere would compare two different
protocols.

### The blocker on a from-scratch training run — 2026-09-13

Described in `docs/guides/training-review-2026-09-13.md`.

**Train, validation and deployment see three different geometries.** Training crops to
5–80% of image area; validation and inference letterbox the whole frame. Train and eval
therefore disagree on object-size prior. Rewriting the recipe toward the super-gradients
shape is a design decision, not a bug fix, and it gates a COCO run being worth its GPU
rental.

### `iscrowd` in the training targets — 2026-09-13

`data/coco.py` drops crowd annotations instead of marking them ignore, so the model gets
no signal in those regions and is implicitly taught they are background.

This was previously listed as a blocker on published accuracy. **It is not**, and the
earlier claim that it explained the gap to Deci's figures was wrong. Measured: stripping
crowd annotations from the ground truth *lowers* AP by 0.43, which shows `COCOEvaluator`
already handles them correctly — it scores against the annotation file, and `pycocotools`
marks crowd regions ignore. The dataset's targets never reach that number.

It still matters for training quality, and for the `DetectionMetrics` (torchmetrics) path,
which does build ground truth from dataset targets. The clean fix widens the target tensor
from `[N, 5]` to `[N, 6]` and ripples through the collate function and the loss.

### Port the training-audit fixes that remain — 2026-09-13

The fixes that applied to the Lightning path have landed. What the audit lists under
"Still open" has not, beyond the two blockers above: `RandomChannelSwap` shuffles all
three channels rather than reversing them, `_BBOX_PARAMS` has no `min_visibility`, and
the warmup is counted in batches rather than optimizer steps.

## Next

### Instance segmentation

Planned as YOLACT-style prototypes: a proto net on the stride-8 neck output plus a
`mask_coeff` branch beside `cls_pred` / `reg_pred`, with `mask = sigmoid(coeffs @ protos)`
cropped to the box. Everything stays convolutional, so the existing ONNX export and QAT
paths carry over without a special case, at roughly +10% FLOPs.

EdgeCrafter's ECInsSeg was the starting reference but does not port: it reuses DETR
decoder queries for mask prediction, and a dense anchor-free DFL head has no queries.

Expect mask AP well below box AP — YOLOv8s-seg publishes 44.6 box against 36.8 mask.
Do not promise parity with the detection numbers above.

The bulk of the work is the data path, not the head: `data/coco.py` returns boxes only,
and all ~15 transforms take `(image, targets)` and are mask-blind. Whether that becomes a
`Sample(image, boxes, labels, masks)` dataclass or a threaded third argument is an open
design decision. It is sequenced *after* the throughput item above, which needs the
current contract intact.

`COCOEvaluator` will need `iouType="segm"` — and the letterbox inversion that bug #2 of
the audit already cost us once will recur for masks, where a wrong result still looks
plausible on screen. Extend the ground-truth-as-prediction test to require mask AP 1.0
before trusting any mask number.

### A clean backbone initialisation via DINOv3 distillation

Deci's weights are research-only and fine-tuning does not launder them. Segmentation has
no pretrained checkpoint at all, so the head must be trained either way, which leaves the
backbone initialisation as the only tainted link.

DINOv3's licence permits commercial use provided "Built with DINOv3" is displayed, so
distilling a DINOv3 backbone into ours would give the project its first clean init — for
detection as much as for segmentation. `stage3` is already stride 16 with 384 channels,
which lines up with DINOv3's patch size directly.

Distil from the released DINOv3 weights, not from EdgeCrafter's checkpoints or code: that
project is under a commercial-on-request licence. Use COCO train plus COCO *unlabeled*
rather than the paper's ImageNet-1K, for the same licence reason that rules out
Objects365.

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
- Replace `print` and `console.print` with the `logging` module. A library should not write
  to stdout on its own initiative, and the CLI's own `--verbose`/`--quiet` flags already
  configure logging levels that most of the code bypasses.
