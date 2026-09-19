# Training code review — 2026-09-13

> **Status on `dev` (2026-09-18).** This review was written against `main`'s manual
> `Trainer`, which no longer exists — training runs on Lightning now. The fixes below were
> re-applied to the Lightning path, and two of them turned out to be live defects there
> too: the Mosaic box-size bug and the BatchNorm pollution during validation. Regression
> tests are in `tests/test_training_audit_fixes.py`. What changed in the re-application:
>
> - **Fix 1 (validation NMS threshold)** was already correct on `dev`:
>   `YoloNASLightningModule` defaults to `conf_threshold=0.001`. The kill chain it
>   describes — `best.pt` gated on `mAP_50 > 0.0`, early stopping on a stuck mAP — died
>   with the manual callbacks; Lightning's `ModelCheckpoint` monitors `val/mAP` when
>   annotations are present and `val/loss` otherwise.
> - **Fix 2 (validation letterbox)** was still live in `cli/train_cmd.py` and is now fixed.
> - **Gradient clipping**, listed under "Smaller" below, is now on by default:
>   `--grad-clip 10.0` on `yolonas train`, `grad_clip` in the recipes.
>
> The "Still open" section at the end is still open, and still gates any published mAP
> number.



Reviewed at `b027a79` on `main`, before deciding whether to rent a GPU for a from-scratch
COCO run. Scope: `training/`, `data/`, `head/dfl.py`, `inference/postprocess.py`,
`cli/train_cmd.py`.

Everything below is a defect that **does not raise**: training runs, loss decreases, and the
result is quietly wrong — or the run dies for a reason that looks like a bad model.

**Status:** the blocker cluster and the Mosaic defect are fixed. `tests/test_target_roundtrip.py`
adds 12 tests: **7 of them fail on the pre-fix code** (verified by reverting the fixes and
re-running); the other 5 pass either way and exist to pin the encode/decode conventions that the
round-trip proved correct, so a later refactor cannot quietly break them. Full suite: 176 passed,
1 skipped. The "still open" items at the end are not done.

---

## Fixed

### 1. Validation NMS ran at the inference confidence threshold

`trainer.py` called `postprocess(pred_bboxes, pred_scores)` with no arguments, taking
`conf_threshold=0.25` from `inference/postprocess.py`.

COCO mAP integrates the full precision/recall curve, which requires the low-confidence tail
(reference implementations use ~0.001 with `max_det=100`). At 0.25 the curve is truncated:
AP is understated and mAR badly so.

**The failure that actually matters is from-scratch.** For the first tens of epochs no
prediction exceeds 0.25, so `postprocess` returns empty for every image and mAP reads
*exactly* 0.0 — indistinguishable from a broken run. That arms a kill chain:

1. `best.pt` is gated on `mAP_50 > self.best_map`, and `best_map` starts at 0.0 → never saved.
2. `EarlyStoppingCallback` watches `val_metrics/mAP` with `_best_map = -1.0`. The first
   validation sets it to 0.0; every later 0.0 fails `> 0.0 + min_delta` → `_wait` increments.
3. `configs/train.yaml` ships `early-stopping-patience: 5` with `val-freq: 10`, so a healthy
   run stops at **epoch 60 with no checkpoint**.

**Fix:** `_validate` now passes `conf_threshold=0.001, max_detections=300`. The NMS IoU is left
at the existing 0.7 — it was not part of this defect, and the value super-gradients evaluates
YOLO-NAS at could not be verified here (super-gradients is not installed and no recipe is on
disk). Worth checking against the upstream recipe before publishing an mAP number. torchmetrics
caps detections at 100 via its own `max_detection_thresholds`.

**Test:** `TestValidationPostprocessingKeepsTheTail` — asserts a randomly-initialised model has
no box above 0.25, that the default threshold therefore yields zero detections, and that
`_validate` submits a non-empty detection set to the metric. Asserting on the mAP *value* would
not catch this: torchmetrics reports 0.0 for zero predictions, which looks like an untrained
model.

### 2. Validation cropped instead of letterboxing

`cli/train_cmd.py` had `LetterboxResize` commented out in both pipelines and used
`CenterCrop(size=input_size)` for validation.

Measured on the pinned albumentations 1.4.24:

```
CenterCrop(640) on a 375x500 image -> CropSizeError:
    Crop size (height, width) exceeds image dimensions: (640, 640) vs (375, 500)
CenterCrop(640) on a 640x960 image -> 3 GT in, 1 GT out
    (both edge objects deleted, only the centre one survives)
```

COCO val2017 images are mostly 640 or less on the long side, so at `input_size=640` **the first
validation pass raises and kills the run** — after `val-freq: 10` epochs of GPU time have
already been paid for. On a rented pod that is the whole experiment.

On larger images it does not raise, it just deletes edge objects from image and ground truth
alike, and the number stops being comparable to `yolonas eval`, which letterboxes.

**Fix:** `val_transforms = Compose([LetterboxResize(target_size=input_size), Normalize()])`,
matching `eval_cmd.py` and the inference path.

### 3. Validation ran the model in training mode

`_validate` called `eval_model.train()`, documented as safe because it sits inside
`torch.no_grad()`. That is true for gradients and false for buffers.

- BatchNorm in train mode normalises by **batch statistics**, so the reported mAP depends on the
  validation batch size and is not what the exported model produces.
- BatchNorm in train mode **updates `running_mean` / `running_var`**. Buffer writes are not
  autograd operations, so `no_grad` does not stop them. Every validation overwrote the EMA
  model's BN statistics with validation-set statistics — and `_save_checkpoint` then wrote those
  contaminated buffers into `best.pt` and `last.pt`.

The only reason for `train()` was that `NDFLHeads.forward` returns raw predictions (needed for
the validation loss) only when `self.training`.

**Also a DDP deadlock waiting to happen.** `convert_sync_batchnorm` runs *before* `ModelEMA`
deep-copies the model, so the EMA model holds `SyncBatchNorm`, and `_validate` runs on rank 0
only. SyncBN in train mode issues collectives while ranks 1..N are in the next epoch's forward
issuing their own. Invisible at `num-gpus: 1`; it appears on the first multi-GPU pod.

**Fix:** added `NDFLHeads.return_raw_outputs`. `_validate` now sets it, stays in `eval()`
throughout, and clears it afterwards. `torch.jit.is_tracing()` still short-circuits first, so
export is unaffected.

**Test:** `TestValidationDoesNotMutateModel` — snapshots every `running_mean`/`running_var` in
the model (and separately in the EMA model), runs `_validate`, and asserts bit-equality.

### 4. Mosaic emitted every box at half size

`data/transforms.py`. After the four-image placement loop, targets are normalized to the
`2s × 2s` canvas. The image is then cropped to `s × s`, and the centres were renormalized by 2 —
but the widths and heights were not:

```python
targets[:, 1] = targets[:, 1] * 2 - crop_x / s   # centre x — correct
targets[:, 2] = targets[:, 2] * 2 - crop_y / s   # centre y — correct
#     columns 3 and 4 (w, h) were never rescaled
```

Measured with four 640×640 source images each holding one 320 px box:

```
pixel w, h on the 640 canvas: 160.0  160.0     <- should be 320.0
```

Exactly half. Concentric boxes at half width and half height have IoU 0.25 with the truth, so
the model would be trained to under-predict extent on every mosaic'd sample.

Mosaic is **not currently wired into `train_cmd.py`**, so this was latent. It matters anyway:
mosaic is the core of the super-gradients YOLO-NAS recipe (Mosaic → RandomAffine → Mixup → HSV →
HFlip → PaddedRescale) and there is no competitive COCO number without it.

**Fix:** scale `targets[:, 3:5]` by 2 alongside the centres, and clip boxes to the crop window
(the old code kept a box whose *centre* was inside but never clipped its extent, so coordinates
could leave `[0, 1]` and reach the loss as GT outside the image).

**Test:** `TestMosaicPreservesBoxSize`. The one that catches the halving is
`test_uncropped_box_keeps_exact_size`: across 50 draws, at least one box must land fully inside
the crop and keep its exact pixel size. Its sibling `test_box_pixel_size_is_unchanged` only
guards against boxes *growing*, so the halving slips past it by design — noted in the test so
nobody later trusts the wrong one. `test_boxes_stay_inside_the_frame` covers the missing clip.

### 5. The ground truth had never been round-tripped through the decoder

There was no test taking GT boxes, encoding them as the loss encodes them, decoding them as the
head decodes them, and checking the result. `tests/test_loss.py`'s `test_forward_backward` only
asserted the loss runs. Nothing covered the assigners, Mosaic, or Mixup.

The chain that needed proving:

| step | code |
|---|---|
| GT xywh normalized → pixel xyxy | `loss.py` `forward` |
| pixel xyxy → grid-unit ltrb | `loss.py` `_bbox2dist`, after `/ pos_stride` |
| ltrb → DFL bin targets | `loss.py` `DFLLoss.forward` |
| logits → distances | `dfl.py` `softmax · proj_conv` |
| distances → pixel xyxy | `dfl.py` `_batch_distance2bbox · stride_tensor` |

**Result: it round-trips.** `TestDFLRoundTrip` builds the logits that exactly represent the DFL
targets, decodes them the way the head does, and gets IoU > 0.99 against the original boxes.
Channel layout is coord-major consistently on both sides, `(l, t, r, b)` ordering agrees, the
stride normalisation agrees, and the clamp to `reg_max - 0.01` matches upstream.

This is the check that costs nothing and rules out the most expensive class of bug. It should
have existed before any of the above was worth investigating.

---

## Still open

Not fixed. Ordered by what would bite a COCO run first.

### Train, validation and deployment see three different geometries

| stage | transform | scale distribution |
|---|---|---|
| train | `RandomResizedCrop(scale=(0.05, 0.8), ratio=(0.75, 1.33))` then `RandomAffine(scale=(0.5, 1.5))` | 5–80% of image area blown up to 640, aspect distorted; the full frame is **never** seen |
| val | `LetterboxResize` (now) | full frame, aspect preserved |
| deploy | `LetterboxResize` | full frame, aspect preserved |

Train and eval now disagree on object-size prior. This pipeline is a small custom-dataset recipe
— `configs/train.yaml` is `num_classes: 3`, `input-size: 320` — not a COCO recipe. Rewriting it
toward the super-gradients shape is a prerequisite for a meaningful COCO number, and is a design
decision, not a bug fix.

Two smaller issues in the same pipeline:

- **`RandomChannelSwap` is not a channel swap.** It wraps `A.ChannelShuffle(p=0.5)`, a uniform
  random permutation of all three channels (6 outcomes). The docstring claims "swap BGR channel
  order to RGB (and vice-versa)"; super-gradients' `DetectionRGB2BGR` reverses the order and
  nothing else. On COCO, colour is class-discriminative (stop sign, banana, fire hydrant,
  traffic light). Either rename it or make it `image[:, :, ::-1]` with probability `p`.
- **No `min_visibility` / `min_area` on `_BBOX_PARAMS`.** `min_width=2` / `min_height=2` let a
  box that has been 99% cropped away survive as a 2-pixel sliver still labelled "person".

### `iscrowd` ground truth is dropped instead of ignored

`data/coco.py` skips crowd annotations entirely. Correct for training, wrong for evaluation:
COCO marks crowd regions *ignore*, not background. A detection that correctly fires inside a
crowd region has no GT to match and scores as a false positive, deflating AP by roughly a point.
`torchmetrics.detection.MeanAveragePrecision` accepts an `iscrowd` key in the target dicts for
exactly this.

Left undone because the clean version changes the target tensor from `[N, 5]` to `[N, 6]` and
ripples through `collate.py` and the loss. Worth doing before publishing any mAP number.

### Subset and label-space traps

Both bite specifically on a reduced-size smoke run:

- `cat_id_to_label` is derived per-dataset from categories that *appear in annotations*, and
  `train_dataset` / `val_dataset` build it independently. On full COCO both yield the same 80
  entries. On a 10k-image subset where a class has no instances in one split, every label after
  it shifts by one — train and val end up in different label spaces, silently. `num_classes` is
  then auto-detected from the train mapping. **Force the canonical 80-class mapping when
  subsetting.**
- Subsets must **stride**, not slice. `self.ids` is sorted and COCO image IDs cluster by
  collection order, so `ids[:10000]` is a biased sample; use `ids[::12]`.

### Smaller

- **No gradient clipping anywhere.** With AMP and from-scratch initialisation this is the usual
  source of a run that NaNs in the first epoch. `clip_grad_norm_` after `scaler.unscale_`.
- **`best.pt` is selected on `mAP_50` while early stopping watches `val_metrics/mAP`.** Two
  criteria; pick one (COCO convention is `mAP`).
- **`scheduler.step()` runs every batch, including non-stepping ones.** `total_steps` is in
  batches too, so the cosine shape is self-consistent, but `warmup_steps=1000` means 1000
  *batches* — at the shipped `gradient_accum: 16` that is only 62 optimizer updates of warmup.
  This also produces the `lr_scheduler.step() before optimizer.step()` warning in the test suite.
- **A batch with zero GT contributes zero loss.** When *every* image in the batch has no
  annotations the loss returns `cls_logits.sum() * 0.0`, skipping the negative classification
  term. Per-image empties are fine (the assigner produces all-zero targets and VFL supervises
  them). Only reachable with `ignore-empty: false` and rare at batch ≥ 8.
- **`cudnn.benchmark` is commented out.** Fixed input size; free throughput on a rented GPU.
- **`TaskAlignedAssigner` does not deduplicate `mask_pos`** before the soft-score scatter, so an
  anchor matched to two GTs of different classes gets a positive soft score on both.
  super-gradients resolves to one. Negligible in practice.
- **`cv2.imread` returns `None` for a missing or corrupt file** and fails later with an unrelated
  error. One `if image is None: raise` saves an hour on a rented pod.

---

## What passes

Stated explicitly, because these are the checks that usually fail:

- **Every head is supervised.** All three `NDFLHeads` outputs are concatenated and both the
  classification and regression branches feed the loss. No dead head, no untrained channels
  sitting inside the latency measurements.
- **Nothing is pixel-meaned.** All three loss terms are normalized by `assigned_scores_sum`,
  matching super-gradients. (This is an anchor-free set-prediction loss, not a sparse heatmap,
  so the usual sparse-target failure does not apply — and the normalisation is right anyway.)
- **No learned loss weighting**, so nothing can chase a badly-scaled term. The weights are the
  fixed super-gradients values (1.0 / 2.5 / 0.5).
- **Latency is already measured on target devices** — `docs/benchmarks/latency_matrix.md`.
- **`eval_size` is never set by the model builders**, so anchors are always regenerated from the
  actual feature maps: no stale-anchor bug under multi-scale or a changed input size.
- **`ModelEMA.update` is safe** despite lacking `@torch.no_grad()`: `state_dict()` returns
  detached tensors, so the in-place ops build no graph and leak no memory.
- **Optimizer param groups are correct** — BatchNorm and bias excluded from weight decay,
  including frozen-parameter handling.

---

## Prediction for the smoke run

Written down before spending anything, so the result is a diagnosis and not just a number.

> With the fixes above and the augmentation pipeline left as-is, a from-scratch `yolo_nas_s` on a
> 10k-image **strided** COCO subset at 640, batch 16, AdamW 2e-4, ATSS warmup 4 epochs should
> show **`mAP_50 > 0.01` by epoch 10 and a monotone rise through epoch 30**. The absolute value
> will be poor — subset, no mosaic, mismatched train/val geometry — and that is expected. What is
> being tested is that the gradient signal reaches the boxes at all.
>
> If `mAP_50` is still exactly 0 at epoch 20, the problem is not the budget. Run
> `tests/test_target_roundtrip.py` first.
>
> `iou_loss` should fall below ~1.0 (GIoU loss of 1.0 means IoU ≈ 0) within the first epoch. If
> it plateaus near 2.0, boxes are being decoded at the wrong scale.

## Budget, for reference

YOLO-NAS-S is ~33 GFLOPs forward at 640 → ~100 GFLOPs/image for training. 118k images × 300
epochs ≈ 3.5 EFLOP. At a realistic 20–40 TFLOPS achieved that is 25–50 A100-hours for the
smallest variant: roughly **$40–80 on a spot 4090-class GPU**, 2–4× that for L.

A COCO-only run will not reach Deci's 47.5 mAP for S — that number depends on Objects365
pretraining, pseudo-labelling and distillation. Expect low 40s, by analogy with PP-YOLOE-S (43.0)
and YOLOv8s (44.9).

COCO annotations are CC-BY 4.0, so from-scratch on COCO is licence-clean. Objects365 has its own
restrictions; do not use it as pretraining.
