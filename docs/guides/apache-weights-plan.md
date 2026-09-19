# Plan: our own COCO weights, under Apache-2.0

Every pretrained checkpoint this project downloads today is Deci's, under the Super
Gradients Model EULA — research use only. That single fact is why `weights.py` carries a
warning, why the Hugging Face Space carries a notice, and why nobody can ship anything built
with this repo. Training our own weights removes it.

Accuracy is the secondary goal. A number a point or two below Deci's, that anyone may use
commercially, is worth more to this project than a number that matches and nobody may touch.

## Ground rules

Decided 2026-09-19, and they constrain every stage below.

**Training data: COCO only.** `train2017` (118k labelled) and `unlabeled2017` (123k, no
labels, same source and terms). No Objects365 — its licence excludes us, and it is the single
largest contributor to Deci's 47.5. No ImageNet as *our* training data.

**External weights: Apache-2.0 or equivalent is acceptable**, judged on the released licence
rather than on what the authors pretrained with. This is the industry norm, and it is a
policy choice rather than a legal finding. It admits D-FINE and RT-DETR as teachers even
though their backbones are almost certainly ImageNet-initialised.

**Excluded outright:** DEIMv2 and EdgeCrafter. Both are Intellindust's, both are
commercial-by-request, and both hold the best numbers in their class — they are reading
material, not components.

**Attribution that must ship with the weights:** "Built with DINOv3" wherever the weights are
documented, per Meta's licence. If SAM 3 contributes masks, its licence wants acknowledgement
in any publication.

## What is measured, and what is not

On the training machine (24 vCPU, RTX 5060 8 GB, 7.53 GiB usable, torch 2.10), 640 px, fp16,
`channels_last`:

| config | img/s | peak VRAM |
|---|---:|---:|
| batch 16, no teacher | 83.8 | 4.84 GiB |
| batch 24 + `torch.compile`, no teacher | **96.9** | 6.00 GiB |
| batch 24, no compile | OOM | — |
| batch 16 + DINOv3-S frozen | 51.4 | 5.08 GiB |
| batch 24 + DINOv3-S | OOM | — |
| batch 16 + DINOv3-B frozen | 33.3 | 5.58 GiB |
| batch 8 + D-FINE-X frozen | 33.3 | 3.59 GiB |

End to end, `yolonas train` over 4952 images: 87.4 img/s at batch 24 with `--compile`. The
dataloader peaks at 1600 img/s without mosaic and 684 with it, so it is not the constraint at
any of these rates.

Two things fall out of this, and they set the shape of the plan:

**A frozen teacher costs almost no memory and a great deal of time.** It runs under
`no_grad`, so it retains no activations — DINOv3-S adds 0.24 GiB — but its forward is 36% of
the step, and DINOv3-B's is 58%. Distillation fits on this card. It just runs at 0.6x.

**So online distillation from a detector is out, and offline pseudo-labelling is in.**
Keeping D-FINE-X in the loop gives 33 img/s, which is 12 days for 300 epochs on 118k images.
Running it *once* over all 241k images as an inference pass is on the order of an hour. Deci
pseudo-labelled rather than distilling online for the same reason.

Not measured, and not to be guessed at: throughput on any other GPU. One pod-hour answers it
and reprices everything below.

## Stages

### 0 — Fix the recipe before spending a week on it

Three defects, two of which the roadmap has carried since 2026-09-13.

`yolonas train` is **not** the COCO recipe. It builds its own pipeline — AdamW, 300 epochs,
`RandomResizedCrop` at 5–80% of area, **no mosaic at all** — while `COCO_RECIPE` in
`recipes.py` is SGD, 100 epochs, letterbox, mosaic with `close_mosaic` over the last 15, and
is reachable only through `yolonas benchmark-dataset coco`. The second is the one that
resembles every published recipe. Either `train` adopts it or the docs say plainly which
command trains COCO.

**Train and validation disagree on geometry.** Training crops; validation letterboxes. The
model learns an object-size prior the metric does not measure. This disappears on the
`benchmark-dataset` path, which letterboxes both.

**`iscrowd` is dropped rather than marked ignore**, so crowd regions teach "background". This
one is in `data/coco.py` and affects both paths. It needs the target tensor to widen from
`[N, 5]` to `[N, 6]` and ripples through collate and the loss.

Also open: `Mixup.inner_transforms` is set only in `run.py`, so on the `train` path half of
all samples blend an aggressively cropped image with an untouched letterboxed one.

**Gate:** a 15-epoch run must produce a sane mAP trajectory before anything longer starts.

### 1 — A clean backbone, distilled from DINOv3

Replaces what Objects365 did for Deci, using only COCO images. The objective needs no labels,
so it runs over all 241k.

DINOv3's patch grid at stride 16 lines up with the backbone's `c4` exactly, so the
distillation head is a 1x1 projection and a cosine loss — already implemented in
`tools/bench_training.py` for the measurement above. **The architecture does not change**, so
`state_dict` compatibility with super-gradients survives; only the initialisation differs.

At 51.4 img/s — an upper bound, since this stage runs no detection head — 20 epochs over 241k
is about **26 hours**. EdgeCrafter used 50, which would be 2.7 days.

**Gate:** the distilled backbone must beat a random init on a short detection fine-tune. If
it does not, stages 2 and 3 are wasted on it.

### 2 — Pseudo-label `unlabeled2017`

One inference pass with D-FINE-X (55.8 AP on COCO, Apache-2.0, COCO-only checkpoint — not the
`_obj365` variants, which carry Objects365's terms). Offline, an hour or so, done once.

D-FINE was trained on COCO, so its boxes follow COCO's conventions by construction. That is
the reason to prefer it over SAM 3 here: SAM 3 is a stronger segmenter but its notion of an
instance is its own, and a pseudo-labeller misaligned with the target annotation style
injects a bias the metric will punish.

**Gate:** pseudo-label `val2017` with the same settings and score it. If the pseudo-labels do
not reproduce roughly D-FINE-X's published AP against the real annotations, the confidence
threshold or the postprocessing is wrong, and 123k bad labels are worse than none.

### 3 — Train detection

Phase A on all 241k with real and pseudo labels mixed, phase B fine-tuning on the 118k real
ones. At 87 img/s: 100 epochs of A is ~3.2 days, 200 of B is ~3.1. With stage 1, call it
**eight days end to end on this machine**, assuming no restarts.

Free and worth taking regardless of any teacher: **GO-LSD**, D-FINE's self-distillation, where
deeper layers supervise shallower ones; and the EMA teacher already in `EMACallback`. Neither
needs external weights or extra data.

**Expected result: low-to-mid 40s AP.** A naive COCO-only run lands near 43 by analogy with
PP-YOLOE-S (43.0) and YOLOv8s (44.9). Deci reached 47.5 with Objects365, pseudo-labels and
distillation; we keep two of those three. Anything more precise would be a guess — there is
no measured point on this pipeline.

### 4 — Publish

Weights to `CondadosAI` on the Hub under Apache-2.0, with "Built with DINOv3" and the recipe
that produced them. Repoint `weights.py` and `hub.py` away from Deci's S3, and keep the EULA
path available for anyone who wants to reproduce the old numbers.

## Then instance segmentation

The detection weights are the prerequisite: the mask head trains on top of a backbone that
must already be clean, or the masks inherit the licence problem the boxes just escaped.

`train2017` already carries masks, so the labelled half needs nothing new. For the
pseudo-labelled half, **SAM 3 prompted with the boxes from stage 2** produces masks without
ever having to guess what the object is — its visual-prompt mode takes a box and segments
within it, which sidesteps the vocabulary mismatch that rules it out as a box source.

Design and sequencing for the mask head itself are in `ROADMAP.md` under "Instance
segmentation": YOLACT-style prototypes on the stride-8 neck output, because the query-based
head that EdgeCrafter and DEIMv2 use has no counterpart in a dense DFL head.

## Where this could go wrong

The gates above exist because each stage can fail quietly. In order of how much they would
cost: a bad pseudo-label pass poisons 123k images and shows up only as a lower final AP; a
backbone distillation that transfers nothing costs a day and looks exactly like one that
worked until the fine-tune; and the geometry mismatch in stage 0 produces a model that trains
and evaluates fine while being systematically worse than it should be.

None of them raise an error. All of them are cheap to check before the expensive step.
