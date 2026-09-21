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

**Gate: the calibration run.** Fifteen epochs on full `train2017` must produce a rising mAP
trajectory before anything longer starts.

**What it deliberately does not include.** The calibration is stage 3 *alone*: random
initialisation (`--no-pretrained`), real annotations only, no distilled backbone and no
pseudo-labels. Running it with the whole pipeline stacked would mean a failure could not say
which stage broke. Its mAP is therefore a **floor**, not a forecast — the finished recipe
adds a DINOv3-distilled backbone in place of a random one, 123k pseudo-labelled images, and
twenty times the epochs.

What it can tell you, and nothing else can this cheaply, is whether the detection pipeline
converges at all. The known failure mode is specific: a letterbox inversion bug once left
this repo's mAP pinned near 0.008 while the loss fell perfectly well. The signal to watch is
recall — if AP is low but AR is climbing, the model is finding objects and learning to name
them, which is the correct shape. If AR is also flat, the geometry is broken.

Measured 2026-09-19, yolo_nas_s from scratch, batch 16, `--recipe coco` at lr 2e-2:

| epoch | AP | AR |
|---|---:|---:|
| 0 | 0.020 | 0.186 |
| 1 | 0.050 | 0.278 |

**What this gate has already paid for.** In its first four minutes it found two COCO
annotations with a zero-height box, which Albumentations rejects outright and which had been
latent since the COCO recipe was written. Setting it up surfaced two more.
`close_mosaic_epochs` equal to the epoch count leaves exactly *one* epoch of mosaic: the
condition `epoch >= max_epochs - close` is true from epoch zero, but that epoch's dataloader
workers are already iterating when the callback fires, so only epoch one onward is affected.
The log reports the configuration, never the effect. And the recipe's SGD learning rate had
never been validated and converged at half the rate of anything else. None of the three
raises an error on its own.


### 1 — A clean backbone, distilled from DINOv3

Replaces what Objects365 did for Deci, using only COCO images. The objective needs no labels,
so it runs over all 241k.

DINOv3's patch grid at stride 16 lines up with the backbone's `c4` exactly, so the
distillation head is a 1x1 projection and a cosine loss — already implemented in
`tools/bench_training.py` for the measurement above. **The architecture does not change**, so
`state_dict` compatibility with super-gradients survives; only the initialisation differs.

**The cache is disk-bound, not GPU-bound.** This was not anticipated: the plan costed the
cache pass from the teacher's throughput alone, as if writing were free. Measured on the
training machine 2026-09-19, the pass runs at 48 img/s until the first shard is flushed and
then settles at **19 img/s** — with the GPU at 100%, 67 °C, no throttling and no I/O wait in
instantaneous samples. Each shard writes 2.32 GB, and `dd ... oflag=direct` under contention
returns 14.6 MB/s.

The drive is an ADATA LEGEND 710, DRAM-less and 78% full. Writing 298 GB continuously
exhausts its SLC cache and drops it to native speed. So the real figure is **~3 hours for
the cache**, not the 40 minutes the teacher's 434 img/s on a 4090 would suggest.

The decision to cache still holds, and by a wide margin:

| | total |
|---|---:|
| cache (3 h) + distillation at 97 img/s | **16.8 h** |
| frozen teacher in the loop at 51.4 img/s | 26 h |

Nine hours, even paying three to a slow disk. Caching only `train2017` would halve the write
to 145 GB and save 1.5 of those hours, at the cost of half the images — a bad trade, since
the 123k unlabelled ones are exactly what stands in for Objects365 here.

**Generalisable:** a feature cache trades GPU time for disk, and the disk side of that trade
needs its own measurement. On consumer NVMe the sustained write rate after the SLC cache
fills is the number that matters, not the burst rate in a benchmark.

**Reclaim the cache once the gate has passed.** The feature cache is 279 GB and is dead
weight the moment the distilled backbone is saved — nothing downstream reads it. Leaving it
costs the next stage directly: the training machine's drive is DRAM-less, and running it near
full is what dropped sustained writes to 14.6 MB/s during the cache pass. Stage 3 writes
checkpoints across hundreds of epochs onto that same disk.

Delete it *after* the gate, not before: a failed gate is exactly when the cheap experiments
— a deeper projection head, MSE instead of cosine — want the cache still there, and
regenerating it is three hours.

```
rm -rf ~/featcache          # 279 GB, only after the distilled backbone is saved and scored
```

**Gate: beat this number.** The distilled backbone has to outperform a random init on a
short detection fine-tune. The baseline is measured rather than left to judgement --
2026-09-19, `yolo_nas_s` from scratch, `--recipe coco` at lr 2e-2, batch 16, on full
`train2017`:

| epoch | AP | AR |
|---:|---:|---:|
| 0 | 0.020 | 0.186 |
| 1 | 0.050 | 0.278 |
| 3 | 0.085 | 0.343 |
| 5 | 0.118 | 0.386 |
| **7** | **0.146** | **0.417** |

So: eight epochs of the same recipe, starting from the distilled backbone instead of a
random one, must clear **AP 0.146**. The comparison is only fair against the same recipe,
the same epoch count and the same learning rate, so run it exactly that way.

A distillation that transferred nothing looks identical to one that worked until this
fine-tune, which is why the gate exists and why the baseline needs a number rather than an
impression. Checkpoint of the baseline run: `runs/calib/epoch=7-step=58632.ckpt`.

If it does not clear the bar, stages 2 and 3 are wasted on it.

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

## Self-supervised pretraining, and when it is worth it

Stages 1 and 3 both assume the target is COCO. Most users of this repository are not
training on COCO — they have a pile of unlabelled frames from their own domain and a
checkpoint trained on somebody else's. `yolonas pretrain` and `yolonas domain-distance`
exist for that case, and the second one exists because the first is expensive.

### `lightly` is not LightlyTrain

The alternatives section above rejects **LightlyTrain** on licence, and that rejection
stands. It does not extend to **`lightly`**, the library, which is a different package
under a different licence: MIT, and the vendor's own wording is *"always be free to use,
even for commercial purposes."* MIT places no condition on weights trained with it.

Stating this explicitly because I conflated the two while researching this section, and
the names make that easy to do. `lightly-train` is AGPL and stays out; `lightly` is MIT
and is an optional dependency (`modern-yolonas[ssl]`).

### Two scales

**Pretrain the YOLO-NAS backbone directly.** Implemented, as `yolonas pretrain`. It needs
only images, writes the same `model.backbone.*` checkpoint the distillation stage writes,
and feeds the same `--init-backbone` flag. On a user's own unlabelled data this is the
cheap option: one backbone, no teacher, no annotation.

**Adapt DINOv3 first, then distil.** Not implemented. At a larger scale the teacher itself
can be tuned on the target domain before stage 1 distils it, which moves the whole
representation rather than just the student. It costs a ViT pretraining run and only makes
sense with a lot of unlabelled data and a domain far from LVD-1689M. Recorded as the
extension, not scheduled.

### Why DenseCL and not SimCLR

A per-image contrastive loss optimises one vector per photo. That rewards summarising an
image, which is the wrong granularity for a detector: it has no pressure to keep positions
distinguishable. DenseCL (arXiv:2011.09157) adds a second contrastive term between *pixels*
of the two views, matched by feature similarity rather than position, since the two views
are different crops and have no positional correspondence.

The published numbers that justify the choice:

| comparison | gain | note |
|---|---|---|
| DenseCL vs MoCo-v2, both pretrained on COCO | +1.1 AP | the like-for-like one |
| DenseCL vs supervised ImageNet init, on VOC | +4.5 AP | the headline, and a weaker baseline |

**Every one of those numbers is on a ResNet.** Our backbone is QARepVGG. The ranking of
methods should carry across architectures; the margin is unmeasured here, and quoting +1.1
AP as something this repository will reproduce would be inventing a result.

Two related findings worth recording, both arguing the same way — that detection
pretraining wants detection-like data and detection-like objectives, not ImageNet:

- **AlignDet** pretrains the detection head as well as the backbone, in 12 epochs.
- **BEiT pretrained on COCO beats BEiT pretrained on ImageNet** for detection, despite COCO
  being the smaller and less curated set.

### What this costs that the cached-feature path does not

Stage 1 is restricted to photometric augmentation, because the teacher features are cached
and a ViT is not flip-equivariant. SSL has no cache and no teacher, so it is free to use the
full geometric augmentation contrastive learning depends on — random resized crops from 0.2
of the image, in DenseCL's configuration. That is a genuine advantage of this path over
stage 1, and the first time in this plan that dropping the cache buys something.

The cost is two forward passes per image instead of one, plus a momentum encoder.

**One known departure from the paper.** MoCo shuffles BatchNorm statistics across GPUs so the
query and key encoders cannot communicate through their BN buffers. On a single GPU there is
nothing to shuffle and this backbone is full of BN, so some leakage is expected and some of
the paper's margin is probably lost with it. The published fix requires multiple GPUs.

### Matching the input scaling

The detection path scales `uint8` by 1/255 and applies no mean/std. `lightly`'s transforms
default to ImageNet normalisation. Left alone, a backbone pretrained on normalised inputs
would meet a shifted distribution at its first convolution the moment it was fine-tuned —
no error, no warning, just a worse number that would have been attributed to the method.
The transform is built with `normalize=None` and a test pins the range.

## Deciding whether to spend the GPU-days

The honest way to find out whether pretraining helps on a given dataset is to fine-tune with
and without it and compare, which costs exactly what the decision was supposed to save.
`yolonas domain-distance` is the cheap proxy: it reports how far a dataset sits from COCO in
the feature space of the backbone that would actually be transferred.

Two statistics, chosen because they fail differently:

**Proxy A-distance**, `d_A = 2(1 − 2ε)`, where ε is the cross-validated error of a linear
domain discriminator. The headline, because it comes with a sentence: 0 means a probe cannot
separate the two sets at all, 2 means it separates them perfectly. The cross-validation is
not optional — an in-sample error on a few thousand high-dimensional vectors is ~0 for any
pair, so the uncorrected version reports maximal distance for every dataset including one
compared against itself, and that failure looks exactly like a working tool.

**KID**, the unbiased MMD² under a polynomial kernel. FID's estimator is biased by sample
size, so an FID over 2000 images is not comparable to one over 5000 — a mistake a `--samples`
flag invites. FD-DINOv2 was considered and rejected: it fixes some of FID's domain bias at, in
its authors' words, tremendous computational cost, which is the wrong trade for a command
whose purpose is to be cheaper than the experiment it replaces.

### No threshold is invented

There is no experiment anywhere in this project mapping a domain distance to an expected AP
gain. Rather than print a traffic light backed by nothing, the command takes `--baseline`: a
second sample of the reference domain, which measures **the floor** — what the statistic reads
when two sets genuinely are the same domain. The user's number is reported beside it.

The floor is worth publishing on its own. Every domain-shift paper needs to know what "no
shift" reads as, and none of them report it.

Measured floor, `yolo_nas_s` backbone features, 1000 images per side at 448px:

<!-- ANCHORS -->

The **ceiling is unmeasured**: no dataset far from COCO has been run through this yet, so the
upper half of the scale has no reference point. That is a gap, and the command's output says
so rather than implying the scale is calibrated.

## Alternatives evaluated and rejected

Recorded so the research is not repeated, and so nobody adopts one of these without
meeting the licence problem the hard way.

### LightlyTrain

**Not to be confused with `lightly`**, the library, which is MIT and *is* used here — see
"`lightly` is not LightlyTrain" above. The rejection below is about the AGPL framework only.

A mature framework for exactly stage 1 — distilling DINOv2/DINOv3 into an arbitrary student
backbone, with support for custom PyTorch models. Technically it is ahead of what is
implemented here.

**Rejected on licence.** It is AGPL-3.0 with a separate commercial licence, and the vendor is
explicit about which side this project falls on: *"Using LightlyTrain at work, in production,
on the edge, or to build proprietary models? You likely need a Commercial License."*

Whether model weights are a derivative work of the software that trained them is arguable.
"Arguable" is the problem: this entire effort exists to replace a licence asterisk, and using
a tool whose vendor claims an interest in the models it produces trades Deci's asterisk for
another one. The repository is Apache-2.0, and a development-only AGPL dependency still
complicates the licensing story being built here.

**Worth taking from it anyway**, since a method is not licensed:

- Their `distillationv2`, aimed at dense tasks, applies **MSE on the spatial features** rather
  than a cosine loss. Cosine was chosen here because scale invariance makes int8 quantisation
  free, but it discards magnitude, which may carry signal for detection. The cache keeps
  per-token norms in fp16 precisely so this can be tried without recaching.
- Their projection head is configurable in depth and width (`n_projection_layers`,
  `projection_hidden_dim`) where this implementation uses a single 1x1 convolution. A more
  expressive head can absorb feature-space mismatch that the backbone is otherwise forced to
  absorb itself. This is the cheapest experiment to run if the stage 1 gate is only narrowly
  missed, and it does not touch the cache.
- They apply **identical geometric augmentation to both sides**, with crops from 0.14 to 1.0
  of the image. That is exactly what the feature cache forbids, and is the clearest statement
  of what caching costs.

### DEIMv2 and EdgeCrafter

Both Intellindust, both commercial-by-request, and both hold the best numbers in their class
— DEIMv2-S reaches 50.9 AP at 9.7M parameters. Reading material, not components. EdgeCrafter's
paper is the origin of the DINOv3-teacher idea used in stage 1.

### SAM 3 as a box source

Its licence is permissive enough, but it is the wrong tool: SAM 3's notion of an instance is
its own, and a pseudo-labeller misaligned with COCO's annotation conventions injects a bias
the metric punishes. It earns its place in the segmentation stage instead, prompted with
boxes that a COCO-trained detector produced.

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
