# Tracking

`modern-yolonas` ships with one tracker, **Deep HM-SORT**, and it is the default.
It reuses the per-object embeddings the detector already computes, so tracking with
appearance costs one forward pass per frame rather than the two a bolted-on
re-identification model would need.

<video src="../../assets/tracking_demo.mp4" autoplay loop muted playsinline width="100%">
  Your browser does not support the video element.
</video>

<sub>Boxes are coloured by track id rather than by class, so an ID-swap shows as a colour
change. 48 frames, 14 ids, 10 of them alive for at least half the clip. This clip is a
legibility demo, not evidence: with this few well-separated people, <code>--fusion min</code>
and <code>--no-appearance</code> produce identical output — see
<a href="#where-the-embeddings-come-from-and-what-that-means">below</a>.</sub>

```bash
yolonas track --source match.mp4 --classes 0
```

```python
from modern_yolonas import YoloNASDetector
from modern_yolonas.tracking import DeepHMSort

detector = YoloNASDetector("yolo_nas_s")
tracker = DeepHMSort()

for frame_index, frame, detections in detector.track_video("match.mp4", tracker):
    detections.tracker_id          # (N,) stable ids
    detections.data["embedding"]   # (N, 768) the vectors it associated on
```

`track_video` yields only the detections that matched a track. Everything else — the
confidence, the class, `data["class_name"]` — comes through untouched, so it slices
and annotates like any other `sv.Detections`.

## What the algorithm does

Deep HM-SORT ([arXiv:2406.12081](https://arxiv.org/abs/2406.12081)) is a two-change
delta on Deep-EIoU ([arXiv:2306.13074](https://arxiv.org/abs/2306.13074)), which is
itself a change to BoT-SORT. The frame loop:

1. **Split by score.** Detections at or above `track_high_threshold` (0.6) are
   strong; between `track_low_threshold` (0.4) and that, weak; below, discarded.
2. **First association** — every live track, including lost ones, against the strong
   detections. Cost is the fused motion and appearance cost, below. Unmatched
   tracks and detections go around again with the boxes expanded further
   (`expansion` 0.3, then 0.6).
3. **Second association** — whatever is still unmatched against the *weak*
   detections, on motion alone. This is the round that holds a track through a frame
   where the detector wavers. A crop that uncertain is not worth learning an
   identity from, so it does not update the track's appearance vector either.
4. **Unconfirmed** — tracks created last frame get one more chance before being
   dropped, so a single spurious detection does not leave a track behind.
5. **New tracks** from leftover detections scoring at least `new_track_threshold`.

There is **no Kalman filter**. Deep-EIoU drops it deliberately: a constant-velocity
Gaussian is a poor model of an athlete changing direction, and the prediction it
contributes is worth less than the expanded box it would be compared against. A
track's position is therefore the last box it actually matched — worth knowing if
you are coming from ByteTrack and expect a predicted one.

### Expansion IoU

Two boxes of the same object in consecutive frames can fail to overlap at all when
the object moves fast, and a plain IoU cost reads that as "different objects".
`expansion` grows both boxes by that fraction of their own width and height before
intersecting, which restores a usable signal without touching the detector's output.
An `expansion` of 0.3 makes a box 1.6× as wide and as tall.

The two rounds are a scale-up: match what you can at the tight radius, then widen it
for whatever is left, rather than starting wide and letting nearby objects compete.

### The harmonic mean

This is the paper's first contribution. For a track `j` and detection `k` with motion
distance `d1` and appearance distance `d2`, the association cost is

```
H(j, k) = 2 / (1/d1 + 1/d2)
```

where Deep-EIoU used `min(d1, d2)`. Both lean towards the smaller of the two, but
`min` *discards* the larger one. Two players in the same kit have near-identical
embeddings; `min` will happily match the wrong one on appearance alone, while the
harmonic mean lets the motion cost veto it.

Two gates run first, both inherited from BoT-SORT: an appearance distance above
`appearance_threshold` (0.3) is not evidence of anything, and an appearance distance
is not *trusted* for a pair that is nowhere near each other geometrically
(`proximity_threshold`, 0.5). A pair that fails either gate falls back to the motion
cost alone.

!!! note "A decision the paper does not make"
    Falling back is not the only option, and the alternative is worse in a way that
    is easy to miss: `H(d, 1) = 2d / (d + 1)`, which is **larger** than `d` for every
    `d < 1`. Fusing a gated pair against a placeholder would charge it for evidence
    that was never available, and because the gate fires on exactly the distant pairs
    the expansion scale-up exists to reach, it would quietly cancel the scale-up — a
    `match_threshold` of 0.8 would start refusing at a motion distance of 0.67. There
    is a test pinning the fallback for that reason.

You can measure the difference yourself rather than take it on faith:

```bash
yolonas track --source match.mp4 --fusion min       # Deep-EIoU's original
yolonas track --source match.mp4 --fusion harmonic  # the default
```

### Keeping every tracklet

The paper's second contribution: lost tracks are never discarded, so an object that
leaves the frame and comes back is re-identified instead of renumbered. That is the
default here (`max_lost=None`).

It is a **sports assumption** — a fixed camera on a closed pitch, where a player who
walks off returns to roughly where they left. On open-world footage the pool grows
with every object that has ever appeared, and both memory and the cost matrix grow
with it. Set a frame budget there:

```bash
yolonas track --source doorway.mp4 --max-lost 300   # 10 s at 30 fps
```

## Thresholds worth knowing

| Option | Default | What it decides |
|:--|--:|:--|
| `track_low_threshold` | 0.4 | Below this a detection does not exist |
| `track_high_threshold` | 0.6 | At or above, it competes in the first round |
| `new_track_threshold` | 0.5 | Lowest score that may start a track |
| `match_threshold` | 0.8 | Cost ceiling in the first round |
| `appearance_threshold` | 0.3 | Above this, appearance is not evidence |
| `proximity_threshold` | 0.5 | Above this motion distance, appearance is not trusted |
| `expansion` / `expansion_step` | 0.3 / 0.3 | Box growth, first round then second |
| `feature_momentum` | 0.9 | How slowly a track's appearance vector changes |

The one that catches people: **the detector has to run at or below
`track_low_threshold`**, or the weak band never arrives and the second association
round does nothing. `track_video` and `yolonas track` set the detector's threshold
from the tracker for you; if you drive the loop yourself, set it.

The paper's own values are 0.4, 0.6, 0.8 and 0.3 as above. Its 0.5 is described as
"the threshold below which we discard tracks", which fits both `new_track_threshold`
and `proximity_threshold`; since Deep-EIoU's default for the latter is also 0.5, the
two readings agree on the configuration shipped here.

## Where the embeddings come from, and what that means

The "Deep" in the paper is an OSNet re-identification model trained on player crops.
What this tracker gets instead is the YOLO-NAS `c5` features pooled over each box —
the same vectors described in the [embeddings guide](embeddings.md), which are a
by-product of detection rather than a representation trained to tell two people
apart.

On a COCO photo those vectors separate classes (0.879 same-class cosine against
0.758 cross-class), but nobody here has measured recall@k on a re-identification
benchmark. So: **the paper's HOTA and ID-switch numbers are a claim about their
embedding, not about this one, and they are not inherited.** The fusion, the
expansion scale-up and the retention are implemented as described; how well the
appearance cue performs on your footage is an open question you should answer with
your own data.

A first data point, from the demo clip at the top of this page: `--fusion harmonic`,
`--fusion min` and `--no-appearance` produce **identical** output on it — the same 14 ids
with the same lifetimes. Fourteen well-separated pedestrians is a scene where motion
settles every association on its own, so neither the appearance cue nor the choice of
fusion ever gets a say. That is not a criticism of either; it is a reminder that the
difference only appears where the paper says it does, in a crowd of lookalikes, and that
you should check which regime your footage is in before tuning anything.

Two ways to get a better answer:

```python
# 1. A different feature map, or several, without leaving the single pass.
from modern_yolonas import FeaturePooler, YoloNASDetector

detector = YoloNASDetector("yolo_nas_s", embedding=FeaturePooler(layers=("c4", "c5")))
```

```python
# 2. Your own re-identification model. Anything (N, D) in this key is associated on.
detections.data["embedding"] = my_reid_model(crops_of(detections))
tracked = tracker.update_with_detections(detections)
```

Without the key at all, the tracker associates on motion alone — HM-SORT without the
"Deep". That is what `--no-appearance` does, and it is a fair baseline to compare
against before concluding the appearance cue is helping.

## Reading the output

```bash
yolonas track --source match.mp4 --output results --show-fps
```

Boxes are coloured by `tracker_id` rather than by class, so an ID-swap shows up as a
colour change instead of hiding inside a row of identical labels. The summary line
reports `unique_ids`: the number of distinct objects the tracker believes it saw.
Far above the truth means ids are fragmenting — raise `max_lost` if you set one, or
widen `expansion`.

## Diagnosing a swap

The matching primitives are plain numpy and importable on their own, so you can ask
what the tracker saw on the frame that went wrong:

```python
import numpy as np
from modern_yolonas.tracking.matching import cosine_distance, expansion_iou_distance, fuse_costs

track_boxes = np.stack([t.xyxy for t in tracker.tracks])
motion = expansion_iou_distance(track_boxes, detections.xyxy, 0.3)
appearance = cosine_distance(
    np.stack([t.feature for t in tracker.tracks]), detections.data["embedding"]
)
fuse_costs(motion, appearance, proximity_threshold=0.5, appearance_threshold=0.3)
```

If the appearance matrix is near-uniform, the cue is carrying no information on your
footage and the motion cost is doing all the work — which is worth knowing before
tuning anything else.

## Citation

```bibtex
@article{granhenriksen2024deephmsort,
  title  = {Deep HM-SORT: Enhancing Multi-Object Tracking in Sports with Deep
            Features, Harmonic Mean, and Expansion IOU},
  author = {Gran-Henriksen, Matias and Lindg{\aa}rd, Hans Andreas and
            Kiss, Gabriel and Lindseth, Frank},
  journal = {arXiv preprint arXiv:2406.12081},
  year   = {2024}
}
```
