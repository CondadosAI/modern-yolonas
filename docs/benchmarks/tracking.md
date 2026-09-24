# Tracking: ByteTrack, OC-SORT and Deep HM-SORT on SportsMOT

Every tracker here sees the same detections: they are computed once, cached, and
replayed through each configuration, so a difference in the metrics can only come
from association. The results are in [`tracking.json`](tracking.json), and the
tables below are printed from it by `examples/render_tracking_table.py`.

## The short answer

With real detections, **ByteTrack scores 53.7 HOTA against 47.6 for Deep HM-SORT's
default configuration**, with 1,885 ID switches against 3,172. It leads in every
sport.

Deep HM-SORT's appearance cue does not help. The per-box embeddings the detector
produces barely separate one player from another, so on ground-truth boxes the
appearance-aware default scores 3.6 HOTA *below* the same tracker running on motion
alone, and on real detections the two are within 0.1.

## What was measured

| | |
|:---|:---|
| Data | [SportsMOT](https://arxiv.org/abs/2304.05170) validation split, all 45 sequences (15 basketball, 15 football, 15 volleyball), 26,970 frames at 1280×720, 25 fps. **CC BY-NC 4.0**: fetched from [MCG-NJU/SportsMOT](https://huggingface.co/datasets/MCG-NJU/SportsMOT), never redistributed. Only metrics are committed here; the caches, which hold embeddings of the frames, are not. |
| Detector | `yolo_nas_l`, Deci's COCO weights (non-commercial, see the README), input 640, confidence floor 0.1, class `person`. Not fine-tuned on SportsMOT. |
| Two sources | **detector**: the boxes YOLO-NAS returns. **oracle**: the ground-truth boxes, with embeddings computed on them. Oracle removes detection error, so it isolates association. |
| Trackers | Deep HM-SORT as implemented here, in the configurations of `SWEEP` in `modern_yolonas/benchmarks/mot.py`. ByteTrack and OC-SORT from [roboflow/trackers](https://github.com/roboflow/trackers) 2.6.0 with library defaults. |
| Metrics | TrackEval 1.3.0, `MotChallenge2DBox`, HOTA / CLEAR / Identity, no preprocessing (SportsMOT has nothing for it to remove). |
| Runs | One run per configuration. The trackers are deterministic given the cache, so a rerun reproduces the numbers exactly; what one run cannot tell you is how they move with a different detector or dataset. |

## Real detections

### All 45 sequences

| Tracker | HOTA | AssA | DetA | IDF1 | MOTA | ID switches |
|:---|---:|---:|---:|---:|---:|---:|
| ByteTrack (roboflow/trackers) | **53.7** | **46.5** | 62.0 | **55.7** | 62.4 | **1885** |
| Deep HM-SORT, motion only, ByteTrack's thresholds | 49.3 | 38.5 | 63.3 | 51.2 | 66.3 | 2385 |
| Deep HM-SORT, motion only | 47.5 | 35.7 | 63.4 | 48.7 | 66.4 | 3080 |
| Deep HM-SORT, harmonic (its default) | 47.6 | 36.0 | 63.2 | 48.6 | 66.3 | 3172 |
| Deep HM-SORT, min fusion (Deep-EIoU) | 47.2 | 35.3 | 63.1 | 47.9 | 66.2 | 3299 |
| Deep HM-SORT, motion only, keep every tracklet | 46.5 | 34.2 | 63.3 | 47.8 | 66.2 | 3426 |
| Deep HM-SORT, harmonic, keep every tracklet | 47.1 | 35.3 | 63.1 | 48.8 | 66.2 | 3286 |
| OC-SORT (roboflow/trackers) | 45.7 | 35.5 | 58.9 | 46.1 | 60.3 | 2191 |

### HOTA by sport

| Tracker | basketball (15) | football (15) | volleyball (15) |
|:---|---:|---:|---:|
| ByteTrack (roboflow/trackers) | **43.9** | **63.2** | **55.6** |
| Deep HM-SORT, motion only, ByteTrack's thresholds | 39.0 | 57.6 | 52.9 |
| Deep HM-SORT, motion only | 38.5 | 55.3 | 49.3 |
| Deep HM-SORT, harmonic (its default) | 38.1 | 56.0 | 49.1 |
| Deep HM-SORT, min fusion (Deep-EIoU) | 37.1 | 55.9 | 48.8 |
| Deep HM-SORT, motion only, keep every tracklet | 37.6 | 54.1 | 48.4 |
| Deep HM-SORT, harmonic, keep every tracklet | 37.1 | 55.7 | 49.0 |
| OC-SORT (roboflow/trackers) | 38.6 | 51.6 | 47.4 |

**ByteTrack wins on association.** Its lead over Deep HM-SORT's default is 6.1 HOTA,
and the gap is in AssA (46.5 against 36.0), not DetA. It has 41% fewer ID switches.
It gives some coverage back for that: its default `track_activation_threshold` of 0.7
starts fewer tracks, which costs it MOTA (62.4 against 66.3) and a little DetA. HOTA
and IDF1 weigh identity as well as coverage, and on both it leads.

**The score floor explains less than a third of the gap.** Deep HM-SORT discards
detections under 0.4, ByteTrack keeps everything the detector returns. Moving Deep
HM-SORT to ByteTrack's thresholds (`motion-matched`: floor 0.1, new tracks from 0.7)
raises it from 47.5 to 49.3 HOTA. That closes 1.8 of the 6.2 points between the two
motion-only trackers. The other 4.4, most of it association (8.0 AssA), comes from
something else in how the two associate. The largest structural difference is that
ByteTrack predicts where a track moves with a Kalman filter, while Deep HM-SORT
compares against the last box it matched, expanded. That is the likely cause and it
is not isolated here: the two also differ in IoU against expansion IoU, in matching
thresholds and in how a track is confirmed.

**OC-SORT is last** on real detections, with the lowest DetA (58.9). Its defaults
require three consecutive frames to confirm a track, against ByteTrack's two, which
is consistent with it losing coverage; no configuration of it was tried beyond the
library defaults.

## Ground-truth boxes

### All 45 sequences

| Tracker | HOTA | AssA | DetA | IDF1 | MOTA | ID switches |
|:---|---:|---:|---:|---:|---:|---:|
| ByteTrack (roboflow/trackers) | 87.5 | 77.3 | 99.1 | 83.1 | 99.4 | 570 |
| Deep HM-SORT, motion only, ByteTrack's thresholds | 89.7 | 80.8 | 99.7 | 86.3 | 99.7 | 827 |
| Deep HM-SORT, motion only | 89.7 | 80.8 | 99.7 | 86.3 | 99.7 | 827 |
| Deep HM-SORT, harmonic (its default) | 86.1 | 74.7 | 99.2 | 82.7 | 99.7 | 900 |
| Deep HM-SORT, min fusion (Deep-EIoU) | 81.4 | 67.2 | 98.5 | 78.1 | 99.6 | 1093 |
| Deep HM-SORT, motion only, keep every tracklet | **90.7** | **82.6** | 99.7 | **88.9** | 99.7 | 892 |
| Deep HM-SORT, harmonic, keep every tracklet | 87.2 | 76.8 | 99.1 | 85.6 | 99.7 | 903 |
| OC-SORT (roboflow/trackers) | 83.8 | 71.9 | 97.6 | 79.0 | 97.8 | 915 |

### HOTA by sport

| Tracker | basketball (15) | football (15) | volleyball (15) |
|:---|---:|---:|---:|
| ByteTrack (roboflow/trackers) | 89.9 | 83.9 | 89.8 |
| Deep HM-SORT, motion only, ByteTrack's thresholds | 93.2 | 84.4 | 92.9 |
| Deep HM-SORT, motion only | 93.2 | 84.4 | 92.9 |
| Deep HM-SORT, harmonic (its default) | 87.3 | 83.6 | 88.3 |
| Deep HM-SORT, min fusion (Deep-EIoU) | 78.5 | 81.7 | 86.3 |
| Deep HM-SORT, motion only, keep every tracklet | **95.8** | 84.3 | 92.4 |
| Deep HM-SORT, harmonic, keep every tracklet | 89.6 | 83.9 | 88.9 |
| OC-SORT (roboflow/trackers) | 89.1 | 77.1 | 85.4 |

With perfect boxes the ranking reverses: Deep HM-SORT on motion alone beats ByteTrack
(89.7 against 87.5, and 90.7 keeping every tracklet). When every player is detected in
every frame, the last matched box, expanded, is a good enough guess of where the
player is, and keeping lost tracklets pays because nothing leaves the court for long.
That advantage does not survive real detections, which is the condition that matters
for a default. It is also why an oracle-only benchmark would have picked the wrong
tracker.

`motion-matched` equals `motion` here because every ground-truth box has confidence
1.0, so the thresholds it changes have nothing to act on.

## Why the appearance cue does not help

Deep HM-SORT's "Deep" is appearance: it fuses a motion cost with the cosine distance
between per-box embeddings. The paper uses a re-identification network trained on
player crops. This implementation uses the detector's own `c5` features pooled over
each box, which cost nothing extra but were never trained to tell two people apart.

On ground-truth boxes, where association is the only source of error, the appearance
cue costs HOTA rather than adding it: the harmonic fusion scores 86.1 against 89.7 for
motion alone, and Deep-EIoU's `min` fusion 81.4. On real detections the harmonic and
motion-only rows are within 0.1 HOTA of each other, because detection errors dominate.

[`tracking_separability.json`](tracking_separability.json) shows why, from the oracle
caches of all 45 sequences (295,573 boxes, every one matched to its ground-truth id):

| | same player, 1 s apart | two players, same frame |
|:---|---:|---:|
| median cosine distance | 0.059 | 0.092 |
| share under the 0.3 appearance gate | 99.7% | 98.2% |

Two *different* players in the same frame are nearly as close as the *same* player a
second later. Deep HM-SORT only trusts an appearance distance under
`appearance_threshold` (0.3), and 98.2% of different-player pairs pass that gate, so
the cue enters the fusion on almost every pair while carrying little information. The
probability that a same-player distance is smaller than a different-player one is
0.661 overall, where 0.5 is chance: 0.591 for basketball, 0.604 for volleyball, 0.789
for football. That ordering matches where appearance costs the most on oracle boxes,
5.9 HOTA in basketball, 4.6 in volleyball and 0.8 in football.

A re-identification model trained for the purpose would change this. Any `(N, D)`
array in `detections.data["embedding"]` is used by `DeepHMSort`; this benchmark did
not try one.

## Limitations

- **One dataset, one camera style.** SportsMOT is broadcast sport: fast motion,
  lookalike uniforms, a mostly steady camera. Pedestrian footage, drones or a fixed
  CCTV view may rank these trackers differently.
- **An off-the-shelf detector.** DetA is about 62 for every tracker but OC-SORT: a COCO
  `person` detector misses a lot of players at this confidence floor. A detector fine-tuned on the sport would
  raise every row and could narrow or widen the gaps between trackers.
- **Library defaults, no tuning.** ByteTrack and OC-SORT ran with roboflow/trackers'
  defaults and Deep HM-SORT with its own. `motion-matched` is the only attempt to
  equalise them. A per-tracker sweep could move any row.
- **The mechanism for the remaining gap is not isolated.** The Kalman filter is the
  largest difference between ByteTrack and Deep HM-SORT, and the likeliest cause. It
  was not tested on its own.
- **Validation split only.** SportsMOT's test annotations are withheld for its
  leaderboard, so these are not comparable to leaderboard numbers.
- **No latency.** Association costs are small next to detection and were not timed.
- **Sampled AUC.** The separability probability is estimated on 20,000 random pairs
  of each kind, seeded, so a rerun gives the same number.

## Reproduce

Caching needs a GPU; replaying and scoring are CPU only. On an A40 the oracle cache
took 11.5 minutes for all 45 sequences and the detector cache 44 minutes.

```bash
uv sync --extra mot
# SportsMOT val, CC BY-NC 4.0, fetched rather than redistributed
curl -L -o val.tar https://huggingface.co/datasets/MCG-NJU/SportsMOT/resolve/main/dataset/val.tar
tar xf val.tar --exclude='._*' --exclude='*/._*'

uv run yolonas benchmark-tracking cache --data val --source oracle --output runs/mot-cache
uv run yolonas benchmark-tracking cache --data val --source detector --output runs/mot-cache

# Every configuration, all sequences, then once per sport (split files ship with SportsMOT)
for group in all basketball football volleyball; do
  seqs=$([ "$group" = all ] || echo "--sequences $group-val.txt")
  for src in oracle detector; do
    uv run yolonas benchmark-tracking evaluate --data val --cache runs/mot-cache \
        --source $src $seqs --output runs/mot-final/$group
  done
done
uv run examples/render_tracking_table.py --runs runs/mot-final --output docs/benchmarks/tracking.json

uv run examples/tracking_separability.py --data val --cache runs/mot-cache/oracle \
    --splits splits_txt --output docs/benchmarks/tracking_separability.json
```

`{sport}-val.txt` is the sport's split file from SportsMOT intersected with the
validation sequences, 15 names each.

## References

- Zhang, Y., Sun, P., Jiang, Y., et al. *ByteTrack: Multi-Object Tracking by
  Associating Every Detection Box*. ECCV 2022. [arXiv:2110.06864](https://arxiv.org/abs/2110.06864)
- Cao, J., Pang, J., Weng, X., et al. *Observation-Centric SORT: Rethinking SORT for
  Robust Multi-Object Tracking*. CVPR 2023. [arXiv:2203.14360](https://arxiv.org/abs/2203.14360)
- Huang, H.-W., Yang, C.-Y., Sun, J., et al. *Iterative Scale-Up ExpansionIoU and Deep
  Features Association for Multi-Object Tracking in Sports*. 2023.
  [arXiv:2306.13074](https://arxiv.org/abs/2306.13074)
- Gran-Henriksen, M., Lindgaard, H. A., Kiss, G., Lindseth, F. *Deep HM-SORT: Enhancing
  Multi-Object Tracking in Sports with Deep Features, Harmonic Mean, and Expansion IOU*.
  2024. [arXiv:2406.12081](https://arxiv.org/abs/2406.12081)
- Cui, Y., Zeng, C., Zhao, X., et al. *SportsMOT: A Large Multi-Object Tracking Dataset
  in Multiple Sports Scenes*. ICCV 2023. [arXiv:2304.05170](https://arxiv.org/abs/2304.05170)
