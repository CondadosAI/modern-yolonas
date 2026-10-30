# Tracking API

Every tracker takes an `sv.Detections` and returns the subset that matched a track, with
`tracker_id` filled in. `track_video` accepts anything that satisfies the `Tracker`
protocol below.

See the [tracking guide](../guides/tracking.md) for how the two trackers differ and the
[tracking benchmark](../benchmarks/tracking.md) for how they measure.

## The contract

::: modern_yolonas.tracking.Tracker

## ByteTrack (default)

Needs the `tracking` extra.

::: modern_yolonas.tracking.external.ByteTrack

::: modern_yolonas.tracking.external.OCSort

## Deep HM-SORT

::: modern_yolonas.tracking.DeepHMSort

## Track state

::: modern_yolonas.tracking.Track

::: modern_yolonas.tracking.TrackState

## Video helpers

::: modern_yolonas.inference.detect.YoloNASDetector.track_video

::: modern_yolonas.inference.detect.YoloNASDetector.track_video_to_file

::: modern_yolonas.inference.detect.YoloNASDetector.annotate_tracks

## Matching primitives

The cost functions are plain numpy and are usable on their own — to build a
different tracker, or to check what the tracker is seeing on a frame you care about.

::: modern_yolonas.tracking.matching.expand_boxes

::: modern_yolonas.tracking.matching.expansion_iou_distance

::: modern_yolonas.tracking.matching.cosine_distance

::: modern_yolonas.tracking.matching.harmonic_mean

::: modern_yolonas.tracking.matching.fuse_costs

::: modern_yolonas.tracking.matching.linear_assignment
