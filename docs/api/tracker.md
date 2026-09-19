# Tracking API

`DeepHMSort` is the built-in multi-object tracker. It takes an `sv.Detections` and
returns the subset that matched a track, with `tracker_id` filled in — the same
contract as `sv.ByteTrack.update_with_detections`, so anything already built around
supervision keeps working.

See the [tracking guide](../guides/tracking.md) for what the algorithm does, which
knobs matter, and what the defaults assume about your footage.

## Tracker

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
