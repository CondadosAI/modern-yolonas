# Image assets

## `street.jpg` — source photograph

A street scene in Porlamar, Venezuela, used as the sample input for the README banner and
as a known-good image for the quickstart examples.

- **Source:** [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Traffic_and_pedestrians_on_Calle_Vel%C3%A1squez_near_Catedral_San_Nicol%C3%A1s_de_Bari_in_Porlamar,_Venezuela.jpg)
- **Author:** [Wilfredor](https://commons.wikimedia.org/wiki/User:Wilfredor)
- **Licence:** [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) (public domain
  dedication — no attribution required, redistribution unrestricted)
- **Changes:** resized to 1600 px wide and re-encoded as JPEG quality 88.

CC0 is why this image is here rather than a COCO sample: COCO's images carry the licences of
their original Flickr uploads, which this repository cannot redistribute.

## `demo.jpg` — annotated banner

`street.jpg` with YOLO-NAS-L predictions drawn on it, at a confidence threshold of 0.40.
Regenerate with:

```bash
uv run python -c "
import cv2
from modern_yolonas import YoloNASDetector
img = cv2.imread('docs/assets/street.jpg')
det = YoloNASDetector('yolo_nas_l')
cv2.imwrite('docs/assets/demo.jpg', det.annotate(img, det(img, conf_threshold=0.40)),
            [cv2.IMWRITE_JPEG_QUALITY, 88])
"
```

The annotations inherit the licensing of the weights that produced them — see the
pretrained weights notice in the top-level README.

## `demo_video.gif` / `demo_video.mp4` — annotated video clip

Three seconds of the Shibuya crossing in Tokyo with YOLO-NAS-L predictions, at a
confidence threshold of 0.25 (the crowd is dense enough that the 0.40 used for the still
image leaves most of it unboxed). Roughly 70 detections per frame.

- **Source:** [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Shibuya_Crossing,_Tokyo,_Japan_(video).webm)
- **Author:** [Basile Morin](https://commons.wikimedia.org/wiki/User:Basile_Morin)
- **Licence:** [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
- **Changes:** 24 frames sampled from around 00:45 at every third frame, annotated,
  cropped to the crossing, scaled to 820 px wide, and encoded at 8 fps.

> **ShareAlike applies to these two files.** CC BY-SA 4.0 is a copyleft licence: because
> the clip and the video are derivative works of the source, they are themselves licensed
> **CC BY-SA 4.0**, not Apache-2.0. That obligation attaches to these media files only —
> it does not reach the source code, which stays Apache-2.0, and it does not reach anyone
> who merely uses the library. Anyone redistributing *these files* or a work derived from
> them must keep the attribution above and license their version under CC BY-SA 4.0.
> `street.jpg` and `demo.jpg` are CC0 and carry no such condition.

The `.gif` is what the README shows, because GitHub renders it inline without a player.
The `.mp4` is a quarter of the size at better quality and is what the documentation site
uses, where a `<video>` element works.

## `street_nyc.jpg` and `pancakes.jpg` — the retrieval example's gallery

The two gallery images for the embedding figure. One is another street scene, so it should
land near the query; the other is nothing like it, so it should land far away. Both are
CC0, like `street.jpg`, so they can be redistributed here.

### `street_nyc.jpg`

- **Source:** [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:People_crossing_street_(Unsplash).jpg)
  (originally [Unsplash](https://unsplash.com/photos/omi6C5fdiLA))
- **Author:** Mike Petrucci
- **Licence:** [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/)
- **Changes:** resized to 1400 px wide and re-encoded as JPEG quality 88.

### `pancakes.jpg`

- **Source:** [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Eating_Pancakes_(Unsplash).jpg)
  (originally [Unsplash](https://unsplash.com/photos/YpngzEY9ijY))
- **Author:** Gabriel Gurrola
- **Licence:** [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/)
- **Changes:** resized to 1400 px wide and re-encoded as JPEG quality 88.

## `embedding_space.png` — the retrieval figure

`street.jpg` as the query against the two images above, with the three vectors plotted at
their true angular distances. Regenerate with:

```bash
uv run examples/embedding_space_figure.py
```

The script prints the similarities it drew, so the figure and the numbers quoted around it
cannot drift apart.
