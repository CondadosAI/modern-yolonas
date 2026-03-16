# Architecture

## Overview

modern-yolonas follows the standard YOLO-NAS architecture:

```
Input (3, 640, 640)
  │
  ▼
Backbone (YoloNASBackbone)
  ├── Stem: Conv 3→48
  ├── Stage1: QARepVGG ↓2 → CSP → 96ch
  ├── Stage2: QARepVGG ↓2 → CSP → 192ch
  ├── Stage3: QARepVGG ↓2 → CSP → 384ch
  ├── Stage4: QARepVGG ↓2 → CSP → 768ch
  └── SPP: Spatial Pyramid Pooling → 768ch
  │
  │ outputs: [c2(96), c3(192), c4(384), c5(768)]
  ▼
Neck (YoloNASPANNeckWithC2)
  ├── neck1 (up): [c5, c4, c3] → upsample + concat + CSP
  ├── neck2 (up): [n1, c3, c2] → upsample + concat + CSP → p3
  ├── neck3 (down): [p3, n2_inter] → downsample + concat + CSP → p4
  └── neck4 (down): [p4, n1_inter] → downsample + concat + CSP → p5
  │
  │ outputs: [p3, p4, p5]
  ▼
Heads (NDFLHeads)
  ├── head1: p3 → cls(80) + reg(4×17) @ stride 8
  ├── head2: p4 → cls(80) + reg(4×17) @ stride 16
  └── head3: p5 → cls(80) + reg(4×17) @ stride 32
  │
  ▼
Output: [B, 8400, 4] bboxes + [B, 8400, 80] scores
```

## Key components

### QARepVGGBlock

The fundamental building block. During training it maintains three branches
(3x3 conv+BN, 1x1 conv, identity) that are fused into a single convolution
for inference. This enables quantization-aware training while keeping inference
fast.

### DFL Heads

Uses Distribution Focal Loss with `reg_max=16` — predicts a discrete
probability distribution over 17 offset bins per box edge, then reduces via
softmax + linear projection. This gives more precise box regression than
direct coordinate prediction.

### State dict compatibility

All attribute names (`backbone.stem`, `neck.neck1`, `heads.head1`, etc.)
exactly match the super-gradients module hierarchy, so pretrained checkpoints
load directly with only DDP/EMA prefix stripping.

## Variants

| Variant | `concat_intermediates` | Head `width_mult` | Params |
|---|---|---|---|
| S | False everywhere | 0.5 | ~12M |
| M | True in stages 1-3 | 0.75 | ~31M |
| L | True everywhere | 1.0 | ~44M |
