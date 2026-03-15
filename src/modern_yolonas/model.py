"""Top-level YoloNAS model.

Usage:
    model = YoloNAS.from_config("yolo_nas_s", num_classes=80)
"""

from __future__ import annotations

from torch import nn, Tensor

from modern_yolonas.backbone.backbone import YoloNASBackbone
from modern_yolonas.neck.pan import YoloNASPANNeckWithC2
from modern_yolonas.head.dfl import NDFLHeads
from modern_yolonas.configs import CONFIGS


class YoloNAS(nn.Module):
    """YOLO-NAS object detection model.

    Attributes ``backbone``, ``neck``, ``heads`` mirror the super-gradients
    top-level structure for state_dict key compatibility.
    """

    def __init__(
        self,
        backbone: YoloNASBackbone,
        neck: YoloNASPANNeckWithC2,
        heads: NDFLHeads,
    ):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.heads = heads
        self._init_bn()

    def _init_bn(self, eps: float = 1e-3, momentum: float = 0.03):
        """Apply super-gradients-compatible BatchNorm settings globally."""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = eps
                m.momentum = momentum

    def forward(self, x: Tensor):
        features = self.backbone(x)
        p3, p4, p5 = self.neck(features)
        return self.heads((p3, p4, p5))

    @classmethod
    def from_config(cls, variant: str, num_classes: int = 80) -> "YoloNAS":
        cfg = CONFIGS[variant]

        backbone = YoloNASBackbone(
            in_channels=3,
            stem_out_channels=cfg["backbone"]["stem_out_channels"],
            stages_config=cfg["backbone"]["stages"],
            spp_output_channels=cfg["backbone"]["spp_output_channels"],
            spp_k=cfg["backbone"]["spp_k"],
        )

        neck = YoloNASPANNeckWithC2(
            in_channels=backbone.out_channels,
            **cfg["neck"],
        )

        heads = NDFLHeads(
            num_classes=num_classes,
            in_channels=tuple(neck.out_channels),
            **cfg["heads"],
        )

        return cls(backbone=backbone, neck=neck, heads=heads)
