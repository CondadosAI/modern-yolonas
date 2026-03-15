"""YoloNAS PAN (Path Aggregation Network) neck with C2.

Attribute names ``neck1`` … ``neck4`` mirror super-gradients
``YoloNASPANNeckWithC2``.
"""

from __future__ import annotations

from functools import partial
from typing import Type

import torch
from torch import nn, Tensor

from modern_yolonas.nn.blocks import Conv, width_multiplier
from modern_yolonas.nn.repvgg import QARepVGGBlock
from modern_yolonas.backbone.stage import YoloNASCSPLayer


class YoloNASUpStage(nn.Module):
    """Upsample stage: conv → upsample → concat with skip(s) → CSP.

    Attributes: ``conv``, ``upsample``, ``reduce_skip`` or ``reduce_skip1``/``reduce_skip2``,
    ``downsample`` (3-input), ``reduce_after_concat``, ``blocks``.
    """

    def __init__(
        self,
        in_channels: list[int],
        out_channels: int,
        width_mult: float,
        num_blocks: int,
        depth_mult: float,
        activation_type: Type[nn.Module] = nn.ReLU,
        hidden_channels: int | None = None,
        concat_intermediates: bool = False,
        reduce_channels: bool = False,
        drop_path_rates: list[float] | None = None,
        dropout_rate: float = 0.0,
    ):
        super().__init__()

        num_inputs = len(in_channels)
        if num_inputs == 2:
            in_ch, skip_in_channels = in_channels
        else:
            in_ch, skip_in_channels1, skip_in_channels2 = in_channels
            skip_in_channels = skip_in_channels1 + out_channels

        out_channels = width_multiplier(out_channels, width_mult, 8)
        num_blocks = max(round(num_blocks * depth_mult), 1) if num_blocks > 1 else num_blocks

        if num_inputs == 2:
            self.reduce_skip = Conv(skip_in_channels, out_channels, 1, 1, activation_type) if reduce_channels else nn.Identity()
        else:
            self.reduce_skip1 = Conv(skip_in_channels1, out_channels, 1, 1, activation_type) if reduce_channels else nn.Identity()
            self.reduce_skip2 = Conv(skip_in_channels2, out_channels, 1, 1, activation_type) if reduce_channels else nn.Identity()

        self.conv = Conv(in_ch, out_channels, 1, 1, activation_type)

        # ConvTranspose2d for upsampling (matches super-gradients CONV_TRANSPOSE mode)
        self.upsample = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)

        if num_inputs == 3:
            downsample_in = out_channels if reduce_channels else skip_in_channels2
            self.downsample = Conv(downsample_in, out_channels, kernel=3, stride=2, activation_type=activation_type)

        self.reduce_after_concat = Conv(num_inputs * out_channels, out_channels, 1, 1, activation_type) if reduce_channels else nn.Identity()

        after_concat_channels = out_channels if reduce_channels else out_channels + skip_in_channels
        self.blocks = YoloNASCSPLayer(
            after_concat_channels,
            out_channels,
            num_blocks,
            QARepVGGBlock,
            activation_type,
            hidden_channels=hidden_channels,
            concat_intermediates=concat_intermediates,
            drop_path_rates=drop_path_rates,
            dropout_rate=dropout_rate,
        )

        self._out_channels = [out_channels, out_channels]
        self._num_inputs = num_inputs

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, inputs: list[Tensor]) -> tuple[Tensor, Tensor]:
        if self._num_inputs == 2:
            x, skip_x = inputs
            skip_x = [self.reduce_skip(skip_x)]
        else:
            x, skip_x1, skip_x2 = inputs
            skip_x1, skip_x2 = self.reduce_skip1(skip_x1), self.reduce_skip2(skip_x2)
            skip_x = [skip_x1, self.downsample(skip_x2)]

        x_inter = self.conv(x)
        x = self.upsample(x_inter)
        x = torch.cat([x, *skip_x], 1)
        x = self.reduce_after_concat(x)
        x = self.blocks(x)
        return x_inter, x


class YoloNASDownStage(nn.Module):
    """Downsample stage: Conv(stride=2) → concat with skip → CSP.

    Attributes: ``conv``, ``blocks``.
    """

    def __init__(
        self,
        in_channels: list[int],
        out_channels: int,
        width_mult: float,
        num_blocks: int,
        depth_mult: float,
        activation_type: Type[nn.Module] = nn.ReLU,
        hidden_channels: int | None = None,
        concat_intermediates: bool = False,
        drop_path_rates: list[float] | None = None,
        dropout_rate: float = 0.0,
    ):
        super().__init__()

        in_ch, skip_in_channels = in_channels
        out_channels = width_multiplier(out_channels, width_mult, 8)
        num_blocks = max(round(num_blocks * depth_mult), 1) if num_blocks > 1 else num_blocks

        self.conv = Conv(in_ch, out_channels // 2, 3, 2, activation_type)
        after_concat_channels = out_channels // 2 + skip_in_channels
        self.blocks = YoloNASCSPLayer(
            in_channels=after_concat_channels,
            out_channels=out_channels,
            num_bottlenecks=num_blocks,
            block_type=partial(Conv, kernel=3, stride=1),
            activation_type=activation_type,
            hidden_channels=hidden_channels,
            concat_intermediates=concat_intermediates,
            drop_path_rates=drop_path_rates,
            dropout_rate=dropout_rate,
        )

        self._out_channels = out_channels

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, inputs: list[Tensor]) -> Tensor:
        x, skip_x = inputs
        x = self.conv(x)
        x = torch.cat([x, skip_x], 1)
        return self.blocks(x)


class YoloNASPANNeckWithC2(nn.Module):
    """PAN neck with 4 stages (2 up + 2 down), C2-aware.

    Input:  ``[c2, c3, c4, c5]`` from backbone.
    Output: ``[p3, p4, p5]``.

    Attributes: ``neck1``, ``neck2``, ``neck3``, ``neck4``.
    """

    def __init__(
        self,
        in_channels: list[int],
        neck1_config: dict,
        neck2_config: dict,
        neck3_config: dict,
        neck4_config: dict,
        activation_type: Type[nn.Module] = nn.ReLU,
    ):
        super().__init__()

        c2_ch, c3_ch, c4_ch, c5_ch = in_channels

        # neck1: up — [c5, c4, c3] → (inter, out)
        self.neck1 = YoloNASUpStage(
            in_channels=[c5_ch, c4_ch, c3_ch],
            activation_type=activation_type,
            **neck1_config,
        )
        n1_inter_ch, n1_out_ch = self.neck1.out_channels

        # neck2: up — [neck1_out, c3, c2] → (inter, out=p3)
        self.neck2 = YoloNASUpStage(
            in_channels=[n1_out_ch, c3_ch, c2_ch],
            activation_type=activation_type,
            **neck2_config,
        )
        n2_inter_ch, n2_out_ch = self.neck2.out_channels

        # neck3: down — [p3, neck2_inter] → p4
        self.neck3 = YoloNASDownStage(
            in_channels=[n2_out_ch, n2_inter_ch],
            activation_type=activation_type,
            **neck3_config,
        )

        # neck4: down — [p4, neck1_inter] → p5
        self.neck4 = YoloNASDownStage(
            in_channels=[self.neck3.out_channels, n1_inter_ch],
            activation_type=activation_type,
            **neck4_config,
        )

        self._out_channels = [
            n2_out_ch,
            self.neck3.out_channels,
            self.neck4.out_channels,
        ]

    @property
    def out_channels(self) -> list[int]:
        return self._out_channels

    def forward(self, inputs: tuple[Tensor, Tensor, Tensor, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        c2, c3, c4, c5 = inputs

        x_n1_inter, x = self.neck1([c5, c4, c3])
        x_n2_inter, p3 = self.neck2([x, c3, c2])
        p4 = self.neck3([p3, x_n2_inter])
        p5 = self.neck4([p4, x_n1_inter])

        return p3, p4, p5
