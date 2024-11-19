from typing import Tuple

import torch
import torch.fft
import torch.nn.functional as F
from torch import nn

from batdetect2.models.blocks import (
    ConvBlockDownCoordF,
    ConvBlockDownStandard,
    ConvBlockUpF,
    ConvBlockUpStandard,
    SelfAttention,
)
from batdetect2.models.typing import BackboneModel

__all__ = [
    "Net2DFast",
    "Net2DFastNoAttn",
    "Net2DFastNoCoordConv",
]


class Net2DFast(BackboneModel):
    def __init__(
        self,
        num_features: int,
        input_height: int = 128,
    ):
        super().__init__()
        self.num_features = num_features
        self.input_height = input_height
        self.bottleneck_height = self.input_height // 32

        # encoder
        self.conv_dn_0 = ConvBlockDownCoordF(
            1,
            self.num_features // 4,
            self.input_height,
            kernel_size=3,
            pad_size=1,
            stride=1,
        )
        self.conv_dn_1 = ConvBlockDownCoordF(
            self.num_features // 4,
            self.num_features // 2,
            self.input_height // 2,
            kernel_size=3,
            pad_size=1,
            stride=1,
        )
        self.conv_dn_2 = ConvBlockDownCoordF(
            self.num_features // 2,
            self.num_features,
            self.input_height // 4,
            kernel_size=3,
            pad_size=1,
            stride=1,
        )
        self.conv_dn_3 = nn.Conv2d(
            self.num_features,
            self.num_features * 2,
            3,
            padding=1,
        )
        self.conv_dn_3_bn = nn.BatchNorm2d(self.num_features * 2)

        # bottleneck
        self.conv_1d = nn.Conv2d(
            self.num_features * 2,
            self.num_features * 2,
            (self.input_height // 8, 1),
            padding=0,
        )
        self.conv_1d_bn = nn.BatchNorm2d(self.num_features * 2)
        self.att = SelfAttention(self.num_features * 2, self.num_features * 2)

        # decoder
        self.conv_up_2 = ConvBlockUpF(
            self.num_features * 2,
            self.num_features // 2,
            self.input_height // 8,
        )
        self.conv_up_3 = ConvBlockUpF(
            self.num_features // 2,
            self.num_features // 4,
            self.input_height // 4,
        )
        self.conv_up_4 = ConvBlockUpF(
            self.num_features // 4,
            self.num_features // 4,
            self.input_height // 2,
        )

        self.conv_op = nn.Conv2d(
            self.num_features // 4,
            self.num_features // 4,
            kernel_size=3,
            padding=1,
        )
        self.conv_op_bn = nn.BatchNorm2d(self.num_features // 4)

        self.out_channels = self.num_features // 4

    def pad_adjust(self, spec: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        h, w = spec.shape[2:]
        h_pad = (32 - h % 32) % 32
        w_pad = (32 - w % 32) % 32
        return F.pad(spec, (0, w_pad, 0, h_pad)), h_pad, w_pad

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        # encoder
        spec, h_pad, w_pad = self.pad_adjust(spec)

        x1 = self.conv_dn_0(spec)
        x2 = self.conv_dn_1(x1)
        x3 = self.conv_dn_2(x2)
        x3 = F.relu_(self.conv_dn_3_bn(self.conv_dn_3(x3)))

        # bottleneck
        x = F.relu_(self.conv_1d_bn(self.conv_1d(x3)))
        x = self.att(x)
        x = x.repeat([1, 1, self.bottleneck_height * 4, 1])

        # decoder
        x = self.conv_up_2(x + x3)
        x = self.conv_up_3(x + x2)
        x = self.conv_up_4(x + x1)

        # Restore original size
        if h_pad > 0:
            x = x[:, :, :-h_pad, :]

        if w_pad > 0:
            x = x[:, :, :, :-w_pad]

        return F.relu_(self.conv_op_bn(self.conv_op(x)))


class Net2DFastNoAttn(BackboneModel):
    def __init__(
        self,
        num_features: int,
        input_height: int = 128,
    ):
        super().__init__()

        self.num_features = num_features
        self.input_height = input_height
        self.bottleneck_height = self.input_height // 32

        self.conv_dn_0 = ConvBlockDownCoordF(
            1,
            self.num_features // 4,
            self.input_height,
            kernel_size=3,
            pad_size=1,
            stride=1,
        )
        self.conv_dn_1 = ConvBlockDownCoordF(
            self.num_features // 4,
            self.num_features // 2,
            self.input_height // 2,
            kernel_size=3,
            pad_size=1,
            stride=1,
        )
        self.conv_dn_2 = ConvBlockDownCoordF(
            self.num_features // 2,
            self.num_features,
            self.input_height // 4,
            kernel_size=3,
            pad_size=1,
            stride=1,
        )
        self.conv_dn_3 = nn.Conv2d(
            self.num_features,
            self.num_features * 2,
            3,
            padding=1,
        )
        self.conv_dn_3_bn = nn.BatchNorm2d(self.num_features * 2)

        self.conv_1d = nn.Conv2d(
            self.num_features * 2,
            self.num_features * 2,
            (self.input_height // 8, 1),
            padding=0,
        )
        self.conv_1d_bn = nn.BatchNorm2d(self.num_features * 2)

        self.conv_up_2 = ConvBlockUpF(
            self.num_features * 2,
            self.num_features // 2,
            self.input_height // 8,
        )
        self.conv_up_3 = ConvBlockUpF(
            self.num_features // 2,
            self.num_features // 4,
            self.input_height // 4,
        )
        self.conv_up_4 = ConvBlockUpF(
            self.num_features // 4,
            self.num_features // 4,
            self.input_height // 2,
        )

        self.conv_op = nn.Conv2d(
            self.num_features // 4,
            self.num_features // 4,
            kernel_size=3,
            padding=1,
        )
        self.conv_op_bn = nn.BatchNorm2d(self.num_features // 4)
        self.out_channels = self.num_features // 4

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        x1 = self.conv_dn_0(spec)
        x2 = self.conv_dn_1(x1)
        x3 = self.conv_dn_2(x2)
        x3 = F.relu_(self.conv_dn_3_bn(self.conv_dn_3(x3)))

        x = F.relu_(self.conv_1d_bn(self.conv_1d(x3)))
        x = x.repeat([1, 1, self.bottleneck_height * 4, 1])

        x = self.conv_up_2(x + x3)
        x = self.conv_up_3(x + x2)
        x = self.conv_up_4(x + x1)

        return F.relu_(self.conv_op_bn(self.conv_op(x)))


class Net2DFastNoCoordConv(BackboneModel):
    def __init__(
        self,
        num_features: int,
        input_height: int = 128,
    ):
        super().__init__()

        self.num_features = num_features
        self.input_height = input_height
        self.bottleneck_height = self.input_height // 32

        self.conv_dn_0 = ConvBlockDownStandard(
            1,
            self.num_features // 4,
            kernel_size=3,
            pad_size=1,
            stride=1,
        )
        self.conv_dn_1 = ConvBlockDownStandard(
            self.num_features // 4,
            self.num_features // 2,
            kernel_size=3,
            pad_size=1,
            stride=1,
        )
        self.conv_dn_2 = ConvBlockDownStandard(
            self.num_features // 2,
            self.num_features,
            kernel_size=3,
            pad_size=1,
            stride=1,
        )
        self.conv_dn_3 = nn.Conv2d(
            self.num_features,
            self.num_features * 2,
            3,
            padding=1,
        )
        self.conv_dn_3_bn = nn.BatchNorm2d(self.num_features * 2)

        self.conv_1d = nn.Conv2d(
            self.num_features * 2,
            self.num_features * 2,
            (self.input_height // 8, 1),
            padding=0,
        )
        self.conv_1d_bn = nn.BatchNorm2d(self.num_features * 2)

        self.att = SelfAttention(self.num_features * 2, self.num_features * 2)

        self.conv_up_2 = ConvBlockUpStandard(
            self.num_features * 2,
            self.num_features // 2,
            self.input_height // 8,
        )
        self.conv_up_3 = ConvBlockUpStandard(
            self.num_features // 2,
            self.num_features // 4,
            self.input_height // 4,
        )
        self.conv_up_4 = ConvBlockUpStandard(
            self.num_features // 4,
            self.num_features // 4,
            self.input_height // 2,
        )

        self.conv_op = nn.Conv2d(
            self.num_features // 4,
            self.num_features // 4,
            kernel_size=3,
            padding=1,
        )
        self.conv_op_bn = nn.BatchNorm2d(self.num_features // 4)
        self.out_channels = self.num_features // 4

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        x1 = self.conv_dn_0(spec)
        x2 = self.conv_dn_1(x1)
        x3 = self.conv_dn_2(x2)
        x3 = F.relu_(self.conv_dn_3_bn(self.conv_dn_3(x3)))

        x = F.relu_(self.conv_1d_bn(self.conv_1d(x3)))
        x = self.att(x)
        x = x.repeat([1, 1, self.bottleneck_height * 4, 1])

        x = self.conv_up_2(x + x3)
        x = self.conv_up_3(x + x2)
        x = self.conv_up_4(x + x1)

        return F.relu_(self.conv_op_bn(self.conv_op(x)))
