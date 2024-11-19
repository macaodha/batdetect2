"""Module containing custom NN blocks.

All these classes are subclasses of `torch.nn.Module` and can be used to build
complex neural network architectures.
"""

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from batdetect2.configs import BaseConfig

__all__ = [
    "SelfAttention",
    "ConvBlockDownCoordF",
    "ConvBlockDownStandard",
    "ConvBlockUpF",
    "ConvBlockUpStandard",
]


class SelfAttentionConfig(BaseConfig):
    temperature: float = 1.0
    input_channels: int = 128
    attention_channels: int = 128


class SelfAttention(nn.Module):
    """Self-Attention module.

    This module implements self-attention mechanism.
    """

    def __init__(
        self,
        in_channels: int,
        attention_channels: int,
        temperature: float = 1.0,
    ):
        super().__init__()

        # Note, does not encode position information (absolute or relative)
        self.temperature = temperature
        self.att_dim = attention_channels
        self.key_fun = nn.Linear(in_channels, attention_channels)
        self.value_fun = nn.Linear(in_channels, attention_channels)
        self.query_fun = nn.Linear(in_channels, attention_channels)
        self.pro_fun = nn.Linear(attention_channels, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze(2).permute(0, 2, 1)

        key = torch.matmul(
            x, self.key_fun.weight.T
        ) + self.key_fun.bias.unsqueeze(0).unsqueeze(0)
        query = torch.matmul(
            x, self.query_fun.weight.T
        ) + self.query_fun.bias.unsqueeze(0).unsqueeze(0)
        value = torch.matmul(
            x, self.value_fun.weight.T
        ) + self.value_fun.bias.unsqueeze(0).unsqueeze(0)

        kk_qq = torch.bmm(key, query.permute(0, 2, 1)) / (
            self.temperature * self.att_dim
        )
        att_weights = F.softmax(kk_qq, 1)
        att = torch.bmm(value.permute(0, 2, 1), att_weights)

        op = torch.matmul(
            att.permute(0, 2, 1), self.pro_fun.weight.T
        ) + self.pro_fun.bias.unsqueeze(0).unsqueeze(0)
        op = op.permute(0, 2, 1).unsqueeze(2)

        return op


class ConvBlockDownCoordFConfig(BaseConfig):
    in_channels: int
    out_channels: int
    input_height: int
    kernel_size: int = 3
    pad_size: int = 1
    stride: int = 1


class ConvBlockDownCoordF(nn.Module):
    """Convolutional Block with Downsampling and Coord Feature.

    This block performs convolution followed by downsampling
    and concatenates with coordinate information.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        input_height: int,
        kernel_size: int = 3,
        pad_size: int = 1,
        stride: int = 1,
    ):
        super().__init__()

        self.coords = nn.Parameter(
            torch.linspace(-1, 1, input_height)[None, None, ..., None],
            requires_grad=False,
        )
        self.conv = nn.Conv2d(
            in_channels + 1,
            out_channels,
            kernel_size=kernel_size,
            padding=pad_size,
            stride=stride,
        )
        self.conv_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        freq_info = self.coords.repeat(x.shape[0], 1, 1, x.shape[3])
        x = torch.cat((x, freq_info), 1)
        x = F.max_pool2d(self.conv(x), 2, 2)
        x = F.relu(self.conv_bn(x), inplace=True)
        return x


class ConvBlockDownStandardConfig(BaseConfig):
    in_channels: int
    out_channels: int
    kernel_size: int = 3
    pad_size: int = 1
    stride: int = 1


class ConvBlockDownStandard(nn.Module):
    """Convolutional Block with Downsampling.

    This block performs convolution followed by downsampling.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pad_size: int = 1,
        stride: int = 1,
    ):
        super(ConvBlockDownStandard, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=pad_size,
            stride=stride,
        )
        self.conv_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.max_pool2d(self.conv(x), 2, 2)
        x = F.relu(self.conv_bn(x), inplace=True)
        return x


class ConvBlockUpFConfig(BaseConfig):
    inp_channels: int
    out_channels: int
    input_height: int
    kernel_size: int = 3
    pad_size: int = 1
    up_mode: str = "bilinear"
    up_scale: Tuple[int, int] = (2, 2)


class ConvBlockUpF(nn.Module):
    """Convolutional Block with Upsampling and Coord Feature.

    This block performs convolution followed by upsampling
    and concatenates with coordinate information.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        input_height: int,
        kernel_size: int = 3,
        pad_size: int = 1,
        up_mode: str = "bilinear",
        up_scale: Tuple[int, int] = (2, 2),
    ):
        super().__init__()

        self.up_scale = up_scale
        self.up_mode = up_mode
        self.coords = nn.Parameter(
            torch.linspace(-1, 1, input_height * up_scale[0])[
                None, None, ..., None
            ],
            requires_grad=False,
        )
        self.conv = nn.Conv2d(
            in_channels + 1,
            out_channels,
            kernel_size=kernel_size,
            padding=pad_size,
        )
        self.conv_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        op = F.interpolate(
            x,
            size=(
                x.shape[-2] * self.up_scale[0],
                x.shape[-1] * self.up_scale[1],
            ),
            mode=self.up_mode,
            align_corners=False,
        )
        freq_info = self.coords.repeat(op.shape[0], 1, 1, op.shape[3])
        op = torch.cat((op, freq_info), 1)
        op = self.conv(op)
        op = F.relu(self.conv_bn(op), inplace=True)
        return op


class ConvBlockUpStandardConfig(BaseConfig):
    in_channels: int
    out_channels: int
    kernel_size: int = 3
    pad_size: int = 1
    up_mode: str = "bilinear"
    up_scale: Tuple[int, int] = (2, 2)


class ConvBlockUpStandard(nn.Module):
    """Convolutional Block with Upsampling.

    This block performs convolution followed by upsampling.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pad_size: int = 1,
        up_mode: str = "bilinear",
        up_scale: Tuple[int, int] = (2, 2),
    ):
        super(ConvBlockUpStandard, self).__init__()
        self.up_scale = up_scale
        self.up_mode = up_mode
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=pad_size,
        )
        self.conv_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        op = F.interpolate(
            x,
            size=(
                x.shape[-2] * self.up_scale[0],
                x.shape[-1] * self.up_scale[1],
            ),
            mode=self.up_mode,
            align_corners=False,
        )
        op = self.conv(op)
        op = F.relu(self.conv_bn(op), inplace=True)
        return op
