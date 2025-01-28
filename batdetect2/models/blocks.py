"""Module containing custom NN blocks.

All these classes are subclasses of `torch.nn.Module` and can be used to build
complex neural network architectures.
"""

import sys
from typing import Iterable, List, Literal, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

if sys.version_info >= (3, 10):
    from itertools import pairwise
else:

    def pairwise(iterable: Sequence) -> Iterable:
        for x, y in zip(iterable[:-1], iterable[1:]):
            yield x, y


__all__ = [
    "ConvBlock",
    "ConvBlockDownCoordF",
    "ConvBlockDownStandard",
    "ConvBlockUpF",
    "ConvBlockUpStandard",
    "SelfAttention",
    "VerticalConv",
    "DownscalingLayer",
    "UpscalingLayer",
]


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


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pad_size: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=pad_size,
        )
        self.conv_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu_(self.conv_bn(self.conv(x)))


class VerticalConv(nn.Module):
    """Convolutional layer over full height.

    This layer applies a convolution that captures information across the
    entire height of the input image. It uses a kernel with the same height as
    the input, effectively condensing the vertical information into a single
    output row.

    More specifically:

    * **Input:**  (B, C, H, W) where B is the batch size, C is the number of
      input channels, H is the image height, and W is the image width.
    * **Kernel:** (C', H, 1) where C' is the number of output channels.
    * **Output:** (B, C', 1, W) - The height dimension is 1 because the
      convolution integrates information from all rows of the input.

    This process effectively extracts features that span the full height of
    the input image.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        input_height: int,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(input_height, 1),
            padding=0,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu_(self.bn(self.conv(x)))


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
        return F.relu(self.conv_bn(x), inplace=True)


DownscalingLayer = Literal["ConvBlockDownStandard", "ConvBlockDownCoordF"]


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


UpscalingLayer = Literal["ConvBlockUpStandard", "ConvBlockUpF"]


def build_downscaling_layer(
    in_channels: int,
    out_channels: int,
    input_height: int,
    layer_type: DownscalingLayer,
) -> nn.Module:
    if layer_type == "ConvBlockDownStandard":
        return ConvBlockDownStandard(
            in_channels=in_channels,
            out_channels=out_channels,
        )

    if layer_type == "ConvBlockDownCoordF":
        return ConvBlockDownCoordF(
            in_channels=in_channels,
            out_channels=out_channels,
            input_height=input_height,
        )

    raise ValueError(
        f"Invalid downscaling layer type {layer_type}. "
        f"Valid values: ConvBlockDownCoordF, ConvBlockDownStandard"
    )


class Encoder(nn.Module):
    def __init__(
        self,
        channels: Sequence[int] = (1, 32, 62, 128),
        input_height: int = 128,
        layer_type: Literal[
            "ConvBlockDownStandard", "ConvBlockDownCoordF"
        ] = "ConvBlockDownStandard",
    ):
        super().__init__()

        self.channels = channels
        self.input_height = input_height

        self.layers = nn.ModuleList(
            [
                build_downscaling_layer(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    input_height=input_height // (2**layer_num),
                    layer_type=layer_type,
                )
                for layer_num, (in_channels, out_channels) in enumerate(
                    pairwise(channels)
                )
            ]
        )
        self.depth = len(self.layers)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outputs = []

        for layer in self.layers:
            x = layer(x)
            outputs.append(x)

        return outputs


def build_upscaling_layer(
    in_channels: int,
    out_channels: int,
    input_height: int,
    layer_type: UpscalingLayer,
) -> nn.Module:
    if layer_type == "ConvBlockUpStandard":
        return ConvBlockUpStandard(
            in_channels=in_channels,
            out_channels=out_channels,
        )

    if layer_type == "ConvBlockUpF":
        return ConvBlockUpF(
            in_channels=in_channels,
            out_channels=out_channels,
            input_height=input_height,
        )

    raise ValueError(
        f"Invalid upscaling layer type {layer_type}. "
        f"Valid values: ConvBlockUpStandard, ConvBlockUpF"
    )


class Decoder(nn.Module):
    def __init__(
        self,
        channels: Sequence[int] = (256, 62, 32, 32),
        input_height: int = 128,
        layer_type: Literal[
            "ConvBlockUpStandard", "ConvBlockUpF"
        ] = "ConvBlockUpStandard",
    ):
        super().__init__()

        self.channels = channels
        self.input_height = input_height
        self.depth = len(self.channels) - 1

        self.layers = nn.ModuleList(
            [
                build_upscaling_layer(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    input_height=input_height
                    // (2 ** (self.depth - layer_num)),
                    layer_type=layer_type,
                )
                for layer_num, (in_channels, out_channels) in enumerate(
                    pairwise(channels)
                )
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        residuals: List[torch.Tensor],
    ) -> torch.Tensor:
        if len(residuals) != len(self.layers):
            raise ValueError(
                f"Incorrect number of residuals provided. "
                f"Expected {len(self.layers)} (matching the number of layers), "
                f"but got {len(residuals)}."
            )

        for layer, res in zip(self.layers, residuals[::-1]):
            x = layer(x + res)

        return x
