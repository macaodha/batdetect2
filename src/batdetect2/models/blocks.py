"""Reusable convolutional building blocks for BatDetect2 models.

This module provides a collection of ``torch.nn.Module`` subclasses that form
the fundamental building blocks for the encoder-decoder backbone used in
BatDetect2. All blocks follow a consistent interface: they store
``in_channels`` and ``out_channels`` as attributes and implement a
``get_output_height`` method that reports how a given input height maps to an
output height (e.g., halved by downsampling blocks, doubled by upsampling
blocks).

Available block families
------------------------
Standard blocks
    ``ConvBlock`` – convolution + batch normalisation + ReLU, no change in
    spatial resolution.

Downsampling blocks
    ``StandardConvDownBlock`` – convolution then 2×2 max-pooling, halves H
    and W.
    ``FreqCoordConvDownBlock`` – like ``StandardConvDownBlock`` but prepends
    a normalised frequency-coordinate channel before the convolution
    (CoordConv concept), helping filters learn frequency-position-dependent
    patterns.

Upsampling blocks
    ``StandardConvUpBlock`` – bilinear interpolation then convolution,
    doubles H and W.
    ``FreqCoordConvUpBlock`` – like ``StandardConvUpBlock`` but prepends a
    frequency-coordinate channel after upsampling.

Bottleneck blocks
    ``VerticalConv`` – 1-D convolution whose kernel spans the entire
    frequency axis, collapsing H to 1 whilst preserving W.
    ``SelfAttention`` – scaled dot-product self-attention along the time
    axis; typically follows a ``VerticalConv``.

Group block
    ``LayerGroup`` – chains several blocks sequentially into one unit,
    useful when a single encoder or decoder "stage" requires more than one
    operation.

Factory function
----------------
``build_layer`` creates any of the above blocks from the matching
configuration object (one of the ``*Config`` classes exported here), using
a discriminated-union ``name`` field to dispatch to the correct class.
"""

from typing import Annotated, Literal

import torch
import torch.nn.functional as F
from pydantic import Field
from torch import nn

from batdetect2.core import (
    BaseConfig,
    ImportConfig,
    Registry,
    add_import_config,
)

__all__ = [
    "BlockImportConfig",
    "ConvBlock",
    "LayerGroupConfig",
    "VerticalConv",
    "FreqCoordConvDownBlock",
    "StandardConvDownBlock",
    "FreqCoordConvUpBlock",
    "StandardConvUpBlock",
    "SelfAttention",
    "ConvConfig",
    "FreqCoordConvDownConfig",
    "StandardConvDownConfig",
    "FreqCoordConvUpConfig",
    "StandardConvUpConfig",
    "LayerConfig",
    "build_layer",
]


class Block(nn.Module):
    """Abstract base class for all BatDetect2 building blocks.

    Subclasses must set ``in_channels`` and ``out_channels`` as integer
    attributes so that factory functions can wire blocks together without
    inspecting configuration objects at runtime. They may also override
    ``get_output_height`` when the block changes the height dimension (e.g.
    downsampling or upsampling blocks).

    Attributes
    ----------
    in_channels : int
        Number of channels expected in the input tensor.
    out_channels : int
        Number of channels produced in the output tensor.
    """

    in_channels: int
    out_channels: int

    def get_output_height(self, input_height: int) -> int:
        """Return the output height for a given input height.

        The default implementation returns ``input_height`` unchanged,
        which is correct for blocks that do not alter spatial resolution.
        Override this in downsampling (returns ``input_height // 2``) or
        upsampling (returns ``input_height * 2``) subclasses.

        Parameters
        ----------
        input_height : int
            Height (number of frequency bins) of the input feature map.

        Returns
        -------
        int
            Height of the output feature map.
        """
        return input_height


block_registry: Registry[Block, [int, int]] = Registry("block")


@add_import_config(block_registry)
class BlockImportConfig(ImportConfig):
    """Use any callable as a model block.

    Set ``name="import"`` and provide a ``target`` pointing to any
    callable to use it instead of a built-in option.
    """

    name: Literal["import"] = "import"


class SelfAttentionConfig(BaseConfig):
    """Configuration for a ``SelfAttention`` block.

    Attributes
    ----------
    name : str
        Discriminator field; always ``"SelfAttention"``.
    attention_channels : int
        Dimensionality of the query, key, and value projections.
    temperature : float
        Scaling factor applied to the weighted values before the final
        linear projection. Defaults to ``1``.
    """

    name: Literal["SelfAttention"] = "SelfAttention"
    attention_channels: int
    temperature: float = 1


class SelfAttention(Block):
    """Self-attention block operating along the time axis.

    Applies a scaled dot-product self-attention mechanism across the time
    steps of an input feature map. Before attention is computed the height
    dimension (frequency axis) is expected to have been reduced to 1, e.g.
    by a preceding ``VerticalConv`` layer.

    For each time step the block computes query, key, and value projections
    with learned linear weights, then calculates attention weights from the
    query–key dot products divided by ``temperature × attention_channels``.
    The weighted sum of values is projected back to ``in_channels`` via a
    final linear layer, and the height dimension is restored so that the
    output shape matches the input shape.

    Parameters
    ----------
    in_channels : int
        Number of input channels (features per time step). The output will
        also have ``in_channels`` channels.
    attention_channels : int
        Dimensionality of the query, key, and value projections.
    temperature : float, default=1.0
        Divisor applied together with ``attention_channels`` when scaling
        the dot-product scores before softmax. Larger values produce softer
        (more uniform) attention distributions.

    Attributes
    ----------
    key_fun : nn.Linear
        Linear projection for keys.
    value_fun : nn.Linear
        Linear projection for values.
    query_fun : nn.Linear
        Linear projection for queries.
    pro_fun : nn.Linear
        Final linear projection applied to the attended values.
    temperature : float
        Scaling divisor used when computing attention scores.
    att_dim : int
        Dimensionality of the attention space (``attention_channels``).
    """

    def __init__(
        self,
        in_channels: int,
        attention_channels: int,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels

        # Note, does not encode position information (absolute or relative)
        self.temperature = temperature
        self.att_dim = attention_channels
        self.output_channels = in_channels

        self.key_fun = nn.Linear(in_channels, attention_channels)
        self.value_fun = nn.Linear(in_channels, attention_channels)
        self.query_fun = nn.Linear(in_channels, attention_channels)
        self.pro_fun = nn.Linear(attention_channels, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention along the time dimension.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape ``(B, C, 1, W)``. The height dimension
            must be 1 (i.e. the frequency axis should already have been
            collapsed by a preceding ``VerticalConv`` layer).

        Returns
        -------
        torch.Tensor
            Output tensor with the same shape ``(B, C, 1, W)`` as the
            input, with each time step updated by attended context from all
            other time steps.
        """

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

    def compute_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Return the softmax attention weight matrix.

        Useful for visualising which time steps attend to which others.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape ``(B, C, 1, W)``.

        Returns
        -------
        torch.Tensor
            Attention weight matrix with shape ``(B, W, W)``. Entry
            ``[b, i, j]`` is the attention weight that time step ``i``
            assigns to time step ``j`` in batch item ``b``.
        """
        x = x.squeeze(2).permute(0, 2, 1)

        key = torch.matmul(
            x, self.key_fun.weight.T
        ) + self.key_fun.bias.unsqueeze(0).unsqueeze(0)
        query = torch.matmul(
            x, self.query_fun.weight.T
        ) + self.query_fun.bias.unsqueeze(0).unsqueeze(0)

        kk_qq = torch.bmm(key, query.permute(0, 2, 1)) / (
            self.temperature * self.att_dim
        )
        att_weights = F.softmax(kk_qq, 1)
        return att_weights

    @block_registry.register(SelfAttentionConfig)
    @staticmethod
    def from_config(
        config: SelfAttentionConfig,
        input_channels: int,
        input_height: int,
    ) -> "SelfAttention":
        return SelfAttention(
            in_channels=input_channels,
            attention_channels=config.attention_channels,
            temperature=config.temperature,
        )


class ConvConfig(BaseConfig):
    """Configuration for a basic ConvBlock."""

    name: Literal["ConvBlock"] = "ConvBlock"
    """Discriminator field indicating the block type."""

    out_channels: int
    """Number of output channels."""

    kernel_size: int = 3
    """Size of the square convolutional kernel."""

    pad_size: int = 1
    """Padding size."""


class ConvBlock(Block):
    """Basic Convolutional Block.

    A standard building block consisting of a 2D convolution, followed by
    batch normalization and a ReLU activation function.

    Sequence: Conv2d -> BatchNorm2d -> ReLU.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input tensor.
    out_channels : int
        Number of channels produced by the convolution.
    kernel_size : int, default=3
        Size of the square convolutional kernel.
    pad_size : int, default=1
        Amount of padding added to preserve spatial dimensions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pad_size: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=pad_size,
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Conv -> BN -> ReLU.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape `(B, C_in, H, W)`.

        Returns
        -------
        torch.Tensor
            Output tensor, shape `(B, C_out, H, W)`.
        """
        return F.relu_(self.batch_norm(self.conv(x)))

    @block_registry.register(ConvConfig)
    @staticmethod
    def from_config(
        config: ConvConfig,
        input_channels: int,
        input_height: int,
    ):
        return ConvBlock(
            in_channels=input_channels,
            out_channels=config.out_channels,
            kernel_size=config.kernel_size,
            pad_size=config.pad_size,
        )


class VerticalConvConfig(BaseConfig):
    """Configuration for a ``VerticalConv`` block.

    Attributes
    ----------
    name : str
        Discriminator field; always ``"VerticalConv"``.
    channels : int
        Number of output channels produced by the vertical convolution.
    """

    name: Literal["VerticalConv"] = "VerticalConv"
    channels: int


class VerticalConv(Block):
    """Convolutional layer that aggregates features across the entire height.

    Applies a 2D convolution using a kernel with shape `(input_height, 1)`.
    This collapses the height dimension (H) to 1 while preserving the width (W),
    effectively summarizing features across the full vertical extent (e.g.,
    frequency axis) at each time step. Followed by BatchNorm and ReLU.

    Useful for summarizing frequency information before applying operations
    along the time axis (like SelfAttention).

    Parameters
    ----------
    in_channels : int
        Number of channels in the input tensor.
    out_channels : int
        Number of channels produced by the convolution.
    input_height : int
        The height (H dimension) of the input tensor. The convolutional kernel
        will be sized `(input_height, 1)`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        input_height: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(input_height, 1),
            padding=0,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Vertical Conv -> BN -> ReLU.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape `(B, C_in, H, W)`, where H must match the
            `input_height` provided during initialization.

        Returns
        -------
        torch.Tensor
            Output tensor, shape `(B, C_out, 1, W)`.
        """
        return F.relu_(self.bn(self.conv(x)))

    @block_registry.register(VerticalConvConfig)
    @staticmethod
    def from_config(
        config: VerticalConvConfig,
        input_channels: int,
        input_height: int,
    ):
        return VerticalConv(
            in_channels=input_channels,
            out_channels=config.channels,
            input_height=input_height,
        )


class FreqCoordConvDownConfig(BaseConfig):
    """Configuration for a FreqCoordConvDownBlock."""

    name: Literal["FreqCoordConvDown"] = "FreqCoordConvDown"
    """Discriminator field indicating the block type."""

    out_channels: int
    """Number of output channels."""

    kernel_size: int = 3
    """Size of the square convolutional kernel."""

    pad_size: int = 1
    """Padding size."""


class FreqCoordConvDownBlock(Block):
    """Downsampling Conv Block incorporating Frequency Coordinate features.

    This block implements a downsampling step (Conv2d + MaxPool2d) commonly
    used in CNN encoders. Before the convolution, it concatenates an extra
    channel representing the normalized vertical coordinate (frequency) to the
    input tensor.

    The purpose of adding coordinate features is to potentially help the
    convolutional filters become spatially aware, allowing them to learn
    patterns that might depend on the relative frequency position within the
    spectrogram.

    Sequence: Concat Coords -> Conv -> MaxPool -> BatchNorm -> ReLU.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input tensor.
    out_channels : int
        Number of output channels after the convolution.
    input_height : int
        Height (H dimension, frequency bins) of the input tensor to this block.
        Used to generate the coordinate features.
    kernel_size : int, default=3
        Size of the square convolutional kernel.
    pad_size : int, default=1
        Padding added before convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        input_height: int,
        kernel_size: int = 3,
        pad_size: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.coords = nn.Parameter(
            torch.linspace(-1, 1, input_height)[None, None, ..., None],
            requires_grad=False,
        )
        self.conv = nn.Conv2d(
            in_channels + 1,
            out_channels,
            kernel_size=kernel_size,
            padding=pad_size,
            stride=1,
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply CoordF -> Conv -> MaxPool -> BN -> ReLU.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape `(B, C_in, H, W)`, where H must match
            `input_height`.

        Returns
        -------
        torch.Tensor
            Output tensor, shape `(B, C_out, H/2, W/2)` (due to MaxPool).
        """
        freq_info = self.coords.repeat(x.shape[0], 1, 1, x.shape[3])
        x = torch.cat((x, freq_info), 1)
        x = F.max_pool2d(self.conv(x), 2, 2)
        x = F.relu(self.batch_norm(x), inplace=True)
        return x

    def get_output_height(self, input_height: int) -> int:
        return input_height // 2

    @block_registry.register(FreqCoordConvDownConfig)
    @staticmethod
    def from_config(
        config: FreqCoordConvDownConfig,
        input_channels: int,
        input_height: int,
    ):
        return FreqCoordConvDownBlock(
            in_channels=input_channels,
            out_channels=config.out_channels,
            input_height=input_height,
            kernel_size=config.kernel_size,
            pad_size=config.pad_size,
        )


class StandardConvDownConfig(BaseConfig):
    """Configuration for a StandardConvDownBlock."""

    name: Literal["StandardConvDown"] = "StandardConvDown"
    """Discriminator field indicating the block type."""

    out_channels: int
    """Number of output channels."""

    kernel_size: int = 3
    """Size of the square convolutional kernel."""

    pad_size: int = 1
    """Padding size."""


class StandardConvDownBlock(Block):
    """Standard Downsampling Convolutional Block.

    A basic downsampling block consisting of a 2D convolution, followed by
    2x2 max pooling, batch normalization, and ReLU activation.

    Sequence: Conv -> MaxPool -> BN -> ReLU.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input tensor.
    out_channels : int
        Number of output channels after the convolution.
    kernel_size : int, default=3
        Size of the square convolutional kernel.
    pad_size : int, default=1
        Padding added before convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pad_size: int = 1,
    ):
        super(StandardConvDownBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=pad_size,
            stride=1,
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        """Apply Conv -> MaxPool -> BN -> ReLU.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape `(B, C_in, H, W)`.

        Returns
        -------
        torch.Tensor
            Output tensor, shape `(B, C_out, H/2, W/2)`.
        """
        x = F.max_pool2d(self.conv(x), 2, 2)
        return F.relu(self.batch_norm(x), inplace=True)

    def get_output_height(self, input_height: int) -> int:
        return input_height // 2

    @block_registry.register(StandardConvDownConfig)
    @staticmethod
    def from_config(
        config: StandardConvDownConfig,
        input_channels: int,
        input_height: int,
    ):
        return StandardConvDownBlock(
            in_channels=input_channels,
            out_channels=config.out_channels,
            kernel_size=config.kernel_size,
            pad_size=config.pad_size,
        )


class FreqCoordConvUpConfig(BaseConfig):
    """Configuration for a FreqCoordConvUpBlock."""

    name: Literal["FreqCoordConvUp"] = "FreqCoordConvUp"
    """Discriminator field indicating the block type."""

    out_channels: int
    """Number of output channels."""

    kernel_size: int = 3
    """Size of the square convolutional kernel."""

    pad_size: int = 1
    """Padding size."""

    up_mode: str = "bilinear"
    """Interpolation mode for upsampling (e.g., "nearest", "bilinear")."""

    up_scale: tuple[int, int] = (2, 2)
    """Scaling factor for height and width during upsampling."""


class FreqCoordConvUpBlock(Block):
    """Upsampling Conv Block incorporating Frequency Coordinate features.

    This block implements an upsampling step  followed by a convolution,
    commonly used in CNN decoders. Before the convolution, it concatenates an
    extra channel representing the normalized vertical coordinate (frequency)
    of the *upsampled* feature map.

    The goal is to provide spatial awareness (frequency position) to the
    filters during the decoding/upsampling process.

    Sequence: Interpolate  -> Concat Coords -> Conv -> BatchNorm -> ReLU.

    Parameters
    ----------
    in_channels
        Number of channels in the input tensor (before upsampling).
    out_channels
        Number of output channels after the convolution.
    input_height
        Height (H dimension, frequency bins) of the tensor *before* upsampling.
        Used to calculate the height for coordinate feature generation after
        upsampling.
    kernel_size
        Size of the square convolutional kernel.
    pad_size
        Padding added before convolution.
    up_mode
        Interpolation mode for upsampling (e.g., "nearest", "bilinear",
        "bicubic").
    up_scale
        Scaling factor for height and width during upsampling
        (typically (2, 2)).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        input_height: int,
        kernel_size: int = 3,
        pad_size: int = 1,
        up_mode: str = "bilinear",
        up_scale: tuple[int, int] = (2, 2),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

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
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Interpolate -> Concat Coords -> Conv -> BN -> ReLU.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape `(B, C_in, H_in, W_in)`, where H_in should match
            `input_height` used during initialization.

        Returns
        -------
        torch.Tensor
            Output tensor, shape `(B, C_out, H_in * scale_h, W_in * scale_w)`.
        """
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
        op = F.relu(self.batch_norm(op), inplace=True)
        return op

    def get_output_height(self, input_height: int) -> int:
        return input_height * 2

    @block_registry.register(FreqCoordConvUpConfig)
    @staticmethod
    def from_config(
        config: FreqCoordConvUpConfig,
        input_channels: int,
        input_height: int,
    ):
        return FreqCoordConvUpBlock(
            in_channels=input_channels,
            out_channels=config.out_channels,
            input_height=input_height,
            kernel_size=config.kernel_size,
            pad_size=config.pad_size,
            up_mode=config.up_mode,
            up_scale=config.up_scale,
        )


class StandardConvUpConfig(BaseConfig):
    """Configuration for a StandardConvUpBlock."""

    name: Literal["StandardConvUp"] = "StandardConvUp"
    """Discriminator field indicating the block type."""

    out_channels: int
    """Number of output channels."""

    kernel_size: int = 3
    """Size of the square convolutional kernel."""

    pad_size: int = 1
    """Padding size."""

    up_mode: str = "bilinear"
    """Interpolation mode for upsampling (e.g., "nearest", "bilinear")."""

    up_scale: tuple[int, int] = (2, 2)
    """Scaling factor for height and width during upsampling."""


class StandardConvUpBlock(Block):
    """Standard Upsampling Convolutional Block.

    A basic upsampling block used in CNN decoders. It first upsamples the input
    feature map using interpolation, then applies a 2D convolution, batch
    normalization, and ReLU activation. Does not use coordinate features.

    Sequence: Interpolate -> Conv -> BN -> ReLU.

    Parameters
    ----------
    in_channels
        Number of channels in the input tensor (before upsampling).
    out_channels
        Number of output channels after the convolution.
    kernel_size
        Size of the square convolutional kernel.
    pad_size
        Padding added before convolution.
    up_mode
        Interpolation mode for upsampling (e.g., "nearest", "bilinear").
    up_scale
        Scaling factor for height and width during upsampling.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pad_size: int = 1,
        up_mode: str = "bilinear",
        up_scale: tuple[int, int] = (2, 2),
    ):
        super(StandardConvUpBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up_scale = up_scale
        self.up_mode = up_mode
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=pad_size,
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Interpolate -> Conv -> BN -> ReLU.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape `(B, C_in, H_in, W_in)`.

        Returns
        -------
        torch.Tensor
            Output tensor, shape `(B, C_out, H_in * scale_h, W_in * scale_w)`.
        """
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
        op = F.relu(self.batch_norm(op), inplace=True)
        return op

    def get_output_height(self, input_height: int) -> int:
        return input_height * 2

    @block_registry.register(StandardConvUpConfig)
    @staticmethod
    def from_config(
        config: StandardConvUpConfig,
        input_channels: int,
        input_height: int,
    ):
        return StandardConvUpBlock(
            in_channels=input_channels,
            out_channels=config.out_channels,
            kernel_size=config.kernel_size,
            pad_size=config.pad_size,
            up_mode=config.up_mode,
            up_scale=config.up_scale,
        )


class LayerGroupConfig(BaseConfig):
    """Configuration for a ``LayerGroup`` — a sequential chain of blocks.

    Use this when a single encoder or decoder stage needs more than one
    block. The blocks are executed in the order they appear in ``layers``,
    with channel counts and heights propagated automatically.

    Attributes
    ----------
    name : str
        Discriminator field; always ``"LayerGroup"``.
    layers : List[LayerConfig]
        Ordered list of block configurations to chain together.
    """

    name: Literal["LayerGroup"] = "LayerGroup"
    layers: list["LayerConfig"]


LayerConfig = Annotated[
    ConvConfig
    | FreqCoordConvDownConfig
    | StandardConvDownConfig
    | FreqCoordConvUpConfig
    | StandardConvUpConfig
    | SelfAttentionConfig
    | LayerGroupConfig,
    Field(discriminator="name"),
]
"""Type alias for the discriminated union of block configuration models."""


class LayerGroup(nn.Module):
    """Sequential chain of blocks that acts as a single composite block.

    Wraps multiple ``Block`` instances in an ``nn.Sequential`` container,
    exposing the same ``in_channels``, ``out_channels``, and
    ``get_output_height`` interface as a regular ``Block`` so it can be
    used transparently wherever a single block is expected.

    Instances are typically constructed by ``build_layer`` when given a
    ``LayerGroupConfig``; you rarely need to create them directly.

    Parameters
    ----------
    layers : list[Block]
        Pre-built block instances to chain, in execution order.
    input_height : int
        Height of the tensor entering the first block.
    input_channels : int
        Number of channels in the tensor entering the first block.

    Attributes
    ----------
    in_channels : int
        Number of input channels (taken from the first block).
    out_channels : int
        Number of output channels (taken from the last block).
    layers : nn.Sequential
        The wrapped sequence of block modules.
    """

    def __init__(
        self,
        layers: list[Block],
        input_height: int,
        input_channels: int,
    ):
        super().__init__()
        self.in_channels = input_channels
        self.out_channels = (
            layers[-1].out_channels if layers else input_channels
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass input through all blocks in sequence.

        Parameters
        ----------
        x : torch.Tensor
            Input feature map, shape ``(B, C_in, H, W)``.

        Returns
        -------
        torch.Tensor
            Output feature map after all blocks have been applied.
        """
        return self.layers(x)

    def get_output_height(self, input_height: int) -> int:
        """Compute the output height by propagating through all blocks.

        Parameters
        ----------
        input_height : int
            Height of the input feature map.

        Returns
        -------
        int
            Height after all blocks in the group have been applied.
        """
        for block in self.layers:
            input_height = block.get_output_height(input_height)  # type: ignore
        return input_height

    @block_registry.register(LayerGroupConfig)
    @staticmethod
    def from_config(
        config: LayerGroupConfig,
        input_channels: int,
        input_height: int,
    ):
        layers = []

        for layer_config in config.layers:
            layer = build_layer(
                input_height=input_height,
                in_channels=input_channels,
                config=layer_config,
            )
            layers.append(layer)
            input_height = layer.get_output_height(input_height)
            input_channels = layer.out_channels

        return LayerGroup(
            layers=layers,
            input_height=input_height,
            input_channels=input_channels,
        )


def build_layer(
    input_height: int,
    in_channels: int,
    config: LayerConfig,
) -> Block:
    """Build a block from its configuration object.

    Looks up the block class corresponding to ``config.name`` in the
    internal block registry and instantiates it with the given input
    dimensions. This is the standard way to construct blocks when
    assembling an encoder or decoder from a configuration file.

    Parameters
    ----------
    input_height : int
        Height (number of frequency bins) of the input tensor to this
        block. Required for blocks whose kernel size depends on the input
        height (e.g. ``VerticalConv``) and for coordinate-aware blocks.
    in_channels : int
        Number of channels in the input tensor to this block.
    config : LayerConfig
        A configuration object for the desired block type. The ``name``
        field selects the block class; remaining fields supply its
        parameters.

    Returns
    -------
    Block
        An initialised block module ready to be added to an
        ``nn.Sequential`` or ``nn.ModuleList``.

    Raises
    ------
    KeyError
        If ``config.name`` does not correspond to a registered block type.
    ValueError
        If the configuration parameters are invalid for the chosen block.
    """
    return block_registry.build(config, in_channels, input_height)
