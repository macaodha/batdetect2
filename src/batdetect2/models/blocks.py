"""Commonly used neural network building blocks for BatDetect2 models.

This module provides various reusable `torch.nn.Module` subclasses that form
the fundamental building blocks for constructing convolutional neural network
architectures, particularly encoder-decoder backbones used in BatDetect2.

It includes standard components like basic convolutional blocks (`ConvBlock`),
blocks incorporating downsampling (`StandardConvDownBlock`), and blocks with
upsampling (`StandardConvUpBlock`).

Additionally, it features specialized layers investigated in BatDetect2
research:

- `SelfAttention`: Applies self-attention along the time dimension, enabling
  the model to weigh information across the entire temporal context, often
  used in the bottleneck of an encoder-decoder.
- `FreqCoordConvDownBlock` / `FreqCoordConvUpBlock`: Implement the "CoordConv"
   concept by concatenating normalized frequency coordinate information as an
   extra channel to the input of convolutional layers. This explicitly provides
   spatial frequency information to filters, potentially enabling them to learn
   frequency-dependent patterns more effectively.

These blocks can be used directly in custom PyTorch model definitions or
assembled into larger architectures.

A unified factory function `build_layer` allows creating instances
of these blocks based on configuration objects.
"""

from typing import Annotated, List, Literal, Protocol, Tuple, Union

import torch
import torch.nn.functional as F
from pydantic import Field
from torch import nn

from batdetect2.core import Registry
from batdetect2.core.configs import BaseConfig

__all__ = [
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


class BlockProtocol(Protocol):
    def get_output_channels(self) -> int:
        raise NotImplementedError

    def get_output_height(self, input_height: int) -> int:
        return input_height


class Block(nn.Module, BlockProtocol): ...


block_registry: Registry[Block, [int, int]] = Registry("block")


class SelfAttentionConfig(BaseConfig):
    name: Literal["SelfAttention"] = "SelfAttention"
    attention_channels: int
    temperature: float = 1


class SelfAttention(Block):
    """Self-Attention mechanism operating along the time dimension.

    This module implements a scaled dot-product self-attention mechanism,
    specifically designed here to operate across the time steps of an input
    feature map, typically after spatial dimensions (like frequency) have been
    condensed or squeezed.

    By calculating attention weights between all pairs of time steps, it allows
    the model to capture long-range temporal dependencies and focus on relevant
    parts of the sequence. It's often employed in the bottleneck or
    intermediate layers of an encoder-decoder architecture to integrate global
    temporal context.

    The implementation uses linear projections to create query, key, and value
    representations, computes scaled dot-product attention scores, applies
    softmax, and produces an output by weighting the values according to the
    attention scores, followed by a final linear projection. Positional encoding
    is not explicitly included in this block.

    Parameters
    ----------
    in_channels : int
        Number of input channels (features per time step after spatial squeeze).
    attention_channels : int
        Number of channels for the query, key, and value projections. Also the
        dimension of the output projection's input.
    temperature : float, default=1.0
        Scaling factor applied *before* the final projection layer. Can be used
        to adjust the sharpness or focus of the attention mechanism, although
        scaling within the softmax (dividing by sqrt(dim)) is more common for
        standard transformers. Here it scales the weighted values.

    Attributes
    ----------
    key_fun : nn.Linear
        Linear layer for key projection.
    value_fun : nn.Linear
        Linear layer for value projection.
    query_fun : nn.Linear
        Linear layer for query projection.
    pro_fun : nn.Linear
        Final linear projection layer applied after attention weighting.
    temperature : float
        Scaling factor applied before final projection.
    att_dim : int
        Dimensionality of the attention space (`attention_channels`).
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
            Input tensor, expected shape `(B, C, H, W)`, where H is typically
            squeezed (e.g., H=1 after a `VerticalConv` or pooling) before
            applying attention along the W (time) dimension.

        Returns
        -------
        torch.Tensor
            Output tensor of the same shape as the input `(B, C, H, W)`, where
            attention has been applied across the W dimension.

        Raises
        ------
        RuntimeError
            If input tensor dimensions are incompatible with operations.
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

    def get_output_channels(self) -> int:
        return self.output_channels

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

    def get_output_channels(self) -> int:
        return self.conv.out_channels

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

    def get_output_channels(self) -> int:
        return self.conv.out_channels

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

    def get_output_channels(self) -> int:
        return self.conv.out_channels

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

    def get_output_channels(self) -> int:
        return self.conv.out_channels

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

    up_scale: Tuple[int, int] = (2, 2)
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
    in_channels : int
        Number of channels in the input tensor (before upsampling).
    out_channels : int
        Number of output channels after the convolution.
    input_height : int
        Height (H dimension, frequency bins) of the tensor *before* upsampling.
        Used to calculate the height for coordinate feature generation after
        upsampling.
    kernel_size : int, default=3
        Size of the square convolutional kernel.
    pad_size : int, default=1
        Padding added before convolution.
    up_mode : str, default="bilinear"
        Interpolation mode for upsampling (e.g., "nearest", "bilinear",
        "bicubic").
    up_scale : Tuple[int, int], default=(2, 2)
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

    def get_output_channels(self) -> int:
        return self.conv.out_channels

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

    up_scale: Tuple[int, int] = (2, 2)
    """Scaling factor for height and width during upsampling."""


class StandardConvUpBlock(Block):
    """Standard Upsampling Convolutional Block.

    A basic upsampling block used in CNN decoders. It first upsamples the input
    feature map using interpolation, then applies a 2D convolution, batch
    normalization, and ReLU activation. Does not use coordinate features.

    Sequence: Interpolate -> Conv -> BN -> ReLU.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input tensor (before upsampling).
    out_channels : int
        Number of output channels after the convolution.
    kernel_size : int, default=3
        Size of the square convolutional kernel.
    pad_size : int, default=1
        Padding added before convolution.
    up_mode : str, default="bilinear"
        Interpolation mode for upsampling (e.g., "nearest", "bilinear").
    up_scale : Tuple[int, int], default=(2, 2)
        Scaling factor for height and width during upsampling.
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
        super(StandardConvUpBlock, self).__init__()
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

    def get_output_channels(self) -> int:
        return self.conv.out_channels

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


LayerConfig = Annotated[
    Union[
        ConvConfig,
        FreqCoordConvDownConfig,
        StandardConvDownConfig,
        FreqCoordConvUpConfig,
        StandardConvUpConfig,
        SelfAttentionConfig,
        "LayerGroupConfig",
    ],
    Field(discriminator="name"),
]
"""Type alias for the discriminated union of block configuration models."""


class LayerGroupConfig(BaseConfig):
    name: Literal["LayerGroup"] = "LayerGroup"
    layers: List[LayerConfig]


class LayerGroup(nn.Module):
    """Standard implementation of the `LayerGroup` architecture."""

    def __init__(
        self,
        layers: list[Block],
        input_height: int,
        input_channels: int,
    ):
        super().__init__()
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

    def get_output_channels(self) -> int:
        return self.layers[-1].get_output_channels()  # type: ignore

    def get_output_height(self, input_height: int) -> int:
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
            input_channels = layer.get_output_channels()

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
    """Factory function to build a specific nn.Module block from its config.

    Takes configuration object (one of the types included in the `LayerConfig`
    union) and instantiates the corresponding nn.Module block with the correct
    parameters derived from the config and the current pipeline state
    (`input_height`, `in_channels`).

    It uses the `name` field within the `config` object to determine
    which block class to instantiate.

    Parameters
    ----------
    input_height : int
        Height (frequency bins) of the input tensor *to this layer*.
    in_channels : int
        Number of channels in the input tensor *to this layer*.
    config : LayerConfig
        A Pydantic configuration object for the desired block (e.g., an
        instance of `ConvConfig`, `FreqCoordConvDownConfig`, etc.), identified
        by its `name` field.

    Returns
    -------
    Tuple[nn.Module, int, int]
        A tuple containing:
        - The instantiated `nn.Module` block.
        - The number of output channels produced by the block.
        - The calculated height of the output produced by the block.

    Raises
    ------
    NotImplementedError
        If the `config.name` does not correspond to a known block type.
    ValueError
        If parameters derived from the config are invalid for the block.
    """
    return block_registry.build(config, in_channels, input_height)
