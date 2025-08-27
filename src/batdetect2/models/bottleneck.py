"""Defines the Bottleneck component of an Encoder-Decoder architecture.

This module provides the configuration (`BottleneckConfig`) and
`torch.nn.Module` implementations (`Bottleneck`, `BottleneckAttn`) for the
bottleneck layer(s) that typically connect the Encoder (downsampling path) and
Decoder (upsampling path) in networks like U-Nets.

The bottleneck processes the lowest-resolution, highest-dimensionality feature
map produced by the Encoder. This module offers a configurable option to include
a `SelfAttention` layer within the bottleneck, allowing the model to capture
global temporal context before features are passed to the Decoder.

A factory function `build_bottleneck` constructs the appropriate bottleneck
module based on the provided configuration.
"""

from typing import Annotated, List, Optional, Union

import torch
from pydantic import Field
from torch import nn

from batdetect2.configs import BaseConfig
from batdetect2.models.blocks import (
    LayerConfig,
    SelfAttentionConfig,
    VerticalConv,
    build_layer_from_config,
)

__all__ = [
    "BottleneckConfig",
    "Bottleneck",
    "build_bottleneck",
]


class Bottleneck(nn.Module):
    """Base Bottleneck module for Encoder-Decoder architectures.

    This implementation represents the simplest bottleneck structure
    considered, primarily consisting of a `VerticalConv` layer. This layer
    collapses the frequency dimension (height) to 1, summarizing information
    across frequencies at each time step. The output is then repeated along the
    height dimension to match the original bottleneck input height before being
    passed to the decoder.

    This base version does *not* include self-attention.

    Parameters
    ----------
    input_height : int
        Height (frequency bins) of the input tensor. Must be positive.
    in_channels : int
        Number of channels in the input tensor from the encoder. Must be
        positive.
    out_channels : int
        Number of output channels. Must be positive.

    Attributes
    ----------
    in_channels : int
        Number of input channels accepted by the bottleneck.
    input_height : int
        Expected height of the input tensor.
    channels : int
        Number of output channels.
    conv_vert : VerticalConv
        The vertical convolution layer.

    Raises
    ------
    ValueError
        If `input_height`, `in_channels`, or `out_channels` are not positive.
    """

    def __init__(
        self,
        input_height: int,
        in_channels: int,
        out_channels: int,
        bottleneck_channels: Optional[int] = None,
        layers: Optional[List[torch.nn.Module]] = None,
    ) -> None:
        """Initialize the base Bottleneck layer."""
        super().__init__()
        self.in_channels = in_channels
        self.input_height = input_height
        self.out_channels = out_channels
        self.bottleneck_channels = (
            bottleneck_channels
            if bottleneck_channels is not None
            else out_channels
        )
        self.layers = nn.ModuleList(layers or [])

        self.conv_vert = VerticalConv(
            in_channels=in_channels,
            out_channels=self.bottleneck_channels,
            input_height=input_height,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input features through the bottleneck.

        Applies vertical convolution and repeats the output height.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor from the encoder bottleneck, shape
            `(B, C_in, H_in, W)`. `C_in` must match `self.in_channels`,
            `H_in` must match `self.input_height`.

        Returns
        -------
        torch.Tensor
            Output tensor, shape `(B, C_out, H_in, W)`. Note that the height
            dimension `H_in` is restored via repetition after the vertical
            convolution.
        """
        x = self.conv_vert(x)

        for layer in self.layers:
            x = layer(x)

        return x.repeat([1, 1, self.input_height, 1])


BottleneckLayerConfig = Annotated[
    Union[SelfAttentionConfig,],
    Field(discriminator="block_type"),
]
"""Type alias for the discriminated union of block configs usable in Decoder."""


class BottleneckConfig(BaseConfig):
    """Configuration for the bottleneck layer(s).

    Defines the number of channels within the bottleneck and whether to include
    a self-attention mechanism.

    Attributes
    ----------
    channels : int
        The number of output channels produced by the main convolutional layer
        within the bottleneck. This often matches the number of channels coming
        from the last encoder stage, but can be different. Must be positive.
        This also defines the channel dimensions used within the optional
        `SelfAttention` layer.
    self_attention : bool
        If True, includes a `SelfAttention` layer operating on the time
        dimension after an initial `VerticalConv` layer within the bottleneck.
        If False, only the initial `VerticalConv` (and height repetition) is
        performed.
    """

    channels: int
    layers: List[BottleneckLayerConfig] = Field(
        default_factory=list,
    )


DEFAULT_BOTTLENECK_CONFIG: BottleneckConfig = BottleneckConfig(
    channels=256,
    layers=[
        SelfAttentionConfig(attention_channels=256),
    ],
)


def build_bottleneck(
    input_height: int,
    in_channels: int,
    config: Optional[BottleneckConfig] = None,
) -> nn.Module:
    """Factory function to build the Bottleneck module from configuration.

    Constructs either a base `Bottleneck` or a `BottleneckAttn` instance based
    on the `config.self_attention` flag.

    Parameters
    ----------
    input_height : int
        Height (frequency bins) of the input tensor. Must be positive.
    in_channels : int
        Number of channels in the input tensor. Must be positive.
    config : BottleneckConfig, optional
        Configuration object specifying the bottleneck channels and whether
        to use self-attention. Uses `DEFAULT_BOTTLENECK_CONFIG` if None.

    Returns
    -------
    nn.Module
        An initialized bottleneck module (`Bottleneck` or `BottleneckAttn`).

    Raises
    ------
    ValueError
        If `input_height` or `in_channels` are not positive.
    """
    config = config or DEFAULT_BOTTLENECK_CONFIG

    current_channels = in_channels
    current_height = input_height

    layers = []

    for layer_config in config.layers:
        layer, current_channels, current_height = build_layer_from_config(
            input_height=current_height,
            in_channels=current_channels,
            config=layer_config,
        )
        assert current_height == input_height, (
            "Bottleneck layers should not change the spectrogram height"
        )
        layers.append(layer)

    return Bottleneck(
        input_height=input_height,
        in_channels=in_channels,
        out_channels=config.channels,
        layers=layers,
    )
