"""Constructs the Encoder part of an Encoder-Decoder neural network.

This module defines the configuration structure (`EncoderConfig`) and provides
the `Encoder` class (an `nn.Module`) along with a factory function
(`build_encoder`) to create sequential encoders commonly used as the
downsampling path in architectures like U-Nets for spectrogram analysis.

The encoder is built by stacking configurable downscaling blocks. Two types
of downscaling blocks are supported, selectable via the configuration:
- `StandardConvDownBlock`: A basic Conv2d -> MaxPool2d -> BN -> ReLU block.
- `FreqCoordConvDownBlock`: A similar block that incorporates frequency
  coordinate information (CoordF) before the convolution to potentially aid
  spatial awareness along the frequency axis.

The `Encoder`'s `forward` method provides access to intermediate feature maps
from each stage, suitable for use as skip connections in a corresponding
Decoder. A separate `encode` method returns only the final output (bottleneck)
features.
"""

from enum import Enum
from typing import List

import torch
from pydantic import Field
from torch import nn

from batdetect2.configs import BaseConfig
from batdetect2.models.blocks import (
    FreqCoordConvDownBlock,
    StandardConvDownBlock,
)

__all__ = [
    "DownscalingLayer",
    "EncoderLayer",
    "EncoderConfig",
    "Encoder",
    "build_encoder",
]


class DownscalingLayer(str, Enum):
    """Enumeration of available downscaling layer types for the Encoder.

    Used in configuration to specify which block implementation to use at each
    stage of the encoder.

    Attributes
    ----------
    standard : str
        Identifier for the `StandardConvDownBlock`.
    coord : str
        Identifier for the `FreqCoordConvDownBlock` (incorporates frequency
        coords).
    """

    standard = "ConvBlockDownStandard"
    coord = "FreqCoordConvDownBlock"


class EncoderLayer(BaseConfig):
    """Configuration for a single layer within the Encoder sequence.

    Attributes
    ----------
    layer_type : DownscalingLayer
        Specifies the type of downscaling block to use for this layer
        (either 'standard' or 'coord').
    channels : int
        The number of output channels this layer should produce. Must be > 0.
    """

    layer_type: DownscalingLayer
    channels: int


class EncoderConfig(BaseConfig):
    """Configuration for building the entire sequential Encoder.

    Attributes
    ----------
    input_height : int
        The expected height (number of frequency bins) of the input spectrogram
        tensor fed into the first layer of the encoder. Required for
        calculating intermediate heights, especially for CoordF layers. Must be
        > 0.
    layers : List[EncoderLayer]
        An ordered list defining the sequence of downscaling layers in the
        encoder. Each item specifies the layer type and its output channel
        count. The number of input channels for each layer is inferred from the
        previous layer's output channels (or `input_channels` for the first
        layer). Must contain at least one layer definition.
    input_channels : int, default=1
        The number of channels in the initial input tensor to the encoder
        (e.g., 1 for a standard single-channel spectrogram). Must be > 0.
    """

    input_height: int = Field(gt=0)
    layers: List[EncoderLayer] = Field(min_length=1)
    input_channels: int = Field(gt=0)


def build_downscaling_layer(
    in_channels: int,
    out_channels: int,
    input_height: int,
    layer_type: DownscalingLayer,
) -> tuple[nn.Module, int, int]:
    """Build a single downscaling layer based on configuration.

    Internal factory function used by `build_encoder`. Instantiates the
    appropriate downscaling block (`StandardConvDownBlock` or
    `FreqCoordConvDownBlock`) and returns it along with its expected output
    channel count and output height (assuming 2x spatial downsampling).

    Parameters
    ----------
    in_channels : int
        Number of input channels to the layer.
    out_channels : int
        Desired number of output channels from the layer.
    input_height : int
        Height of the input feature map to this layer.
    layer_type : DownscalingLayer
        The type of layer to build ('standard' or 'coord').

    Returns
    -------
    Tuple[nn.Module, int, int]
        A tuple containing:
        - The instantiated `nn.Module` layer.
        - The number of output channels (`out_channels`).
        - The expected output height (`input_height // 2`).

    Raises
    ------
    ValueError
        If `layer_type` is invalid.
    """
    if layer_type == DownscalingLayer.standard:
        return (
            StandardConvDownBlock(
                in_channels=in_channels,
                out_channels=out_channels,
            ),
            out_channels,
            input_height // 2,
        )

    if layer_type == DownscalingLayer.coord:
        return (
            FreqCoordConvDownBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                input_height=input_height,
            ),
            out_channels,
            input_height // 2,
        )

    raise ValueError(
        f"Invalid downscaling layer type {layer_type}. "
        f"Valid values: ConvBlockDownCoordF, ConvBlockDownStandard"
    )


class Encoder(nn.Module):
    """Sequential Encoder module composed of configurable downscaling layers.

    Constructs the downsampling path of an encoder-decoder network by stacking
    multiple downscaling blocks.

    The `forward` method executes the sequence and returns the output feature
    map from *each* downscaling stage, facilitating the implementation of skip
    connections in U-Net-like architectures. The `encode` method returns only
    the final output tensor (bottleneck features).

    Attributes
    ----------
    input_channels : int
        Number of channels expected in the input tensor.
    input_height : int
        Height (frequency bins) expected in the input tensor.
    output_channels : int
        Number of channels in the final output tensor (bottleneck).
    layers : nn.ModuleList
        The sequence of instantiated downscaling layer modules.
    depth : int
        The number of downscaling layers in the encoder.
    """

    def __init__(
        self,
        output_channels: int,
        layers: List[nn.Module],
        input_height: int = 128,
        input_channels: int = 1,
    ):
        """Initialize the Encoder module.

        Note: This constructor is typically called internally by the
        `build_encoder` factory function, which prepares the `layers` list.

        Parameters
        ----------
        output_channels : int
            Number of channels produced by the final layer.
        layers : List[nn.Module]
            A list of pre-instantiated downscaling layer modules (e.g.,
            `StandardConvDownBlock` or `FreqCoordConvDownBlock`) in the desired
            sequence.
        input_height : int, default=128
            Expected height of the input tensor.
        input_channels : int, default=1
            Expected number of channels in the input tensor.
        """
        super().__init__()

        self.input_channels = input_channels
        self.input_height = input_height
        self.output_channels = output_channels

        self.layers = nn.ModuleList(layers)
        self.depth = len(self.layers)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Pass input through encoder layers, returns all intermediate outputs.

        This method is typically used when the Encoder is part of a U-Net or
        similar architecture requiring skip connections.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape `(B, C_in, H_in, W)`, where `C_in` must match
            `self.input_channels` and `H_in` must match `self.input_height`.

        Returns
        -------
        List[torch.Tensor]
            A list containing the output tensors from *each* downscaling layer
            in the sequence. `outputs[0]` is the output of the first layer,
            `outputs[-1]` is the final output (bottleneck) of the encoder.

        Raises
        ------
        ValueError
            If input tensor channel count or height does not match expected
            values.
        """
        if x.shape[1] != self.input_channels:
            raise ValueError(
                f"Input tensor has {x.shape[1]} channels, "
                f"but encoder expects {self.input_channels}."
            )

        if x.shape[2] != self.input_height:
            raise ValueError(
                f"Input tensor height {x.shape[2]} does not match "
                f"encoder expected input_height {self.input_height}."
            )

        outputs = []

        for layer in self.layers:
            x = layer(x)
            outputs.append(x)

        return outputs

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Pass input through encoder layers, returning only the final output.

        This method provides access to the bottleneck features produced after
        the last downscaling layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape `(B, C_in, H_in, W)`. Must match expected
            `input_channels` and `input_height`.

        Returns
        -------
        torch.Tensor
            The final output tensor (bottleneck features) from the last layer
            of the encoder. Shape `(B, C_out, H_out, W_out)`.

        Raises
        ------
        ValueError
            If input tensor channel count or height does not match expected
            values.
        """
        if x.shape[1] != self.input_channels:
            raise ValueError(
                f"Input tensor has {x.shape[1]} channels, "
                f"but encoder expects {self.input_channels}."
            )

        if x.shape[2] != self.input_height:
            raise ValueError(
                f"Input tensor height {x.shape[2]} does not match "
                f"encoder expected input_height {self.input_height}."
            )

        for layer in self.layers:
            x = layer(x)

        return x


def build_encoder(config: EncoderConfig) -> Encoder:
    """Factory function to build an Encoder instance from configuration.

    Constructs a sequential `Encoder` module based on the specifications in
    an `EncoderConfig` object. It iteratively builds the specified sequence
    of downscaling layers (`StandardConvDownBlock` or `FreqCoordConvDownBlock`),
    tracking the changing number of channels and feature map height.

    Parameters
    ----------
    config : EncoderConfig
        The configuration object detailing the encoder architecture, including
        input dimensions, layer types, and channel counts for each stage.

    Returns
    -------
    Encoder
        An initialized `Encoder` module.

    Raises
    ------
    ValueError
        If the layer configuration is invalid (e.g., unknown layer type).
    """
    current_channels = config.input_channels
    current_height = config.input_height

    layers = []

    for layer_config in config.layers:
        layer, current_channels, current_height = build_downscaling_layer(
            in_channels=current_channels,
            out_channels=layer_config.channels,
            input_height=current_height,
            layer_type=layer_config.layer_type,
        )
        layers.append(layer)

    return Encoder(
        input_height=config.input_height,
        layers=layers,
        input_channels=config.input_channels,
        output_channels=current_channels,
    )
