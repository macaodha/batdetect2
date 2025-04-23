"""Constructs the Encoder part of a configurable neural network backbone.

This module defines the configuration structure (`EncoderConfig`) and provides
the `Encoder` class (an `nn.Module`) along with a factory function
(`build_encoder`) to create sequential encoders. Encoders typically form the
downsampling path in architectures like U-Nets, processing input feature maps
(like spectrograms) to produce lower-resolution, higher-dimensionality feature
representations (bottleneck features).

The encoder is built dynamically by stacking neural network blocks based on a
list of configuration objects provided in `EncoderConfig.layers`. Each
configuration object specifies the type of block (e.g., standard convolution,
coordinate-feature convolution with downsampling) and its parameters
(e.g., output channels). This allows for flexible definition of encoder
architectures via configuration files.

The `Encoder`'s `forward` method returns outputs from all intermediate layers,
suitable for skip connections, while the `encode` method returns only the final
bottleneck output. A default configuration (`DEFAULT_ENCODER_CONFIG`) is also
provided.
"""

from typing import Annotated, List, Optional, Union

import torch
from pydantic import Field
from torch import nn

from batdetect2.configs import BaseConfig
from batdetect2.models.blocks import (
    BlockGroupConfig,
    ConvConfig,
    FreqCoordConvDownConfig,
    StandardConvDownConfig,
    build_layer_from_config,
)

__all__ = [
    "EncoderConfig",
    "Encoder",
    "build_encoder",
    "DEFAULT_ENCODER_CONFIG",
]

EncoderLayerConfig = Annotated[
    Union[
        ConvConfig,
        FreqCoordConvDownConfig,
        StandardConvDownConfig,
        BlockGroupConfig,
    ],
    Field(discriminator="block_type"),
]
"""Type alias for the discriminated union of block configs usable in Encoder."""


class EncoderConfig(BaseConfig):
    """Configuration for building the sequential Encoder module.

    Defines the sequence of neural network blocks that constitute the encoder
    (downsampling path).

    Attributes
    ----------
    layers : List[EncoderLayerConfig]
        An ordered list of configuration objects, each defining one layer or
        block in the encoder sequence. Each item must be a valid block config
        (e.g., `ConvConfig`, `FreqCoordConvDownConfig`,
        `StandardConvDownConfig`) including a `block_type` field and necessary
        parameters like `out_channels`. Input channels for each layer are
        inferred sequentially. The list must contain at least one layer.
    """

    layers: List[EncoderLayerConfig] = Field(min_length=1)


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
    in_channels : int
        Number of channels expected in the input tensor.
    input_height : int
        Height (frequency bins) expected in the input tensor.
    output_channels : int
        Number of channels in the final output tensor (bottleneck).
    output_height : int
        Height (frequency bins) expected in the output tensor.
    layers : nn.ModuleList
        The sequence of instantiated downscaling layer modules.
    depth : int
        The number of downscaling layers in the encoder.
    """

    def __init__(
        self,
        output_channels: int,
        output_height: int,
        layers: List[nn.Module],
        input_height: int = 128,
        in_channels: int = 1,
    ):
        """Initialize the Encoder module.

        Note: This constructor is typically called internally by the
        `build_encoder` factory function, which prepares the `layers` list.

        Parameters
        ----------
        output_channels : int
            Number of channels produced by the final layer.
        output_height : int
            The expected height of the output tensor.
        layers : List[nn.Module]
            A list of pre-instantiated downscaling layer modules (e.g.,
            `StandardConvDownBlock` or `FreqCoordConvDownBlock`) in the desired
            sequence.
        input_height : int, default=128
            Expected height of the input tensor.
        in_channels : int, default=1
            Expected number of channels in the input tensor.
        """
        super().__init__()

        self.in_channels = in_channels
        self.input_height = input_height
        self.out_channels = output_channels
        self.output_height = output_height

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
            `self.in_channels` and `H_in` must match `self.input_height`.

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
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Input tensor has {x.shape[1]} channels, "
                f"but encoder expects {self.in_channels}."
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
            `in_channels` and `input_height`.

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
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Input tensor has {x.shape[1]} channels, "
                f"but encoder expects {self.in_channels}."
            )

        if x.shape[2] != self.input_height:
            raise ValueError(
                f"Input tensor height {x.shape[2]} does not match "
                f"encoder expected input_height {self.input_height}."
            )

        for layer in self.layers:
            x = layer(x)

        return x


DEFAULT_ENCODER_CONFIG: EncoderConfig = EncoderConfig(
    layers=[
        FreqCoordConvDownConfig(out_channels=32),
        FreqCoordConvDownConfig(out_channels=64),
        BlockGroupConfig(
            blocks=[
                FreqCoordConvDownConfig(out_channels=128),
                ConvConfig(out_channels=256),
            ]
        ),
    ],
)
"""Default configuration for the Encoder.

Specifies an architecture typically used in BatDetect2:
- Input: 1 channel, 128 frequency bins.
- Layer 1: FreqCoordConvDown -> 32 channels, H=64
- Layer 2: FreqCoordConvDown -> 64 channels, H=32
- Layer 3: FreqCoordConvDown -> 128 channels, H=16
- Layer 4: ConvBlock -> 256 channels, H=16 (Bottleneck)
"""


def build_encoder(
    in_channels: int,
    input_height: int,
    config: Optional[EncoderConfig] = None,
) -> Encoder:
    """Factory function to build an Encoder instance from configuration.

    Constructs a sequential `Encoder` module based on the layer sequence
    defined in an `EncoderConfig` object and the provided input dimensions.
    If no config is provided, uses the default layer sequence from
    `DEFAULT_ENCODER_CONFIG`.

    It iteratively builds the layers using the unified
    `build_layer_from_config` factory (from `.blocks`), tracking the changing
    number of channels and feature map height required for each subsequent
    layer, especially for coordinate- aware blocks.

    Parameters
    ----------
    in_channels : int
        The number of channels expected in the input tensor to the encoder.
        Must be > 0.
    input_height : int
        The height (frequency bins) expected in the input tensor. Must be > 0.
        Crucial for initializing coordinate-aware layers correctly.
    config : EncoderConfig, optional
        The configuration object detailing the sequence of layers and their
        parameters. If None, `DEFAULT_ENCODER_CONFIG` is used.

    Returns
    -------
    Encoder
        An initialized `Encoder` module.

    Raises
    ------
    ValueError
        If `in_channels` or `input_height` are not positive, or if the layer
        configuration is invalid (e.g., empty list, unknown `block_type`).
    NotImplementedError
        If `build_layer_from_config` encounters an unknown `block_type`.
    """
    if in_channels <= 0 or input_height <= 0:
        raise ValueError("in_channels and input_height must be positive.")

    config = config or DEFAULT_ENCODER_CONFIG

    current_channels = in_channels
    current_height = input_height

    layers = []

    for layer_config in config.layers:
        layer, current_channels, current_height = build_layer_from_config(
            in_channels=current_channels,
            input_height=current_height,
            config=layer_config,
        )
        layers.append(layer)

    return Encoder(
        input_height=input_height,
        layers=layers,
        in_channels=in_channels,
        output_channels=current_channels,
        output_height=current_height,
    )
