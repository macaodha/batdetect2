"""Constructs the Decoder part of an Encoder-Decoder neural network.

This module defines the configuration structure (`DecoderConfig`) for the layer
sequence and provides the `Decoder` class (an `nn.Module`) along with a factory
function (`build_decoder`). Decoders typically form the upsampling path in
architectures like U-Nets, taking bottleneck features
(usually from an `Encoder`) and skip connections to reconstruct
higher-resolution feature maps.

The decoder is built dynamically by stacking neural network blocks based on a
list of configuration objects provided in `DecoderConfig.layers`. Each config
object specifies the type of block (e.g., standard convolution,
coordinate-feature convolution with upsampling) and its parameters. This allows
flexible definition of decoder architectures via configuration files.

The `Decoder`'s `forward` method is designed to accept skip connection tensors
(`residuals`) from the encoder, merging them with the upsampled feature maps
at each stage.
"""

from typing import Annotated, List, Optional, Union

import torch
from pydantic import Field
from torch import nn

from batdetect2.configs import BaseConfig
from batdetect2.models.blocks import (
    BlockGroupConfig,
    ConvConfig,
    FreqCoordConvUpConfig,
    StandardConvUpConfig,
    build_layer_from_config,
)

__all__ = [
    "DecoderConfig",
    "Decoder",
    "build_decoder",
    "DEFAULT_DECODER_CONFIG",
]

DecoderLayerConfig = Annotated[
    Union[
        ConvConfig,
        FreqCoordConvUpConfig,
        StandardConvUpConfig,
        BlockGroupConfig,
    ],
    Field(discriminator="block_type"),
]
"""Type alias for the discriminated union of block configs usable in Decoder."""


class DecoderConfig(BaseConfig):
    """Configuration for the sequence of layers in the Decoder module.

    Defines the types and parameters of the neural network blocks that
    constitute the decoder's upsampling path.

    Attributes
    ----------
    layers : List[DecoderLayerConfig]
        An ordered list of configuration objects, each defining one layer or
        block in the decoder sequence. Each item must be a valid block
        config including a `block_type` field and necessary parameters like
        `out_channels`. Input channels for each layer are inferred sequentially.
        The list must contain at least one layer.
    """

    layers: List[DecoderLayerConfig] = Field(min_length=1)


class Decoder(nn.Module):
    """Sequential Decoder module composed of configurable upsampling layers.

    Constructs the upsampling path of an encoder-decoder network by stacking
    multiple blocks (e.g., `StandardConvUpBlock`, `FreqCoordConvUpBlock`)
    based on a list of layer modules provided during initialization (typically
    created by the `build_decoder` factory function).

    The `forward` method is designed to integrate skip connection tensors
    (`residuals`) from the corresponding encoder stages, by adding them
    element-wise to the input of each decoder layer before processing.

    Attributes
    ----------
    in_channels : int
        Number of channels expected in the input tensor.
    out_channels : int
        Number of channels in the final output tensor produced by the last
        layer.
    input_height : int
        Height (frequency bins) expected in the input tensor.
    output_height : int
        Height (frequency bins) expected in the output tensor.
    layers : nn.ModuleList
        The sequence of instantiated upscaling layer modules.
    depth : int
        The number of upscaling layers (depth) in the decoder.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        input_height: int,
        output_height: int,
        layers: List[nn.Module],
    ):
        """Initialize the Decoder module.

        Note: This constructor is typically called internally by the
        `build_decoder` factory function.

        Parameters
        ----------
        out_channels : int
            Number of channels produced by the final layer.
        input_height : int
            Expected height of the input tensor (bottleneck).
        in_channels : int
            Expected number of channels in the input tensor (bottleneck).
        layers : List[nn.Module]
            A list of pre-instantiated upscaling layer modules (e.g.,
            `StandardConvUpBlock` or `FreqCoordConvUpBlock`) in the desired
            sequence (from bottleneck towards output resolution).
        """
        super().__init__()

        self.input_height = input_height
        self.output_height = output_height

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.layers = nn.ModuleList(layers)
        self.depth = len(self.layers)

    def forward(
        self,
        x: torch.Tensor,
        residuals: List[torch.Tensor],
    ) -> torch.Tensor:
        """Pass input through decoder layers, incorporating skip connections.

        Processes the input tensor `x` sequentially through the upscaling
        layers. At each stage, the corresponding skip connection tensor from
        the `residuals` list is added element-wise to the input before passing
        it to the upscaling block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor from the previous stage (e.g., encoder bottleneck).
            Shape `(B, C_in, H_in, W_in)`, where `C_in` matches
            `self.in_channels`.
        residuals : List[torch.Tensor]
            List containing the skip connection tensors from the corresponding
            encoder stages. Should be ordered from the deepest encoder layer
            output (lowest resolution) to the shallowest (highest resolution
            near input). The number of tensors in this list must match the
            number of decoder layers (`self.depth`). Each residual tensor's
            channel count must be compatible with the input tensor `x` for
            element-wise addition (or concatenation if the blocks were designed
            for it).

        Returns
        -------
        torch.Tensor
            The final decoded feature map tensor produced by the last layer.
            Shape `(B, C_out, H_out, W_out)`.

        Raises
        ------
        ValueError
            If the number of `residuals` provided does not match the decoder
            depth.
        RuntimeError
            If shapes mismatch during skip connection addition or layer
            processing.
        """
        if len(residuals) != len(self.layers):
            raise ValueError(
                f"Incorrect number of residuals provided. "
                f"Expected {len(self.layers)} (matching the number of layers), "
                f"but got {len(residuals)}."
            )

        for layer, res in zip(self.layers, residuals[::-1]):
            x = layer(x + res)

        return x


DEFAULT_DECODER_CONFIG: DecoderConfig = DecoderConfig(
    layers=[
        FreqCoordConvUpConfig(out_channels=64),
        FreqCoordConvUpConfig(out_channels=32),
        BlockGroupConfig(
            blocks=[
                FreqCoordConvUpConfig(out_channels=32),
                ConvConfig(out_channels=32),
            ]
        ),
    ],
)
"""A default configuration for the Decoder's *layer sequence*.

Specifies an architecture often used in BatDetect2, consisting of three
frequency coordinate-aware upsampling blocks followed by a standard
convolutional block.
"""


def build_decoder(
    in_channels: int,
    input_height: int,
    config: Optional[DecoderConfig] = None,
) -> Decoder:
    """Factory function to build a Decoder instance from configuration.

    Constructs a sequential `Decoder` module based on the layer sequence
    defined in a `DecoderConfig` object and the provided input dimensions
    (bottleneck channels and height). If no config is provided, uses the
    default layer sequence from `DEFAULT_DECODER_CONFIG`.

    It iteratively builds the layers using the unified `build_layer_from_config`
    factory (from `.blocks`), tracking the changing number of channels and
    feature map height required for each subsequent layer.

    Parameters
    ----------
    in_channels : int
        The number of channels in the input tensor to the decoder. Must be > 0.
    input_height : int
        The height (frequency bins) of the input tensor to the decoder. Must be
        > 0.
    config : DecoderConfig, optional
        The configuration object detailing the sequence of layers and their
        parameters. If None, `DEFAULT_DECODER_CONFIG` is used.

    Returns
    -------
    Decoder
        An initialized `Decoder` module.

    Raises
    ------
    ValueError
        If `in_channels` or `input_height` are not positive, or if the layer
        configuration is invalid (e.g., empty list, unknown `block_type`).
    NotImplementedError
        If `build_layer_from_config` encounters an unknown `block_type`.
    """
    config = config or DEFAULT_DECODER_CONFIG

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

    return Decoder(
        in_channels=in_channels,
        out_channels=current_channels,
        input_height=input_height,
        output_height=current_height,
        layers=layers,
    )
