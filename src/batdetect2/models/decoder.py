"""Decoder (upsampling path) for the BatDetect2 backbone.

This module defines ``DecoderConfig`` and the ``Decoder`` ``nn.Module``,
together with the ``build_decoder`` factory function.

In a U-Net-style network the decoder progressively restores the spatial
resolution of the feature map back towards the input resolution. At each
stage it combines the upsampled features with the corresponding skip-connection
tensor from the encoder (the residual) by element-wise addition before passing
the result to the upsampling block.

The decoder is fully configurable: the type, number, and parameters of the
upsampling blocks are described by a ``DecoderConfig`` object containing an
ordered list of block configuration objects (see ``batdetect2.models.blocks``
for available block types).

A default configuration ``DEFAULT_DECODER_CONFIG`` is provided and used by
``build_decoder`` when no explicit configuration is supplied.
"""

from typing import Annotated, List

import torch
from pydantic import Field
from torch import nn

from batdetect2.core.configs import BaseConfig
from batdetect2.models.blocks import (
    ConvConfig,
    FreqCoordConvUpConfig,
    LayerGroupConfig,
    StandardConvUpConfig,
    build_layer,
)

__all__ = [
    "DecoderConfig",
    "Decoder",
    "build_decoder",
    "DEFAULT_DECODER_CONFIG",
]

DecoderLayerConfig = Annotated[
    ConvConfig
    | FreqCoordConvUpConfig
    | StandardConvUpConfig
    | LayerGroupConfig,
    Field(discriminator="name"),
]
"""Type alias for the discriminated union of block configs usable in Decoder."""


class DecoderConfig(BaseConfig):
    """Configuration for the sequential ``Decoder`` module.

    Attributes
    ----------
    layers : List[DecoderLayerConfig]
        Ordered list of block configuration objects defining the decoder's
        upsampling stages (from deepest to shallowest). Each entry
        specifies the block type (via its ``name`` field) and any
        block-specific parameters such as ``out_channels``. Input channels
        for each block are inferred automatically from the output of the
        previous block. Must contain at least one entry.
    """

    layers: List[DecoderLayerConfig] = Field(min_length=1)


class Decoder(nn.Module):
    """Sequential decoder module composed of configurable upsampling layers.

    Executes a series of upsampling blocks in order, adding the
    corresponding encoder skip-connection tensor (residual) to the feature
    map before each block. The residuals are consumed in reverse order (from
    deepest encoder layer to shallowest) to match the spatial resolutions at
    each decoder stage.

    Instances are typically created by ``build_decoder``.

    Attributes
    ----------
    in_channels : int
        Number of channels expected in the input tensor (bottleneck output).
    out_channels : int
        Number of channels in the final output feature map.
    input_height : int
        Height (frequency bins) of the input tensor.
    output_height : int
        Height (frequency bins) of the output tensor.
    layers : nn.ModuleList
        Sequence of instantiated upsampling block modules.
    depth : int
        Number of upsampling layers.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        input_height: int,
        output_height: int,
        layers: List[nn.Module],
    ):
        """Initialise the Decoder module.

        This constructor is typically called by the ``build_decoder``
        factory function.

        Parameters
        ----------
        in_channels : int
            Number of channels in the input tensor (bottleneck output).
        out_channels : int
            Number of channels produced by the final layer.
        input_height : int
            Height of the input tensor (bottleneck output height).
        output_height : int
            Height of the output tensor after all layers have been applied.
        layers : List[nn.Module]
            Pre-built upsampling block modules in execution order (deepest
            stage first).
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
        """Pass input through all decoder layers, incorporating skip connections.

        At each stage the corresponding residual tensor is added
        element-wise to ``x`` before it is passed to the upsampling block.
        Residuals are consumed in reverse order — the last element of
        ``residuals`` (the output of the shallowest encoder layer) is added
        at the first decoder stage, and the first element (output of the
        deepest encoder layer) is added at the last decoder stage.

        Parameters
        ----------
        x : torch.Tensor
            Bottleneck feature map, shape ``(B, C_in, H_in, W)``.
        residuals : List[torch.Tensor]
            Skip-connection tensors from the encoder, ordered from shallowest
            (index 0) to deepest (index -1). Must contain exactly
            ``self.depth`` tensors. Each tensor must have the same spatial
            dimensions and channel count as ``x`` at the corresponding
            decoder stage.

        Returns
        -------
        torch.Tensor
            Decoded feature map, shape ``(B, C_out, H_out, W)``.

        Raises
        ------
        ValueError
            If the number of ``residuals`` does not equal ``self.depth``.
        """
        if len(residuals) != len(self.layers):
            raise ValueError(
                f"Incorrect number of residuals provided. "
                f"Expected {len(self.layers)} (matching the number of layers), "
                f"but got {len(residuals)}."
            )

        for layer, res in zip(self.layers, residuals[::-1], strict=False):
            x = layer(x + res)

        return x


DEFAULT_DECODER_CONFIG: DecoderConfig = DecoderConfig(
    layers=[
        FreqCoordConvUpConfig(out_channels=64),
        FreqCoordConvUpConfig(out_channels=32),
        LayerGroupConfig(
            layers=[
                FreqCoordConvUpConfig(out_channels=32),
                ConvConfig(out_channels=32),
            ]
        ),
    ],
)
"""Default decoder configuration used in standard BatDetect2 models.

Mirrors ``DEFAULT_ENCODER_CONFIG`` in reverse. Assumes the bottleneck
output has 256 channels and height 16, and produces:

- Stage 1 (``FreqCoordConvUp``): 64 channels, height 32.
- Stage 2 (``FreqCoordConvUp``): 32 channels, height 64.
- Stage 3 (``LayerGroup``):

  - ``FreqCoordConvUp``: 32 channels, height 128.
  - ``ConvBlock``: 32 channels, height 128 (final feature map).
"""


def build_decoder(
    in_channels: int,
    input_height: int,
    config: DecoderConfig | None = None,
) -> Decoder:
    """Build a ``Decoder`` from configuration.

    Constructs a sequential ``Decoder`` by iterating over the block
    configurations in ``config.layers``, building each block with
    ``build_layer``, and tracking the channel count and feature-map height
    as they change through the sequence.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input tensor (bottleneck output). Must
        be positive.
    input_height : int
        Height (number of frequency bins) of the input tensor. Must be
        positive.
    config : DecoderConfig, optional
        Configuration specifying the layer sequence. Defaults to
        ``DEFAULT_DECODER_CONFIG`` if not provided.

    Returns
    -------
    Decoder
        An initialised ``Decoder`` module.

    Raises
    ------
    ValueError
        If ``in_channels`` or ``input_height`` are not positive.
    KeyError
        If a layer configuration specifies an unknown block type.
    """
    config = config or DEFAULT_DECODER_CONFIG

    current_channels = in_channels
    current_height = input_height

    layers = []

    for layer_config in config.layers:
        layer = build_layer(
            in_channels=current_channels,
            input_height=current_height,
            config=layer_config,
        )
        current_height = layer.get_output_height(current_height)
        current_channels = layer.out_channels
        layers.append(layer)

    return Decoder(
        in_channels=in_channels,
        out_channels=current_channels,
        input_height=input_height,
        output_height=current_height,
        layers=layers,
    )
