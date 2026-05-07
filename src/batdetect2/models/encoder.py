"""Encoder (downsampling path) for the BatDetect2 backbone.

This module defines ``EncoderConfig`` and the ``Encoder`` ``nn.Module``,
together with the ``build_encoder`` factory function.

In a U-Net-style network the encoder progressively reduces the spatial
resolution of the spectrogram whilst increasing the number of feature
channels. Each layer in the encoder produces a feature map that is stored
for use as a skip connection in the corresponding decoder layer.

The encoder is fully configurable: the type, number, and parameters of the
downsampling blocks are described by an ``EncoderConfig`` object containing
an ordered list of block configuration objects (see ``batdetect2.models.blocks``
for available block types).

``Encoder.forward`` returns the outputs of *all* encoder layers as a list,
so that skip connections are available to the decoder.
``Encoder.encode`` returns only the final output (the input to the bottleneck).

A default configuration ``DEFAULT_ENCODER_CONFIG`` is provided and used by
``build_encoder`` when no explicit configuration is supplied.
"""

from typing import Annotated, List

import torch
from pydantic import Field
from torch import nn

from batdetect2.core.configs import BaseConfig
from batdetect2.models.blocks import (
    ConvConfig,
    FreqCoordConvDownConfig,
    LayerGroupConfig,
    StandardConvDownConfig,
    build_layer,
)

__all__ = [
    "EncoderConfig",
    "Encoder",
    "build_encoder",
    "DEFAULT_ENCODER_CONFIG",
]

EncoderLayerConfig = Annotated[
    ConvConfig
    | FreqCoordConvDownConfig
    | StandardConvDownConfig
    | LayerGroupConfig,
    Field(discriminator="name"),
]
"""Type alias for the discriminated union of block configs usable in Encoder."""


class EncoderConfig(BaseConfig):
    """Configuration for the sequential ``Encoder`` module.

    Attributes
    ----------
    layers : List[EncoderLayerConfig]
        Ordered list of block configuration objects defining the encoder's
        downsampling stages. Each entry specifies the block type (via its
        ``name`` field) and any block-specific parameters such as
        ``out_channels``. Input channels for each block are inferred
        automatically from the output of the previous block. Must contain
        at least one entry.
    """

    layers: List[EncoderLayerConfig] = Field(min_length=1)


class Encoder(nn.Module):
    """Sequential encoder module composed of configurable downsampling layers.

    Executes a series of downsampling blocks in order, storing the output of
    each block so that it can be passed as a skip connection to the
    corresponding decoder layer.

    ``forward`` returns the outputs of *all* layers (useful when skip
    connections are needed). ``encode`` returns only the final output
    (the input to the bottleneck).

    Attributes
    ----------
    in_channels : int
        Number of channels expected in the input tensor.
    input_height : int
        Height (frequency bins) expected in the input tensor.
    out_channels : int
        Number of channels in the final output tensor (bottleneck input).
    output_height : int
        Height (frequency bins) of the final output tensor.
    layers : nn.ModuleList
        Sequence of instantiated downsampling block modules.
    depth : int
        Number of downsampling layers.
    """

    def __init__(
        self,
        output_channels: int,
        output_height: int,
        layers: List[nn.Module],
        input_height: int = 128,
        in_channels: int = 1,
    ):
        """Initialise the Encoder module.

        This constructor is typically called by the ``build_encoder`` factory
        function, which takes care of building the ``layers`` list from a
        configuration object.

        Parameters
        ----------
        output_channels : int
            Number of channels produced by the final layer.
        output_height : int
            Height of the output tensor after all layers have been applied.
        layers : List[nn.Module]
            Pre-built downsampling block modules in execution order.
        input_height : int, default=128
            Expected height of the input tensor (frequency bins).
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
        """Pass input through all encoder layers and return every output.

        Used when skip connections are needed (e.g. in a U-Net decoder).

        Parameters
        ----------
        x : torch.Tensor
            Input spectrogram feature map, shape ``(B, C_in, H_in, W)``.
            ``C_in`` must match ``self.in_channels`` and ``H_in`` must
            match ``self.input_height``.

        Returns
        -------
        List[torch.Tensor]
            Output tensors from every layer in order.
            ``outputs[0]`` is the output of the first (shallowest) layer;
            ``outputs[-1]`` is the output of the last (deepest) layer,
            which serves as the input to the bottleneck.

        Raises
        ------
        ValueError
            If the input channel count or height does not match the
            expected values.
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
        """Pass input through all encoder layers and return only the final output.

        Use this when skip connections are not needed and you only require
        the bottleneck feature map.

        Parameters
        ----------
        x : torch.Tensor
            Input spectrogram feature map, shape ``(B, C_in, H_in, W)``.
            Must satisfy the same shape requirements as ``forward``.

        Returns
        -------
        torch.Tensor
            Output of the last encoder layer, shape
            ``(B, C_out, H_out, W)``, where ``C_out`` is
            ``self.out_channels`` and ``H_out`` is ``self.output_height``.

        Raises
        ------
        ValueError
            If the input channel count or height does not match the
            expected values.
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
        LayerGroupConfig(
            layers=[
                FreqCoordConvDownConfig(out_channels=128),
                ConvConfig(out_channels=256),
            ]
        ),
    ],
)
"""Default encoder configuration used in standard BatDetect2 models.

Assumes a 1-channel input with 128 frequency bins and produces the
following feature maps:

- Stage 1 (``FreqCoordConvDown``): 32 channels, height 64.
- Stage 2 (``FreqCoordConvDown``): 64 channels, height 32.
- Stage 3 (``LayerGroup``):

  - ``FreqCoordConvDown``: 128 channels, height 16.
  - ``ConvBlock``: 256 channels, height 16 (bottleneck input).
"""


def build_encoder(
    in_channels: int,
    input_height: int,
    config: EncoderConfig | None = None,
) -> Encoder:
    """Build an ``Encoder`` from configuration.

    Constructs a sequential ``Encoder`` by iterating over the block
    configurations in ``config.layers``, building each block with
    ``build_layer``, and tracking the channel count and feature-map height
    as they change through the sequence.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input spectrogram tensor. Must be
        positive.
    input_height : int
        Height (number of frequency bins) of the input spectrogram.
        Must be positive and should be divisible by
        ``2 ** (number of downsampling stages)`` to avoid size mismatches
        later in the network.
    config : EncoderConfig, optional
        Configuration specifying the layer sequence. Defaults to
        ``DEFAULT_ENCODER_CONFIG`` if not provided.

    Returns
    -------
    Encoder
        An initialised ``Encoder`` module.

    Raises
    ------
    ValueError
        If ``in_channels`` or ``input_height`` are not positive.
    KeyError
        If a layer configuration specifies an unknown block type.
    """
    if in_channels <= 0 or input_height <= 0:
        raise ValueError("in_channels and input_height must be positive.")

    config = config or DEFAULT_ENCODER_CONFIG

    current_channels = in_channels
    current_height = input_height

    layers = []

    for layer_config in config.layers:
        layer = build_layer(
            in_channels=current_channels,
            input_height=current_height,
            config=layer_config,
        )
        layers.append(layer)
        current_height = layer.get_output_height(current_height)
        current_channels = layer.out_channels

    return Encoder(
        input_height=input_height,
        layers=layers,
        in_channels=in_channels,
        output_channels=current_channels,
        output_height=current_height,
    )
