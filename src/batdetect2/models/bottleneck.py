"""Bottleneck component for encoder-decoder network architectures.

The bottleneck sits between the encoder (downsampling path) and the decoder
(upsampling path) and processes the lowest-resolution, highest-channel feature
map produced by the encoder.

This module provides:

- ``BottleneckConfig`` – configuration dataclass describing the number of
  internal channels and an optional sequence of additional layers (currently
  only ``SelfAttention`` is supported).
- ``Bottleneck`` – the ``torch.nn.Module`` implementation. It first applies a
  ``VerticalConv`` to collapse the frequency axis to a single bin, optionally
  runs one or more additional layers (e.g. self-attention along the time axis),
  then repeats the output along the height dimension to restore the original
  frequency resolution before passing features to the decoder.
- ``build_bottleneck`` – factory function that constructs a ``Bottleneck``
  instance from a ``BottleneckConfig`` and the encoder's output dimensions.
"""

from typing import Annotated, List

import torch
from pydantic import Field
from torch import nn

from batdetect2.core.configs import BaseConfig
from batdetect2.models.blocks import (
    Block,
    SelfAttentionConfig,
    VerticalConv,
    build_layer,
)
from batdetect2.typing.models import BottleneckProtocol

__all__ = [
    "BottleneckConfig",
    "Bottleneck",
    "build_bottleneck",
]


class Bottleneck(Block):
    """Bottleneck module for encoder-decoder architectures.

    Processes the lowest-resolution feature map that links the encoder and
    decoder. The sequence of operations is:

    1. ``VerticalConv`` – collapses the frequency axis (height) to a single
       bin by applying a convolution whose kernel spans the full height.
    2. Optional additional layers (e.g. ``SelfAttention``) – applied while
       the feature map has height 1, so they operate purely along the time
       axis.
    3. Height restoration – the single-bin output is repeated along the
       height axis to restore the original frequency resolution, producing
       a tensor that the decoder can accept.

    Parameters
    ----------
    input_height : int
        Height (number of frequency bins) of the input tensor. Must be
        positive.
    in_channels : int
        Number of channels in the input tensor from the encoder. Must be
        positive.
    out_channels : int
        Number of output channels after the bottleneck. Must be positive.
    bottleneck_channels : int, optional
        Number of internal channels used by the ``VerticalConv`` layer.
        Defaults to ``out_channels`` if not provided.
    layers : List[torch.nn.Module], optional
        Additional modules (e.g. ``SelfAttention``) to apply after the
        ``VerticalConv`` and before height restoration.

    Attributes
    ----------
    in_channels : int
        Number of input channels accepted by the bottleneck.
    out_channels : int
        Number of output channels produced by the bottleneck.
    input_height : int
        Expected height of the input tensor.
    bottleneck_channels : int
        Number of channels used internally by the vertical convolution.
    conv_vert : VerticalConv
        The vertical convolution layer.
    layers : nn.ModuleList
        Additional layers applied after the vertical convolution.
    """

    def __init__(
        self,
        input_height: int,
        in_channels: int,
        out_channels: int,
        bottleneck_channels: int | None = None,
        layers: List[torch.nn.Module] | None = None,
    ) -> None:
        """Initialise the Bottleneck layer.

        Parameters
        ----------
        input_height : int
            Height (number of frequency bins) of the input tensor.
        in_channels : int
            Number of channels in the input tensor.
        out_channels : int
            Number of channels in the output tensor.
        bottleneck_channels : int, optional
            Number of internal channels for the ``VerticalConv``. Defaults
            to ``out_channels``.
        layers : List[torch.nn.Module], optional
            Additional modules applied after the ``VerticalConv``, such as
            a ``SelfAttention`` block.
        """
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
        """Process the encoder's bottleneck features.

        Applies vertical convolution, optional additional layers, then
        restores the height dimension by repetition.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor from the encoder, shape
            ``(B, C_in, H_in, W)``. ``C_in`` must match
            ``self.in_channels`` and ``H_in`` must match
            ``self.input_height``.

        Returns
        -------
        torch.Tensor
            Output tensor with shape ``(B, C_out, H_in, W)``. The height
            ``H_in`` is restored by repeating the single-bin result.
        """
        x = self.conv_vert(x)

        for layer in self.layers:
            x = layer(x)

        return x.repeat([1, 1, self.input_height, 1])


BottleneckLayerConfig = Annotated[
    SelfAttentionConfig,
    Field(discriminator="name"),
]
"""Type alias for the discriminated union of block configs usable in the Bottleneck."""


class BottleneckConfig(BaseConfig):
    """Configuration for the bottleneck component.

    Attributes
    ----------
    channels : int
        Number of output channels produced by the bottleneck. This value
        is also used as the dimensionality of any optional layers (e.g.
        self-attention). Must be positive.
    layers : List[BottleneckLayerConfig]
        Ordered list of additional block configurations to apply after the
        initial ``VerticalConv``. Currently only ``SelfAttentionConfig`` is
        supported. Defaults to an empty list (no extra layers).
    """

    channels: int
    layers: List[BottleneckLayerConfig] = Field(default_factory=list)


DEFAULT_BOTTLENECK_CONFIG: BottleneckConfig = BottleneckConfig(
    channels=256,
    layers=[
        SelfAttentionConfig(attention_channels=256),
    ],
)


def build_bottleneck(
    input_height: int,
    in_channels: int,
    config: BottleneckConfig | None = None,
) -> BottleneckProtocol:
    """Build a ``Bottleneck`` module from configuration.

    Constructs a ``Bottleneck`` instance whose internal channel count and
    optional extra layers (e.g. self-attention) are controlled by
    ``config``. If no configuration is provided, the default
    ``DEFAULT_BOTTLENECK_CONFIG`` is used, which includes a
    ``SelfAttention`` layer.

    Parameters
    ----------
    input_height : int
        Height (number of frequency bins) of the input tensor from the
        encoder. Must be positive.
    in_channels : int
        Number of channels in the input tensor from the encoder. Must be
        positive.
    config : BottleneckConfig, optional
        Configuration specifying the output channel count and any
        additional layers. Uses ``DEFAULT_BOTTLENECK_CONFIG`` if ``None``.

    Returns
    -------
    BottleneckProtocol
        An initialised ``Bottleneck`` module.

    Raises
    ------
    AssertionError
        If any configured layer changes the height of the feature map
        (bottleneck layers must preserve height so that it can be restored
        by repetition).
    """
    config = config or DEFAULT_BOTTLENECK_CONFIG

    current_channels = in_channels
    current_height = input_height

    layers = []

    for layer_config in config.layers:
        layer = build_layer(
            input_height=current_height,
            in_channels=current_channels,
            config=layer_config,
        )
        current_height = layer.get_output_height(current_height)
        current_channels = layer.out_channels
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
