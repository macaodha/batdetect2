"""Assembles a complete Encoder-Decoder Backbone network.

This module defines the configuration (`BackboneConfig`) and implementation
(`Backbone`) for a standard encoder-decoder style neural network backbone.

It orchestrates the connection between three main components, built using their
respective configurations and factory functions from sibling modules:
1.  Encoder (`batdetect2.models.encoder`): Downsampling path, extracts features
    at multiple resolutions and provides skip connections.
2.  Bottleneck (`batdetect2.models.bottleneck`): Processes features at the
    lowest resolution, optionally applying self-attention.
3.  Decoder (`batdetect2.models.decoder`): Upsampling path, reconstructs high-
    resolution features using bottleneck features and skip connections.

The resulting `Backbone` module takes a spectrogram as input and outputs a
final feature map, typically used by subsequent prediction heads. It includes
automatic padding to handle input sizes not perfectly divisible by the
network's total downsampling factor.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from soundevent import data
from torch import nn

from batdetect2.configs import BaseConfig, load_config
from batdetect2.models.blocks import ConvBlock
from batdetect2.models.bottleneck import BottleneckConfig, build_bottleneck
from batdetect2.models.decoder import Decoder, DecoderConfig, build_decoder
from batdetect2.models.encoder import Encoder, EncoderConfig, build_encoder
from batdetect2.models.types import BackboneModel

__all__ = [
    "Backbone",
    "BackboneConfig",
    "load_backbone_config",
    "build_backbone",
]


class Backbone(BackboneModel):
    """Encoder-Decoder Backbone Network Implementation.

    Combines an Encoder, Bottleneck, and Decoder module sequentially, using
    skip connections between the Encoder and Decoder. Implements the standard
    U-Net style forward pass. Includes automatic input padding to handle
    various input sizes and a final convolutional block to adjust the output
    channels.

    This class inherits from `BackboneModel` and implements its `forward`
    method. Instances are typically created using the `build_backbone` factory
    function.

    Attributes
    ----------
    input_height : int
        Expected height of the input spectrogram.
    out_channels : int
        Number of channels in the final output feature map.
    encoder : Encoder
        The instantiated encoder module.
    decoder : Decoder
        The instantiated decoder module.
    bottleneck : nn.Module
        The instantiated bottleneck module.
    final_conv : ConvBlock
        Final convolutional block applied after the decoder.
    divide_factor : int
        The total downsampling factor (2^depth) applied by the encoder,
        used for automatic input padding.
    """

    def __init__(
        self,
        input_height: int,
        out_channels: int,
        encoder: Encoder,
        decoder: Decoder,
        bottleneck: nn.Module,
    ):
        """Initialize the Backbone network.

        Parameters
        ----------
        input_height : int
            Expected height of the input spectrogram.
        out_channels : int
            Desired number of output channels for the backbone's feature map.
        encoder : Encoder
            An initialized Encoder module.
        decoder : Decoder
            An initialized Decoder module.
        bottleneck : nn.Module
            An initialized Bottleneck module.

        Raises
        ------
        ValueError
            If component output/input channels or heights are incompatible.
        """
        super().__init__()
        self.input_height = input_height
        self.out_channels = out_channels

        self.encoder = encoder
        self.decoder = decoder
        self.bottleneck = bottleneck

        self.final_conv = ConvBlock(
            in_channels=decoder.out_channels,
            out_channels=out_channels,
        )

        # Down/Up scaling factor. Need to ensure inputs are divisible by
        # this factor in order to be processed by the down/up scaling layers
        # and recover the correct shape
        self.divide_factor = input_height // self.encoder.output_height

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """Perform the forward pass through the encoder-decoder backbone.

        Applies padding, runs encoder, bottleneck, decoder (with skip
        connections), removes padding, and applies a final convolution.

        Parameters
        ----------
        spec : torch.Tensor
            Input spectrogram tensor, shape `(B, C_in, H_in, W_in)`. Must match
            `self.encoder.input_channels` and `self.input_height`.

        Returns
        -------
        torch.Tensor
            Output feature map tensor, shape `(B, C_out, H_in, W_in)`, where
            `C_out` is `self.out_channels`.
        """
        spec, h_pad, w_pad = _pad_adjust(spec, factor=self.divide_factor)

        # encoder
        residuals = self.encoder(spec)

        # bottleneck
        x = self.bottleneck(residuals[-1])

        # decoder
        x = self.decoder(x, residuals=residuals)

        # Restore original size
        x = _restore_pad(x, h_pad=h_pad, w_pad=w_pad)

        return self.final_conv(x)


class BackboneConfig(BaseConfig):
    """Configuration for the Encoder-Decoder Backbone network.

    Aggregates configurations for the encoder, bottleneck, and decoder
    components, along with defining the input and final output dimensions
    for the complete backbone.

    Attributes
    ----------
    input_height : int, default=128
        Expected height (frequency bins) of the input spectrograms to the
        backbone. Must be positive.
    in_channels : int, default=1
        Expected number of channels in the input spectrograms (e.g., 1 for
        mono). Must be positive.
    encoder : EncoderConfig, optional
        Configuration for the encoder. If None or omitted,
        the default encoder configuration (`DEFAULT_ENCODER_CONFIG` from the
        encoder module) will be used.
    bottleneck : BottleneckConfig, optional
        Configuration for the bottleneck layer connecting encoder and decoder.
        If None or omitted, the default bottleneck configuration will be used.
    decoder : DecoderConfig, optional
        Configuration for the decoder. If None or omitted,
        the default decoder configuration (`DEFAULT_DECODER_CONFIG` from the
        decoder module) will be used.
    out_channels : int, default=32
        Desired number of channels in the final feature map output by the
        backbone. Must be positive.
    """

    input_height: int = 128
    in_channels: int = 1
    encoder: Optional[EncoderConfig] = None
    bottleneck: Optional[BottleneckConfig] = None
    decoder: Optional[DecoderConfig] = None
    out_channels: int = 32


def load_backbone_config(
    path: data.PathLike,
    field: Optional[str] = None,
) -> BackboneConfig:
    """Load the backbone configuration from a file.

    Reads a configuration file (YAML) and validates it against the
    `BackboneConfig` schema, potentially extracting data from a nested field.

    Parameters
    ----------
    path : PathLike
        Path to the configuration file.
    field : str, optional
        Dot-separated path to a nested section within the file containing the
        backbone configuration (e.g., "model.backbone"). If None, the entire
        file content is used.

    Returns
    -------
    BackboneConfig
        The loaded and validated backbone configuration object.

    Raises
    ------
    FileNotFoundError
        If the config file path does not exist.
    yaml.YAMLError
        If the file content is not valid YAML.
    pydantic.ValidationError
        If the loaded config data does not conform to `BackboneConfig`.
    KeyError, TypeError
        If `field` specifies an invalid path.
    """
    return load_config(path, schema=BackboneConfig, field=field)


def build_backbone(config: BackboneConfig) -> BackboneModel:
    """Factory function to build a Backbone from configuration.

    Constructs the `Encoder`, `Bottleneck`, and `Decoder` components based on
    the provided `BackboneConfig`, validates their compatibility, and assembles
    them into a `Backbone` instance.

    Parameters
    ----------
    config : BackboneConfig
        The configuration object detailing the backbone architecture, including
        input dimensions and configurations for encoder, bottleneck, and
        decoder.

    Returns
    -------
    BackboneModel
        An initialized `Backbone` module ready for use.

    Raises
    ------
    ValueError
        If sub-component configurations are incompatible
        (e.g., channel mismatches, decoder output height doesn't match backbone
        input height).
    NotImplementedError
        If an unknown block type is specified in sub-configs.
    """
    encoder = build_encoder(
        in_channels=config.in_channels,
        input_height=config.input_height,
        config=config.encoder,
    )

    bottleneck = build_bottleneck(
        input_height=encoder.output_height,
        in_channels=encoder.out_channels,
        config=config.bottleneck,
    )

    decoder = build_decoder(
        in_channels=bottleneck.out_channels,
        input_height=encoder.output_height,
        config=config.decoder,
    )

    if decoder.output_height != config.input_height:
        raise ValueError(
            "Invalid configuration: Decoder output height "
            f"({decoder.output_height}) must match the Backbone input height "
            f"({config.input_height}). Check encoder/decoder layer "
            "configurations and input/bottleneck heights."
        )

    return Backbone(
        input_height=config.input_height,
        out_channels=config.out_channels,
        encoder=encoder,
        decoder=decoder,
        bottleneck=bottleneck,
    )


def _pad_adjust(
    spec: torch.Tensor,
    factor: int = 32,
) -> Tuple[torch.Tensor, int, int]:
    """Pad tensor height and width to be divisible by a factor.

    Calculates the required padding for the last two dimensions (H, W) to make
    them divisible by `factor` and applies right/bottom padding using
    `torch.nn.functional.pad`.

    Parameters
    ----------
    spec : torch.Tensor
        Input tensor, typically shape `(B, C, H, W)`.
    factor : int, default=32
        The factor to make height and width divisible by.

    Returns
    -------
    Tuple[torch.Tensor, int, int]
        A tuple containing:
        - The padded tensor.
        - The amount of padding added to height (`h_pad`).
        - The amount of padding added to width (`w_pad`).
    """
    h, w = spec.shape[2:]
    h_pad = -h % factor
    w_pad = -w % factor

    if h_pad == 0 and w_pad == 0:
        return spec, 0, 0

    return F.pad(spec, (0, w_pad, 0, h_pad)), h_pad, w_pad


def _restore_pad(
    x: torch.Tensor, h_pad: int = 0, w_pad: int = 0
) -> torch.Tensor:
    """Remove padding added by _pad_adjust.

    Removes padding from the bottom and right edges of the tensor.

    Parameters
    ----------
    x : torch.Tensor
        Padded tensor, typically shape `(B, C, H_padded, W_padded)`.
    h_pad : int, default=0
        Amount of padding previously added to the height (bottom).
    w_pad : int, default=0
        Amount of padding previously added to the width (right).

    Returns
    -------
    torch.Tensor
        Tensor with padding removed, shape `(B, C, H_original, W_original)`.
    """
    if h_pad > 0:
        x = x[:, :, :-h_pad, :]

    if w_pad > 0:
        x = x[:, :, :, :-w_pad]

    return x
