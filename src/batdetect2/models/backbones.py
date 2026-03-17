"""Assembles a complete encoder-decoder backbone network.

This module defines ``UNetBackboneConfig`` and the ``UNetBackbone``
``nn.Module``, together with the ``build_backbone`` and
``load_backbone_config`` helpers.

A backbone combines three components built from the sibling modules:

1. **Encoder** (``batdetect2.models.encoder``) – reduces spatial resolution
   while extracting hierarchical features and storing skip-connection tensors.
2. **Bottleneck** (``batdetect2.models.bottleneck``) – processes the
   lowest-resolution features, optionally applying self-attention.
3. **Decoder** (``batdetect2.models.decoder``) – restores spatial resolution
   using bottleneck features and skip connections from the encoder.

The resulting ``UNetBackbone`` takes a spectrogram tensor as input and returns
a high-resolution feature map consumed by the prediction heads in
``batdetect2.models.detectors``.

Input padding is handled automatically: the backbone pads the input to be
divisible by the total downsampling factor and strips the padding from the
output so that the output spatial dimensions always match the input spatial
dimensions.
"""

from typing import Annotated, Literal

import torch
import torch.nn.functional as F
from pydantic import Field, TypeAdapter
from soundevent import data

from batdetect2.core.configs import BaseConfig, load_config
from batdetect2.core.registries import (
    ImportConfig,
    Registry,
    add_import_config,
)
from batdetect2.models.bottleneck import (
    DEFAULT_BOTTLENECK_CONFIG,
    BottleneckConfig,
    build_bottleneck,
)
from batdetect2.models.decoder import (
    DEFAULT_DECODER_CONFIG,
    DecoderConfig,
    build_decoder,
)
from batdetect2.models.encoder import (
    DEFAULT_ENCODER_CONFIG,
    EncoderConfig,
    build_encoder,
)
from batdetect2.typing.models import (
    BackboneModel,
    BottleneckProtocol,
    DecoderProtocol,
    EncoderProtocol,
)

__all__ = [
    "BackboneImportConfig",
    "UNetBackbone",
    "BackboneConfig",
    "load_backbone_config",
    "build_backbone",
]


class UNetBackboneConfig(BaseConfig):
    """Configuration for a U-Net-style encoder-decoder backbone.

    All fields have sensible defaults that reproduce the standard BatDetect2
    architecture, so you can start with ``UNetBackboneConfig()`` and override
    only the fields you want to change.

    Attributes
    ----------
    name : str
        Discriminator field used by the backbone registry; always
        ``"UNetBackbone"``.
    input_height : int
        Number of frequency bins in the input spectrogram. Defaults to
        ``128``.
    in_channels : int
        Number of channels in the input spectrogram (e.g. ``1`` for a
        standard mel-spectrogram). Defaults to ``1``.
    encoder : EncoderConfig
        Configuration for the downsampling path. Defaults to
        ``DEFAULT_ENCODER_CONFIG``.
    bottleneck : BottleneckConfig
        Configuration for the bottleneck. Defaults to
        ``DEFAULT_BOTTLENECK_CONFIG``.
    decoder : DecoderConfig
        Configuration for the upsampling path. Defaults to
        ``DEFAULT_DECODER_CONFIG``.
    """

    name: Literal["UNetBackbone"] = "UNetBackbone"
    input_height: int = 128
    in_channels: int = 1
    encoder: EncoderConfig = DEFAULT_ENCODER_CONFIG
    bottleneck: BottleneckConfig = DEFAULT_BOTTLENECK_CONFIG
    decoder: DecoderConfig = DEFAULT_DECODER_CONFIG


backbone_registry: Registry[BackboneModel, []] = Registry("backbone")


@add_import_config(backbone_registry)
class BackboneImportConfig(ImportConfig):
    """Use any callable as a backbone model.

    Set ``name="import"`` and provide a ``target`` pointing to any
    callable to use it instead of a built-in option.
    """

    name: Literal["import"] = "import"


class UNetBackbone(BackboneModel):
    """U-Net-style encoder-decoder backbone network.

    Combines an encoder, a bottleneck, and a decoder into a single module
    that produces a high-resolution feature map from an input spectrogram.
    Skip connections from each encoder stage are added element-wise to the
    corresponding decoder stage input.

    Input spectrograms of arbitrary width are handled automatically: the
    backbone pads the input so that its dimensions are divisible by
    ``divide_factor`` and removes the padding from the output.

    Instances are typically created via ``build_backbone``.

    Attributes
    ----------
    input_height : int
        Expected height (frequency bins) of the input spectrogram.
    out_channels : int
        Number of channels in the output feature map (taken from the
        decoder's output channel count).
    encoder : EncoderProtocol
        The instantiated encoder module.
    decoder : DecoderProtocol
        The instantiated decoder module.
    bottleneck : BottleneckProtocol
        The instantiated bottleneck module.
    divide_factor : int
        The total spatial downsampling factor applied by the encoder
        (``input_height // encoder.output_height``). The input width is
        padded to be a multiple of this value before processing.
    """

    def __init__(
        self,
        input_height: int,
        encoder: EncoderProtocol,
        decoder: DecoderProtocol,
        bottleneck: BottleneckProtocol,
    ):
        """Initialise the backbone network.

        Parameters
        ----------
        input_height : int
            Expected height (frequency bins) of the input spectrogram.
        encoder : EncoderProtocol
            An initialised encoder module.
        decoder : DecoderProtocol
            An initialised decoder module. Its ``output_height`` must equal
            ``input_height``; a ``ValueError`` is raised otherwise.
        bottleneck : BottleneckProtocol
            An initialised bottleneck module.
        """
        super().__init__()
        self.input_height = input_height

        self.encoder = encoder
        self.decoder = decoder
        self.bottleneck = bottleneck

        self.out_channels = decoder.out_channels

        # Down/Up scaling factor. Need to ensure inputs are divisible by
        # this factor in order to be processed by the down/up scaling layers
        # and recover the correct shape
        self.divide_factor = input_height // self.encoder.output_height

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """Produce a feature map from an input spectrogram.

        Pads the input if necessary, runs it through the encoder, then
        the bottleneck, then the decoder (incorporating encoder skip
        connections), and finally removes any padding added earlier.

        Parameters
        ----------
        spec : torch.Tensor
            Input spectrogram tensor, shape
            ``(B, C_in, H_in, W_in)``. ``H_in`` must equal
            ``self.input_height``; ``W_in`` can be any positive integer.

        Returns
        -------
        torch.Tensor
            Feature map tensor, shape ``(B, C_out, H_in, W_in)``, where
            ``C_out`` is ``self.out_channels``. The spatial dimensions
            always match those of the input.
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

        return x

    @backbone_registry.register(UNetBackboneConfig)
    @staticmethod
    def from_config(config: UNetBackboneConfig) -> BackboneModel:
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

        return UNetBackbone(
            input_height=config.input_height,
            encoder=encoder,
            decoder=decoder,
            bottleneck=bottleneck,
        )


BackboneConfig = Annotated[
    UNetBackboneConfig | BackboneImportConfig,
    Field(discriminator="name"),
]


def build_backbone(config: BackboneConfig | None = None) -> BackboneModel:
    """Build a backbone network from configuration.

    Looks up the backbone class corresponding to ``config.name`` in the
    backbone registry and calls its ``from_config`` method. If no
    configuration is provided, a default ``UNetBackbone`` is returned.

    Parameters
    ----------
    config : BackboneConfig, optional
        A configuration object describing the desired backbone. Currently
        ``UNetBackboneConfig`` is the only supported type. Defaults to
        ``UNetBackboneConfig()`` if not provided.

    Returns
    -------
    BackboneModel
        An initialised backbone module.
    """
    config = config or UNetBackboneConfig()
    return backbone_registry.build(config)


def _pad_adjust(
    spec: torch.Tensor,
    factor: int = 32,
) -> tuple[torch.Tensor, int, int]:
    """Pad a tensor's height and width to be divisible by ``factor``.

    Adds zero-padding to the bottom and right edges of the tensor so that
    both dimensions are exact multiples of ``factor``. If both dimensions
    are already divisible, the tensor is returned unchanged.

    Parameters
    ----------
    spec : torch.Tensor
        Input tensor, typically shape ``(B, C, H, W)``.
    factor : int, default=32
        The factor that both H and W should be divisible by after padding.

    Returns
    -------
    tuple[torch.Tensor, int, int]
        - Padded tensor.
        - Number of rows added to the height (``h_pad``).
        - Number of columns added to the width (``w_pad``).
    """
    h, w = spec.shape[-2:]
    h_pad = -h % factor
    w_pad = -w % factor

    if h_pad == 0 and w_pad == 0:
        return spec, 0, 0

    return F.pad(spec, (0, w_pad, 0, h_pad)), h_pad, w_pad


def _restore_pad(
    x: torch.Tensor, h_pad: int = 0, w_pad: int = 0
) -> torch.Tensor:
    """Remove padding previously added by ``_pad_adjust``.

    Trims ``h_pad`` rows from the bottom and ``w_pad`` columns from the
    right of the tensor, restoring its original spatial dimensions.

    Parameters
    ----------
    x : torch.Tensor
        Padded tensor, typically shape ``(B, C, H_padded, W_padded)``.
    h_pad : int, default=0
        Number of rows to remove from the bottom.
    w_pad : int, default=0
        Number of columns to remove from the right.

    Returns
    -------
    torch.Tensor
        Tensor with padding removed, shape
        ``(B, C, H_padded - h_pad, W_padded - w_pad)``.
    """
    if h_pad > 0:
        x = x[..., :-h_pad, :]

    if w_pad > 0:
        x = x[..., :-w_pad]

    return x


def load_backbone_config(
    path: data.PathLike,
    field: str | None = None,
) -> BackboneConfig:
    """Load a backbone configuration from a YAML or JSON file.

    Reads the file at ``path``, optionally descends into a named sub-field,
    and validates the result against the ``BackboneConfig`` discriminated
    union.

    Parameters
    ----------
    path : PathLike
        Path to the configuration file. Both YAML and JSON formats are
        supported.
    field : str, optional
        Dot-separated key path to the sub-field that contains the backbone
        configuration (e.g. ``"model"``). If ``None``, the root of the
        file is used.

    Returns
    -------
    BackboneConfig
        A validated backbone configuration object (currently always a
        ``UNetBackboneConfig`` instance).

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValidationError
        If the loaded data does not conform to a known ``BackboneConfig``
        schema.
    """
    return load_config(
        path,
        schema=TypeAdapter(BackboneConfig),
        field=field,
    )
