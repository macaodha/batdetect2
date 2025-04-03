from typing import Sequence, Tuple

import torch
import torch.nn.functional as F

from batdetect2.models.blocks import (
    ConvBlock,
    Decoder,
    DownscalingLayer,
    Encoder,
    SelfAttention,
    UpscalingLayer,
    VerticalConv,
)
from batdetect2.models.typing import BackboneModel

__all__ = [
    "Net2DFast",
    "Net2DFastNoAttn",
    "Net2DFastNoCoordConv",
]


class Net2DPlain(BackboneModel):
    downscaling_layer_type: DownscalingLayer = "ConvBlockDownStandard"
    upscaling_layer_type: UpscalingLayer = "ConvBlockUpStandard"

    def __init__(
        self,
        input_height: int = 128,
        encoder_channels: Sequence[int] = (1, 32, 64, 128),
        bottleneck_channels: int = 256,
        decoder_channels: Sequence[int] = (256, 64, 32, 32),
        out_channels: int = 32,
    ):
        super().__init__()

        self.input_height = input_height
        self.encoder_channels = tuple(encoder_channels)
        self.decoder_channels = tuple(decoder_channels)
        self.out_channels = out_channels

        if len(encoder_channels) != len(decoder_channels):
            raise ValueError(
                f"Mismatched encoder and decoder channel lists. "
                f"The encoder has {len(encoder_channels)} channels "
                f"(implying {len(encoder_channels) - 1} layers), "
                f"while the decoder has {len(decoder_channels)} channels "
                f"(implying {len(decoder_channels) - 1} layers). "
                f"These lengths must be equal."
            )

        self.divide_factor = 2 ** (len(encoder_channels) - 1)
        if self.input_height % self.divide_factor != 0:
            raise ValueError(
                f"Input height ({self.input_height}) must be divisible by "
                f"the divide factor ({self.divide_factor}). "
                f"This ensures proper upscaling after downscaling to recover "
                f"the original input height."
            )

        self.encoder = Encoder(
            channels=encoder_channels,
            input_height=self.input_height,
            layer_type=self.downscaling_layer_type,
        )

        self.conv_same_1 = ConvBlock(
            in_channels=encoder_channels[-1],
            out_channels=bottleneck_channels,
        )

        # bottleneck
        self.conv_vert = VerticalConv(
            in_channels=bottleneck_channels,
            out_channels=bottleneck_channels,
            input_height=self.input_height // (2**self.encoder.depth),
        )

        self.decoder = Decoder(
            channels=decoder_channels,
            input_height=self.input_height,
            layer_type=self.upscaling_layer_type,
        )

        self.conv_same_2 = ConvBlock(
            in_channels=decoder_channels[-1],
            out_channels=out_channels,
        )

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        spec, h_pad, w_pad = pad_adjust(spec, factor=self.divide_factor)

        # encoder
        residuals = self.encoder(spec)
        residuals[-1] = self.conv_same_1(residuals[-1])

        # bottleneck
        x = self.conv_vert(residuals[-1])
        x = x.repeat([1, 1, residuals[-1].shape[-2], 1])

        # decoder
        x = self.decoder(x, residuals=residuals)

        # Restore original size
        x = restore_pad(x, h_pad=h_pad, w_pad=w_pad)

        return self.conv_same_2(x)


class Net2DFast(Net2DPlain):
    downscaling_layer_type = "ConvBlockDownCoordF"
    upscaling_layer_type = "ConvBlockUpF"

    def __init__(
        self,
        input_height: int = 128,
        encoder_channels: Sequence[int] = (1, 32, 64, 128),
        bottleneck_channels: int = 256,
        decoder_channels: Sequence[int] = (256, 64, 32, 32),
        out_channels: int = 32,
    ):
        super().__init__(
            input_height=input_height,
            encoder_channels=encoder_channels,
            bottleneck_channels=bottleneck_channels,
            decoder_channels=decoder_channels,
            out_channels=out_channels,
        )

        self.att = SelfAttention(bottleneck_channels, bottleneck_channels)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        spec, h_pad, w_pad = pad_adjust(spec, factor=self.divide_factor)

        # encoder
        residuals = self.encoder(spec)
        residuals[-1] = self.conv_same_1(residuals[-1])

        # bottleneck
        x = self.conv_vert(residuals[-1])
        x = self.att(x)
        x = x.repeat([1, 1, residuals[-1].shape[-2], 1])

        # decoder
        x = self.decoder(x, residuals=residuals)

        # Restore original size
        x = restore_pad(x, h_pad=h_pad, w_pad=w_pad)

        return self.conv_same_2(x)


class Net2DFastNoAttn(Net2DPlain):
    downscaling_layer_type = "ConvBlockDownCoordF"
    upscaling_layer_type = "ConvBlockUpF"


class Net2DFastNoCoordConv(Net2DFast):
    downscaling_layer_type = "ConvBlockDownStandard"
    upscaling_layer_type = "ConvBlockUpStandard"


def pad_adjust(
    spec: torch.Tensor,
    factor: int = 32,
) -> Tuple[torch.Tensor, int, int]:
    print(spec.shape)
    h, w = spec.shape[2:]
    h_pad = -h % factor
    w_pad = -w % factor
    return F.pad(spec, (0, w_pad, 0, h_pad)), h_pad, w_pad


def restore_pad(
    x: torch.Tensor, h_pad: int = 0, w_pad: int = 0
) -> torch.Tensor:
    # Restore original size
    if h_pad > 0:
        x = x[:, :, :-h_pad, :]

    if w_pad > 0:
        x = x[:, :, :, :-w_pad]

    return x
