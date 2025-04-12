from typing import Optional

from batdetect2.models.backbones import (
    Net2DFast,
    Net2DFastNoAttn,
    Net2DFastNoCoordConv,
    Net2DPlain,
)
from batdetect2.models.config import ModelConfig, ModelType
from batdetect2.models.types import BackboneModel

__all__ = [
    "build_architecture",
]


def build_architecture(
    config: Optional[ModelConfig] = None,
) -> BackboneModel:
    config = config or ModelConfig()

    if config.name == ModelType.Net2DFast:
        return Net2DFast(
            input_height=config.input_height,
            encoder_channels=config.encoder_channels,
            bottleneck_channels=config.bottleneck_channels,
            decoder_channels=config.decoder_channels,
            out_channels=config.out_channels,
        )

    if config.name == ModelType.Net2DFastNoAttn:
        return Net2DFastNoAttn(
            input_height=config.input_height,
            encoder_channels=config.encoder_channels,
            bottleneck_channels=config.bottleneck_channels,
            decoder_channels=config.decoder_channels,
            out_channels=config.out_channels,
        )

    if config.name == ModelType.Net2DFastNoCoordConv:
        return Net2DFastNoCoordConv(
            input_height=config.input_height,
            encoder_channels=config.encoder_channels,
            bottleneck_channels=config.bottleneck_channels,
            decoder_channels=config.decoder_channels,
            out_channels=config.out_channels,
        )

    if config.name == ModelType.Net2DPlain:
        return Net2DPlain(
            input_height=config.input_height,
            encoder_channels=config.encoder_channels,
            bottleneck_channels=config.bottleneck_channels,
            decoder_channels=config.decoder_channels,
            out_channels=config.out_channels,
        )

    raise ValueError(f"Unknown model type: {config.name}")
