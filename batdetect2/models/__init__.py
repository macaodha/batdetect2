from enum import Enum
from typing import Optional, Tuple

from soundevent.data import PathLike

from batdetect2.configs import BaseConfig, load_config
from batdetect2.models.backbones import (
    Net2DFast,
    Net2DFastNoAttn,
    Net2DFastNoCoordConv,
    Net2DPlain,
)
from batdetect2.models.heads import BBoxHead, ClassifierHead
from batdetect2.models.typing import BackboneModel

__all__ = [
    "BBoxHead",
    "ClassifierHead",
    "ModelConfig",
    "ModelType",
    "Net2DFast",
    "Net2DFastNoAttn",
    "Net2DFastNoCoordConv",
    "build_architecture",
    "load_model_config",
]


class ModelType(str, Enum):
    Net2DFast = "Net2DFast"
    Net2DFastNoAttn = "Net2DFastNoAttn"
    Net2DFastNoCoordConv = "Net2DFastNoCoordConv"
    Net2DPlain = "Net2DPlain"


class ModelConfig(BaseConfig):
    name: ModelType = ModelType.Net2DFast
    input_height: int = 128
    encoder_channels: Tuple[int, ...] = (1, 32, 64, 128)
    bottleneck_channels: int = 256
    decoder_channels: Tuple[int, ...] = (256, 64, 32, 32)
    out_channels: int = 32


def load_model_config(
    path: PathLike, field: Optional[str] = None
) -> ModelConfig:
    return load_config(path, schema=ModelConfig, field=field)


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
