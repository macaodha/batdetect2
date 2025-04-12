from enum import Enum
from typing import Optional, Tuple

from soundevent.data import PathLike

from batdetect2.configs import BaseConfig, load_config

__all__ = [
    "ModelType",
    "ModelConfig",
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
