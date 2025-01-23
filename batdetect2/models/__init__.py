from enum import Enum

from batdetect2.configs import BaseConfig
from batdetect2.models.backbones import (
    Net2DFast,
    Net2DFastNoAttn,
    Net2DFastNoCoordConv,
)
from batdetect2.models.heads import BBoxHead, ClassifierHead
from batdetect2.models.typing import BackboneModel

__all__ = [
    "get_backbone",
    "Net2DFast",
    "Net2DFastNoAttn",
    "Net2DFastNoCoordConv",
    "ModelType",
    "BBoxHead",
    "ClassifierHead",
]


class ModelType(str, Enum):
    Net2DFast = "Net2DFast"
    Net2DFastNoAttn = "Net2DFastNoAttn"
    Net2DFastNoCoordConv = "Net2DFastNoCoordConv"


class ModelConfig(BaseConfig):
    name: ModelType = ModelType.Net2DFast
    num_features: int = 32


def get_backbone(
    config: ModelConfig,
    input_height: int = 128,
) -> BackboneModel:
    if config.name == ModelType.Net2DFast:
        return Net2DFast(
            input_height=input_height,
            num_features=config.num_features,
        )
    elif config.name == ModelType.Net2DFastNoAttn:
        return Net2DFastNoAttn(
            num_features=config.num_features,
            input_height=input_height,
        )
    elif config.name == ModelType.Net2DFastNoCoordConv:
        return Net2DFastNoCoordConv(
            num_features=config.num_features,
            input_height=input_height,
        )
    else:
        raise ValueError(f"Unknown model type: {config.name}")
