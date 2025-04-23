from typing import Optional

from pydantic import Field
from soundevent.data import PathLike

from batdetect2.configs import BaseConfig, load_config
from batdetect2.train.augmentations import (
    DEFAULT_AUGMENTATION_CONFIG,
    AugmentationsConfig,
)
from batdetect2.train.clips import ClipingConfig
from batdetect2.train.losses import LossConfig

__all__ = [
    "OptimizerConfig",
    "TrainingConfig",
    "load_train_config",
]


class OptimizerConfig(BaseConfig):
    learning_rate: float = 1e-3
    t_max: int = 100


class TrainingConfig(BaseConfig):
    batch_size: int = 32

    loss: LossConfig = Field(default_factory=LossConfig)

    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)

    augmentations: AugmentationsConfig = Field(
        default_factory=lambda: DEFAULT_AUGMENTATION_CONFIG
    )

    cliping: ClipingConfig = Field(default_factory=ClipingConfig)


def load_train_config(
    path: PathLike,
    field: Optional[str] = None,
) -> TrainingConfig:
    return load_config(path, schema=TrainingConfig, field=field)
