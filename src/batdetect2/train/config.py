from typing import Optional, Union

from pydantic import Field
from soundevent import data

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


class TrainerConfig(BaseConfig):
    accelerator: str = "auto"
    accumulate_grad_batches: int = 1
    deterministic: bool = True
    check_val_every_n_epoch: int = 1
    devices: Union[str, int] = "auto"
    enable_checkpointing: bool = True
    gradient_clip_val: Optional[float] = None
    limit_train_batches: Optional[Union[int, float]] = None
    limit_test_batches: Optional[Union[int, float]] = None
    limit_val_batches: Optional[Union[int, float]] = None
    log_every_n_steps: Optional[int] = None
    max_epochs: Optional[int] = 200
    min_epochs: Optional[int] = None
    max_steps: Optional[int] = None
    min_steps: Optional[int] = None
    max_time: Optional[str] = None
    precision: Optional[str] = None
    val_check_interval: Optional[Union[int, float]] = None


class TrainingConfig(BaseConfig):
    batch_size: int = 8

    loss: LossConfig = Field(default_factory=LossConfig)

    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)

    augmentations: AugmentationsConfig = Field(
        default_factory=lambda: DEFAULT_AUGMENTATION_CONFIG
    )

    cliping: ClipingConfig = Field(default_factory=ClipingConfig)

    trainer: TrainerConfig = Field(default_factory=TrainerConfig)


def load_train_config(
    path: data.PathLike,
    field: Optional[str] = None,
) -> TrainingConfig:
    return load_config(path, schema=TrainingConfig, field=field)
