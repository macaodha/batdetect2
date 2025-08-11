from typing import Optional, Union

from pydantic import Field
from soundevent import data

from batdetect2.configs import BaseConfig, load_config
from batdetect2.evaluate.config import EvaluationConfig
from batdetect2.models import BackboneConfig
from batdetect2.postprocess import PostprocessConfig
from batdetect2.preprocess import PreprocessingConfig
from batdetect2.targets import TargetConfig
from batdetect2.train.augmentations import (
    DEFAULT_AUGMENTATION_CONFIG,
    AugmentationsConfig,
)
from batdetect2.train.clips import ClipingConfig
from batdetect2.train.logging import CSVLoggerConfig, LoggerConfig
from batdetect2.train.losses import LossConfig

__all__ = [
    "TrainingConfig",
    "load_train_config",
    "FullTrainingConfig",
    "load_full_training_config",
]


class PLTrainerConfig(BaseConfig):
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


class DataLoaderConfig(BaseConfig):
    batch_size: int
    shuffle: bool
    num_workers: int = 0


DEFAULT_TRAIN_LOADER_CONFIG = DataLoaderConfig(batch_size=8, shuffle=True)
DEFAULT_VAL_LOADER_CONFIG = DataLoaderConfig(batch_size=8, shuffle=False)


class LoadersConfig(BaseConfig):
    train: DataLoaderConfig = Field(
        default_factory=lambda: DEFAULT_TRAIN_LOADER_CONFIG.model_copy()
    )
    val: DataLoaderConfig = Field(
        default_factory=lambda: DEFAULT_VAL_LOADER_CONFIG.model_copy()
    )


class TrainingConfig(BaseConfig):
    learning_rate: float = 1e-3
    t_max: int = 100
    dataloaders: LoadersConfig = Field(default_factory=LoadersConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    augmentations: Optional[AugmentationsConfig] = Field(
        default_factory=lambda: DEFAULT_AUGMENTATION_CONFIG
    )
    cliping: ClipingConfig = Field(default_factory=ClipingConfig)
    trainer: PLTrainerConfig = Field(default_factory=PLTrainerConfig)
    logger: LoggerConfig = Field(default_factory=CSVLoggerConfig)


def load_train_config(
    path: data.PathLike,
    field: Optional[str] = None,
) -> TrainingConfig:
    return load_config(path, schema=TrainingConfig, field=field)


class FullTrainingConfig(BaseConfig):
    """Full training configuration."""

    train: TrainingConfig = Field(default_factory=TrainingConfig)
    targets: TargetConfig = Field(default_factory=TargetConfig)
    model: BackboneConfig = Field(default_factory=BackboneConfig)
    preprocess: PreprocessingConfig = Field(
        default_factory=PreprocessingConfig
    )
    postprocess: PostprocessConfig = Field(default_factory=PostprocessConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)


def load_full_training_config(
    path: data.PathLike,
    field: Optional[str] = None,
) -> FullTrainingConfig:
    """Load the full training configuration."""
    return load_config(path, schema=FullTrainingConfig, field=field)
