
from pydantic import Field
from soundevent import data

from batdetect2.core.configs import BaseConfig, load_config
from batdetect2.evaluate.config import EvaluationConfig
from batdetect2.logging import LoggerConfig, TensorBoardLoggerConfig
from batdetect2.train.checkpoints import CheckpointConfig
from batdetect2.train.dataset import TrainLoaderConfig, ValLoaderConfig
from batdetect2.train.labels import LabelConfig
from batdetect2.train.losses import LossConfig

__all__ = [
    "TrainingConfig",
    "load_train_config",
]


class PLTrainerConfig(BaseConfig):
    accelerator: str = "auto"
    accumulate_grad_batches: int = 1
    deterministic: bool = True
    check_val_every_n_epoch: int = 1
    devices: str | int = "auto"
    enable_checkpointing: bool = True
    gradient_clip_val: float | None = None
    limit_train_batches: int | float | None = None
    limit_test_batches: int | float | None = None
    limit_val_batches: int | float | None = None
    log_every_n_steps: int | None = None
    max_epochs: int | None = 200
    min_epochs: int | None = None
    max_steps: int | None = None
    min_steps: int | None = None
    max_time: str | None = None
    precision: str | None = None
    val_check_interval: int | float | None = None


class OptimizerConfig(BaseConfig):
    learning_rate: float = 1e-3
    t_max: int = 100


class TrainingConfig(BaseConfig):
    train_loader: TrainLoaderConfig = Field(default_factory=TrainLoaderConfig)
    val_loader: ValLoaderConfig = Field(default_factory=ValLoaderConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    trainer: PLTrainerConfig = Field(default_factory=PLTrainerConfig)
    logger: LoggerConfig = Field(default_factory=TensorBoardLoggerConfig)
    labels: LabelConfig = Field(default_factory=LabelConfig)
    validation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    checkpoints: CheckpointConfig = Field(default_factory=CheckpointConfig)


def load_train_config(
    path: data.PathLike,
    field: str | None = None,
) -> TrainingConfig:
    return load_config(path, schema=TrainingConfig, field=field)
