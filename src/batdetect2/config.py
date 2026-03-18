from typing import Literal

from pydantic import Field

from batdetect2.audio import AudioConfig
from batdetect2.core.configs import BaseConfig
from batdetect2.evaluate.config import (
    EvaluationConfig,
    get_default_eval_config,
)
from batdetect2.inference.config import InferenceConfig
from batdetect2.logging import AppLoggingConfig
from batdetect2.models import ModelConfig
from batdetect2.outputs import OutputsConfig
from batdetect2.train.config import TrainingConfig

__all__ = ["BatDetect2Config"]


class BatDetect2Config(BaseConfig):
    config_version: Literal["v1"] = "v1"

    train: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(
        default_factory=get_default_eval_config
    )
    model: ModelConfig = Field(default_factory=ModelConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    outputs: OutputsConfig = Field(default_factory=OutputsConfig)
    logging: AppLoggingConfig = Field(default_factory=AppLoggingConfig)
