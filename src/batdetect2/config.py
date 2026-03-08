from typing import Literal

from pydantic import Field
from soundevent.data import PathLike

from batdetect2.audio import AudioConfig
from batdetect2.core.configs import BaseConfig, load_config
from batdetect2.data.predictions import OutputFormatConfig
from batdetect2.data.predictions.raw import RawOutputConfig
from batdetect2.evaluate.config import (
    EvaluationConfig,
    get_default_eval_config,
)
from batdetect2.inference.config import InferenceConfig
from batdetect2.models.backbones import BackboneConfig, UNetBackboneConfig
from batdetect2.postprocess.config import PostprocessConfig
from batdetect2.preprocess.config import PreprocessingConfig
from batdetect2.targets.config import TargetConfig
from batdetect2.train.config import TrainingConfig

__all__ = [
    "BatDetect2Config",
    "load_full_config",
    "validate_config",
]


class BatDetect2Config(BaseConfig):
    config_version: Literal["v1"] = "v1"

    train: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(
        default_factory=get_default_eval_config
    )
    model: BackboneConfig = Field(default_factory=UNetBackboneConfig)
    preprocess: PreprocessingConfig = Field(
        default_factory=PreprocessingConfig
    )
    postprocess: PostprocessConfig = Field(default_factory=PostprocessConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    targets: TargetConfig = Field(default_factory=TargetConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    output: OutputFormatConfig = Field(default_factory=RawOutputConfig)


def validate_config(config: dict | None) -> BatDetect2Config:
    if config is None:
        return BatDetect2Config()

    return BatDetect2Config.model_validate(config)


def load_full_config(
    path: PathLike,
    field: str | None = None,
) -> BatDetect2Config:
    return load_config(path, schema=BatDetect2Config, field=field)
