from typing import Literal, Optional

from pydantic import Field
from soundevent.data import PathLike

from batdetect2.audio import AudioConfig
from batdetect2.core import BaseConfig
from batdetect2.core.configs import load_config
from batdetect2.evaluate.config import EvaluationConfig
from batdetect2.inference.config import InferenceConfig
from batdetect2.models.config import BackboneConfig
from batdetect2.postprocess.config import PostprocessConfig
from batdetect2.preprocess.config import PreprocessingConfig
from batdetect2.targets.config import TargetConfig
from batdetect2.train.config import TrainingConfig

__all__ = [
    "BatDetect2Config",
    "load_full_config",
]


class BatDetect2Config(BaseConfig):
    config_version: Literal["v1"] = "v1"

    train: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    model: BackboneConfig = Field(default_factory=BackboneConfig)
    preprocess: PreprocessingConfig = Field(
        default_factory=PreprocessingConfig
    )
    postprocess: PostprocessConfig = Field(default_factory=PostprocessConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    targets: TargetConfig = Field(default_factory=TargetConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)


def load_full_config(
    path: PathLike,
    field: Optional[str] = None,
) -> BatDetect2Config:
    return load_config(path, schema=BatDetect2Config, field=field)
