from typing import Literal

from batdetect2.core import BaseConfig
from batdetect2.evaluate.config import EvaluationConfig
from batdetect2.models.backbones import BackboneConfig
from batdetect2.preprocess import PreprocessingConfig
from batdetect2.train.config import TrainingConfig


class BatDetect2Config(BaseConfig):
    config_version: Literal["v1"] = "v1"

    train: TrainingConfig
    evaluation: EvaluationConfig
    model: BackboneConfig
    preprocess: PreprocessingConfig
