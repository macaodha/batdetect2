from typing import Optional

from pydantic import Field
from soundevent import data

from batdetect2.core.configs import BaseConfig, load_config
from batdetect2.evaluate.evaluator import (
    EvaluatorConfig,
    MultipleEvaluatorConfig,
)
from batdetect2.logging import CSVLoggerConfig, LoggerConfig

__all__ = [
    "EvaluationConfig",
    "load_evaluation_config",
]


class EvaluationConfig(BaseConfig):
    evaluator: EvaluatorConfig = Field(default_factory=MultipleEvaluatorConfig)
    logger: LoggerConfig = Field(default_factory=CSVLoggerConfig)


def load_evaluation_config(
    path: data.PathLike,
    field: Optional[str] = None,
) -> EvaluationConfig:
    return load_config(path, schema=EvaluationConfig, field=field)
