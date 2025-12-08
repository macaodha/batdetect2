from typing import List

from pydantic import Field
from soundevent import data

from batdetect2.core.configs import BaseConfig, load_config
from batdetect2.evaluate.tasks import (
    TaskConfig,
)
from batdetect2.evaluate.tasks.classification import ClassificationTaskConfig
from batdetect2.evaluate.tasks.detection import DetectionTaskConfig
from batdetect2.logging import CSVLoggerConfig, LoggerConfig

__all__ = [
    "EvaluationConfig",
    "load_evaluation_config",
]


class EvaluationConfig(BaseConfig):
    tasks: List[TaskConfig] = Field(
        default_factory=lambda: [
            DetectionTaskConfig(),
            ClassificationTaskConfig(),
        ]
    )
    logger: LoggerConfig = Field(default_factory=CSVLoggerConfig)


def get_default_eval_config() -> EvaluationConfig:
    return EvaluationConfig.model_validate(
        {
            "tasks": [
                {
                    "name": "sound_event_detection",
                    "plots": [
                        {"name": "pr_curve"},
                        {"name": "score_distribution"},
                    ],
                },
                {
                    "name": "sound_event_classification",
                    "plots": [
                        {"name": "pr_curve"},
                    ],
                },
            ]
        }
    )


def load_evaluation_config(
    path: data.PathLike,
    field: str | None = None,
) -> EvaluationConfig:
    return load_config(path, schema=EvaluationConfig, field=field)
