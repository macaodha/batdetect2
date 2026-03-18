from typing import List

from pydantic import Field

from batdetect2.core.configs import BaseConfig
from batdetect2.evaluate.tasks import TaskConfig
from batdetect2.evaluate.tasks.classification import ClassificationTaskConfig
from batdetect2.evaluate.tasks.detection import DetectionTaskConfig

__all__ = [
    "EvaluationConfig",
]


class EvaluationConfig(BaseConfig):
    tasks: List[TaskConfig] = Field(
        default_factory=lambda: [
            DetectionTaskConfig(),
            ClassificationTaskConfig(),
        ]
    )


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
