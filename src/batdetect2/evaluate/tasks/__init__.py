from typing import Annotated, Optional, Union

from pydantic import Field

from batdetect2.evaluate.tasks.base import tasks_registry
from batdetect2.evaluate.tasks.classification import ClassificationTaskConfig
from batdetect2.evaluate.tasks.clip_classification import (
    ClipClassificationTaskConfig,
)
from batdetect2.evaluate.tasks.clip_detection import ClipDetectionTaskConfig
from batdetect2.evaluate.tasks.detection import DetectionTaskConfig
from batdetect2.evaluate.tasks.top_class import TopClassDetectionTaskConfig
from batdetect2.targets import build_targets
from batdetect2.typing import EvaluatorProtocol, TargetProtocol

__all__ = [
    "TaskConfig",
    "build_task",
]


TaskConfig = Annotated[
    Union[
        ClassificationTaskConfig,
        DetectionTaskConfig,
        ClipDetectionTaskConfig,
        ClipClassificationTaskConfig,
        TopClassDetectionTaskConfig,
    ],
    Field(discriminator="name"),
]


def build_task(
    config: TaskConfig,
    targets: Optional[TargetProtocol] = None,
) -> EvaluatorProtocol:
    targets = targets or build_targets()
    return tasks_registry.build(config, targets)
