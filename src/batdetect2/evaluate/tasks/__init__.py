from typing import Annotated, Optional, Sequence, Union

from pydantic import Field
from soundevent import data

from batdetect2.evaluate.tasks.base import BaseTaskConfig, tasks_registry
from batdetect2.evaluate.tasks.classification import ClassificationTaskConfig
from batdetect2.evaluate.tasks.clip_classification import (
    ClipClassificationTaskConfig,
)
from batdetect2.evaluate.tasks.clip_detection import ClipDetectionTaskConfig
from batdetect2.evaluate.tasks.detection import DetectionTaskConfig
from batdetect2.evaluate.tasks.top_class import TopClassDetectionTaskConfig
from batdetect2.targets import build_targets
from batdetect2.typing import (
    BatDetect2Prediction,
    EvaluatorProtocol,
    TargetProtocol,
)

__all__ = [
    "TaskConfig",
    "build_task",
    "evaluate_task",
]


TaskConfig = Annotated[
    ClassificationTaskConfig | DetectionTaskConfig | ClipDetectionTaskConfig | ClipClassificationTaskConfig | TopClassDetectionTaskConfig,
    Field(discriminator="name"),
]


def build_task(
    config: TaskConfig,
    targets: TargetProtocol | None = None,
) -> EvaluatorProtocol:
    targets = targets or build_targets()
    return tasks_registry.build(config, targets)


def evaluate_task(
    clip_annotations: Sequence[data.ClipAnnotation],
    predictions: Sequence[BatDetect2Prediction],
    task: Optional["str"] = None,
    targets: TargetProtocol | None = None,
    config: TaskConfig | dict | None = None,
):
    if isinstance(config, BaseTaskConfig):
        task_obj = build_task(config, targets)
        return task_obj.evaluate(clip_annotations, predictions)

    if task is None:
        raise ValueError(
            "Task must be specified if a full config is not provided.",
        )

    config_class = tasks_registry.get_config_type(task)
    config = config_class.model_validate(config or {})  # type: ignore
    task_obj = build_task(config, targets)  # type: ignore
    return task_obj.evaluate(clip_annotations, predictions)
