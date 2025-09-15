from typing import List, Optional

from pydantic import Field
from soundevent import data

from batdetect2.configs import BaseConfig, load_config
from batdetect2.evaluate.match import MatchConfig, StartTimeMatchConfig
from batdetect2.evaluate.metrics import (
    ClassificationAPConfig,
    DetectionAPConfig,
    MetricConfig,
)
from batdetect2.evaluate.plots import ExampleGalleryConfig, PlotConfig

__all__ = [
    "EvaluationConfig",
    "load_evaluation_config",
]


class EvaluationConfig(BaseConfig):
    ignore_start_end: float = 0.01
    match: MatchConfig = Field(default_factory=StartTimeMatchConfig)
    metrics: List[MetricConfig] = Field(
        default_factory=lambda: [
            DetectionAPConfig(),
            ClassificationAPConfig(),
        ]
    )
    plots: List[PlotConfig] = Field(
        default_factory=lambda: [
            ExampleGalleryConfig(),
        ]
    )


def load_evaluation_config(
    path: data.PathLike,
    field: Optional[str] = None,
) -> EvaluationConfig:
    return load_config(path, schema=EvaluationConfig, field=field)
