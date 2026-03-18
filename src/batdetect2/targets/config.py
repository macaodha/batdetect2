from collections import Counter
from typing import List

from pydantic import Field, field_validator

from batdetect2.core.configs import BaseConfig
from batdetect2.targets.classes import (
    DEFAULT_CLASSES,
    DEFAULT_DETECTION_CLASS,
    TargetClassConfig,
)
from batdetect2.targets.rois import AnchorBBoxMapperConfig, ROIMapperConfig

__all__ = [
    "TargetConfig",
]


class TargetConfig(BaseConfig):
    detection_target: TargetClassConfig = Field(
        default=DEFAULT_DETECTION_CLASS
    )

    classification_targets: List[TargetClassConfig] = Field(
        default_factory=lambda: DEFAULT_CLASSES
    )

    roi: ROIMapperConfig = Field(default_factory=AnchorBBoxMapperConfig)

    @field_validator("classification_targets")
    def check_unique_class_names(cls, v: List[TargetClassConfig]):
        """Ensure all defined class names are unique."""
        names = [c.name for c in v]

        if len(names) != len(set(names)):
            name_counts = Counter(names)
            duplicates = [
                name for name, count in name_counts.items() if count > 1
            ]
            raise ValueError(
                "Class names must be unique. Found duplicates: "
                f"{', '.join(duplicates)}"
            )
        return v
