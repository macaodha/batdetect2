from collections import Counter
from typing import List, Optional

from pydantic import Field, field_validator
from soundevent import data

from batdetect2.core.configs import BaseConfig, load_config
from batdetect2.targets.classes import (
    DEFAULT_CLASSES,
    DEFAULT_DETECTION_CLASS,
    TargetClassConfig,
)
from batdetect2.targets.rois import AnchorBBoxMapperConfig, ROIMapperConfig

__all__ = [
    "TargetConfig",
    "load_target_config",
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


def load_target_config(
    path: data.PathLike,
    field: str | None = None,
) -> TargetConfig:
    """Load the unified target configuration from a file.

    Reads a configuration file (typically YAML) and validates it against the
    `TargetConfig` schema, potentially extracting data from a nested field.

    Parameters
    ----------
    path : data.PathLike
        Path to the configuration file.
    field : str, optional
        Dot-separated path to a nested section within the file containing the
        target configuration. If None, the entire file content is used.

    Returns
    -------
    TargetConfig
        The loaded and validated unified target configuration object.

    Raises
    ------
    FileNotFoundError
        If the config file path does not exist.
    yaml.YAMLError
        If the file content is not valid YAML.
    pydantic.ValidationError
        If the loaded configuration data does not conform to the
        `TargetConfig` schema (including validation within nested configs
        like `ClassesConfig`).
    KeyError, TypeError
        If `field` specifies an invalid path within the loaded data.
    """
    return load_config(path=path, schema=TargetConfig, field=field)
