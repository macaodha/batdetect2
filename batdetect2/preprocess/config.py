from typing import Optional

from pydantic import Field
from soundevent.data import PathLike

from batdetect2.configs import BaseConfig, load_config
from batdetect2.preprocess.audio import (
    AudioConfig,
)
from batdetect2.preprocess.spectrogram import (
    SpectrogramConfig,
)

__all__ = [
    "PreprocessingConfig",
    "load_preprocessing_config",
]


class PreprocessingConfig(BaseConfig):
    """Configuration for preprocessing data."""

    audio: AudioConfig = Field(default_factory=AudioConfig)
    spectrogram: SpectrogramConfig = Field(default_factory=SpectrogramConfig)


def load_preprocessing_config(
    path: PathLike,
    field: Optional[str] = None,
) -> PreprocessingConfig:
    return load_config(path, schema=PreprocessingConfig, field=field)
