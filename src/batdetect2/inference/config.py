from pydantic import Field

from batdetect2.core.configs import BaseConfig
from batdetect2.inference.dataset import InferenceLoaderConfig

__all__ = ["InferenceConfig"]


class ClipingConfig(BaseConfig):
    enabled: bool = True
    duration: float = 0.5
    overlap: float = 0.0
    max_empty: float = 0.0
    discard_empty: bool = True


class InferenceConfig(BaseConfig):
    loader: InferenceLoaderConfig = Field(
        default_factory=InferenceLoaderConfig
    )
    clipping: ClipingConfig = Field(default_factory=ClipingConfig)
