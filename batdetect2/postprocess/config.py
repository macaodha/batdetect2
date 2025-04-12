from typing import Optional

from pydantic import Field
from soundevent import data

from batdetect2.configs import BaseConfig, load_config

__all__ = [
    "PostprocessConfig",
    "load_postprocess_config",
]

NMS_KERNEL_SIZE = 9
DETECTION_THRESHOLD = 0.01
TOP_K_PER_SEC = 200


class PostprocessConfig(BaseConfig):
    """Configuration for postprocessing model outputs."""

    nms_kernel_size: int = Field(default=NMS_KERNEL_SIZE, gt=0)
    detection_threshold: float = Field(default=DETECTION_THRESHOLD, ge=0)
    min_freq: int = Field(default=10000, gt=0)
    max_freq: int = Field(default=120000, gt=0)
    top_k_per_sec: int = Field(default=TOP_K_PER_SEC, gt=0)


def load_postprocess_config(
    path: data.PathLike,
    field: Optional[str] = None,
) -> PostprocessConfig:
    return load_config(path, schema=PostprocessConfig, field=field)
