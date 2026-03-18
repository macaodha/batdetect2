"""Main entry point for the BatDetect2 Postprocessing pipeline."""

from batdetect2.postprocess.config import (
    PostprocessConfig,
    load_postprocess_config,
)
from batdetect2.postprocess.nms import non_max_suppression
from batdetect2.postprocess.postprocessor import (
    Postprocessor,
    build_postprocessor,
)

__all__ = [
    "PostprocessConfig",
    "Postprocessor",
    "build_postprocessor",
    "load_postprocess_config",
    "non_max_suppression",
]
