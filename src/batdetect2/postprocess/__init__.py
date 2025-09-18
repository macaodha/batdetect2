"""Main entry point for the BatDetect2 Postprocessing pipeline."""

from batdetect2.postprocess.config import (
    PostprocessConfig,
    load_postprocess_config,
)
from batdetect2.postprocess.decoding import (
    convert_raw_predictions_to_clip_prediction,
    to_raw_predictions,
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
    "convert_raw_predictions_to_clip_prediction",
    "to_raw_predictions",
    "load_postprocess_config",
    "non_max_suppression",
]
