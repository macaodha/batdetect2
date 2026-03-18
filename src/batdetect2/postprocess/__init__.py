"""Main entry point for the BatDetect2 Postprocessing pipeline."""

from batdetect2.postprocess.config import PostprocessConfig
from batdetect2.postprocess.nms import non_max_suppression
from batdetect2.postprocess.postprocessor import (
    Postprocessor,
    build_postprocessor,
)
from batdetect2.postprocess.types import (
    ClipDetections,
    ClipDetectionsArray,
    ClipDetectionsTensor,
    ClipPrediction,
    Detection,
    GeometryDecoder,
    PostprocessorProtocol,
)

__all__ = [
    "ClipDetections",
    "ClipDetectionsArray",
    "ClipDetectionsTensor",
    "ClipPrediction",
    "Detection",
    "GeometryDecoder",
    "PostprocessConfig",
    "Postprocessor",
    "PostprocessorProtocol",
    "build_postprocessor",
    "non_max_suppression",
]
