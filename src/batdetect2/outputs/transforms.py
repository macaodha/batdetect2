from collections.abc import Sequence
from dataclasses import replace
from typing import Protocol

from soundevent.geometry import shift_geometry

from batdetect2.core.configs import BaseConfig
from batdetect2.postprocess.types import ClipDetections, Detection

__all__ = [
    "OutputTransform",
    "OutputTransformConfig",
    "OutputTransformProtocol",
    "build_output_transform",
]


class OutputTransformConfig(BaseConfig):
    shift_time_to_clip_start: bool = True


class OutputTransformProtocol(Protocol):
    def __call__(
        self,
        predictions: Sequence[ClipDetections],
    ) -> list[ClipDetections]: ...

    def transform_detections(
        self,
        detections: Sequence[Detection],
        start_time: float = 0,
    ) -> list[Detection]: ...


def shift_detection_time(detection: Detection, time: float) -> Detection:
    geometry = shift_geometry(detection.geometry, time=time)
    return replace(detection, geometry=geometry)


class OutputTransform(OutputTransformProtocol):
    def __init__(self, shift_time_to_clip_start: bool = True):
        self.shift_time_to_clip_start = shift_time_to_clip_start

    def __call__(
        self,
        predictions: Sequence[ClipDetections],
    ) -> list[ClipDetections]:
        return [
            self.transform_prediction(prediction) for prediction in predictions
        ]

    def transform_prediction(
        self, prediction: ClipDetections
    ) -> ClipDetections:
        if not self.shift_time_to_clip_start:
            return prediction

        detections = self.transform_detections(
            prediction.detections,
            start_time=prediction.clip.start_time,
        )
        return ClipDetections(clip=prediction.clip, detections=detections)

    def transform_detections(
        self,
        detections: Sequence[Detection],
        start_time: float = 0,
    ) -> list[Detection]:
        if not self.shift_time_to_clip_start or start_time == 0:
            return list(detections)

        return [
            shift_detection_time(detection, time=start_time)
            for detection in detections
        ]


def build_output_transform(
    config: OutputTransformConfig | dict | None = None,
) -> OutputTransformProtocol:
    if config is None:
        config = OutputTransformConfig()

    if not isinstance(config, OutputTransformConfig):
        config = OutputTransformConfig.model_validate(config)

    return OutputTransform(
        shift_time_to_clip_start=config.shift_time_to_clip_start,
    )
