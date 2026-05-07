from collections.abc import Sequence
from dataclasses import replace
from typing import Annotated, Literal

from pydantic import Field
from soundevent.geometry import compute_bounds, shift_geometry

from batdetect2.core import (
    BaseConfig,
    ImportConfig,
    Registry,
    add_import_config,
)
from batdetect2.outputs.types import DetectionTransform
from batdetect2.postprocess.types import Detection

__all__ = [
    "DetectionTransformConfig",
    "detection_transforms",
    "shift_detection_time",
    "shift_detections_to_start_time",
]


detection_transforms: Registry[DetectionTransform, []] = Registry(
    "detection_transform"
)


@add_import_config(detection_transforms)
class DetectionTransformImportConfig(ImportConfig):
    name: Literal["import"] = "import"


class FilterByFrequencyConfig(BaseConfig):
    """Configuration for `FilterByFrequency`.

    Defines parameters for filtering detections by frequency.

    Attributes
    ----------
    name : Literal["filter_by_frequency"]
        The unique identifier for this transform type.
    min_freq : float
        The minimum frequency (in Hz) for detections to be kept.
    max_freq : float
        The maximum frequency (in Hz) for detections to be kept.
    mode : Literal["low_freq", "high_freq", "both"]
        Criteria for filtering detections by frequency.
        If "low_freq", keep detections with a low frequency within the
        specified range. If "high_freq", keep detections with a high
        frequency within the specified range. If "both", keep detections
        with a low frequency within the specified range or a high frequency
        within the specified range.
    """

    name: Literal["filter_by_frequency"] = "filter_by_frequency"
    min_freq: float = 0
    max_freq: float = float("inf")
    mode: Literal["low_freq", "high_freq", "both"] = "both"


class FilterByFrequency:
    def __init__(
        self,
        min_freq: float = 0,
        max_freq: float = float("inf"),
        mode: Literal["low_freq", "high_freq", "both"] = "both",
    ):
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.mode = mode

    def __call__(self, detection: Detection) -> Detection | None:
        if self._is_within_frequency_range(detection):
            return detection

    def _is_within_frequency_range(self, detection: Detection) -> bool:
        _, low_freq, _, high_freq = compute_bounds(detection.geometry)

        if self.mode == "low_freq":
            return (low_freq >= self.min_freq) and (low_freq <= self.max_freq)

        if self.mode == "high_freq":
            return (high_freq >= self.min_freq) and (
                high_freq <= self.max_freq
            )

        return (low_freq >= self.min_freq) or (high_freq <= self.max_freq)

    @detection_transforms.register(FilterByFrequencyConfig)
    @staticmethod
    def from_config(config: FilterByFrequencyConfig):
        return FilterByFrequency(
            min_freq=config.min_freq,
            max_freq=config.max_freq,
            mode=config.mode,
        )


class FilterByDurationConfig(BaseConfig):
    """Configuration for `FilterByDuration`.

    Defines parameters for filtering detections by duration.

    Attributes
    ----------
    name : Literal["filter_by_duration"]
        The unique identifier for this transform type.
    min_duration : float
        The minimum duration (in seconds) for detections to be kept.
    max_duration : float
        The maximum duration (in seconds) for detections to be kept.
    """

    name: Literal["filter_by_duration"] = "filter_by_duration"
    min_duration: float = 0
    max_duration: float = float("inf")


class FilterByDuration:
    def __init__(
        self,
        min_duration: float = 0,
        max_duration: float = float("inf"),
    ):
        self.min_duration = min_duration
        self.max_duration = max_duration

    def __call__(self, detection: Detection) -> Detection | None:
        if self._is_within_duration_range(detection):
            return detection

    def _is_within_duration_range(self, detection: Detection) -> bool:
        start_time, _, end_time, _ = compute_bounds(detection.geometry)
        duration = end_time - start_time
        return (duration >= self.min_duration) and (
            duration <= self.max_duration
        )

    @detection_transforms.register(FilterByDurationConfig)
    @staticmethod
    def from_config(config: FilterByDurationConfig):
        return FilterByDuration(
            min_duration=config.min_duration,
            max_duration=config.max_duration,
        )


DetectionTransformConfig = Annotated[
    DetectionTransformImportConfig
    | FilterByFrequencyConfig
    | FilterByDurationConfig,
    Field(discriminator="name"),
]


def shift_detection_time(detection: Detection, time: float) -> Detection:
    geometry = shift_geometry(detection.geometry, time=time)
    return replace(detection, geometry=geometry)


def shift_detections_to_start_time(
    detections: Sequence[Detection],
    start_time: float = 0,
) -> list[Detection]:
    if start_time == 0:
        return list(detections)

    return [
        shift_detection_time(detection, time=start_time)
        for detection in detections
    ]
