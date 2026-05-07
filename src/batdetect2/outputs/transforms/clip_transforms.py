from typing import Annotated, Literal

from pydantic import Field
from soundevent.geometry import compute_bounds

from batdetect2.core import (
    BaseConfig,
    ImportConfig,
    Registry,
    add_import_config,
)
from batdetect2.outputs.types import ClipDetectionsTransform
from batdetect2.postprocess.types import ClipDetections, Detection

__all__ = [
    "ClipDetectionsTransformConfig",
    "clip_transforms",
]


clip_transforms: Registry[ClipDetectionsTransform, []] = Registry(
    "clip_detection_transform"
)


@add_import_config(clip_transforms)
class ClipDetectionsTransformImportConfig(ImportConfig):
    name: Literal["import"] = "import"


class RemoveAboveNyquistConfig(BaseConfig):
    """Configuration for `RemoveAboveNyquist`.

    Defines parameters for removing detections above the Nyquist frequency.

    Attributes
    ----------
    name : Literal["remove_above_nyquist"]
        The unique identifier for this transform type.
    min_freq : float
        The minimum frequency (in Hz) for detections to be kept.
    """

    name: Literal["remove_above_nyquist"] = "remove_above_nyquist"
    mode: Literal["low_freq", "high_freq"] = "high_freq"
    buffer: float = 0


class RemoveAboveNyquist:
    def __init__(self, mode: Literal["low_freq", "high_freq"], buffer: float):
        self.mode = mode
        self.buffer = buffer

    def __call__(self, detections: ClipDetections) -> ClipDetections:
        recording = detections.clip.recording
        nyquist = recording.samplerate / 2
        threshold = nyquist - self.buffer

        return ClipDetections(
            clip=detections.clip,
            detections=[
                detection
                for detection in detections.detections
                if self._is_below_threshold(detection, threshold)
            ],
        )

    def _is_below_threshold(
        self,
        detection: Detection,
        threshold: float,
    ) -> bool:
        _, low_freq, _, high_freq = compute_bounds(detection.geometry)

        if self.mode == "low_freq":
            return low_freq < threshold

        return high_freq < threshold

    @clip_transforms.register(RemoveAboveNyquistConfig)
    @staticmethod
    def from_config(config: RemoveAboveNyquistConfig):
        return RemoveAboveNyquist(
            mode=config.mode,
            buffer=config.buffer,
        )


class RemoveAtEdgesConfig(BaseConfig):
    """Configuration for `RemoveAtEdges`.

    Defines parameters for removing detections at the edges of the clip.

    Attributes
    ----------
    name : Literal["remove_at_edges"]
        The unique identifier for this transform type.
    buffer : float
        The amount of time (in seconds) to remove detections from the edge.
    mode : Literal["start_time", "end_time", "both"]
        Criteria for removing detections at the edges of the clip.
        If "start_time", remove detections with a start time within the
        buffer. If "end_time", remove detections with an end time within
        the buffer. If "both", remove detections with a start time within
        the buffer or an end time within the buffer.
    """

    name: Literal["remove_at_edges"] = "remove_at_edges"
    buffer: float = 0.1
    mode: Literal["start_time", "end_time", "both"] = "both"


class RemoveAtEdges:
    def __init__(
        self,
        buffer: float,
        mode: Literal["start_time", "end_time", "both"],
    ):
        self.buffer = buffer
        self.mode = mode

    def __call__(self, detections: ClipDetections) -> ClipDetections:
        clip = detections.clip
        start = clip.start_time + self.buffer
        end = clip.end_time - self.buffer

        return ClipDetections(
            clip=detections.clip,
            detections=[
                detection
                for detection in detections.detections
                if self._is_within_buffer(detection, start, end)
            ],
        )

    def _is_within_buffer(
        self,
        detection: Detection,
        start: float,
        end: float,
    ) -> bool:
        start_time, _, end_time, _ = compute_bounds(detection.geometry)

        if self.mode == "start_time":
            return (start_time >= start) and (start_time <= end)

        if self.mode == "end_time":
            return (end_time >= start) and (end_time <= end)

        return (start_time >= start) and (end_time <= end)

    @clip_transforms.register(RemoveAtEdgesConfig)
    @staticmethod
    def from_config(config: RemoveAtEdgesConfig):
        return RemoveAtEdges(
            buffer=config.buffer,
            mode=config.mode,
        )


ClipDetectionsTransformConfig = Annotated[
    ClipDetectionsTransformImportConfig
    | RemoveAboveNyquistConfig
    | RemoveAtEdgesConfig,
    Field(discriminator="name"),
]
