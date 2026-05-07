from collections.abc import Callable, Sequence
from typing import Generic, Protocol, TypeVar

from soundevent import data
from soundevent.data import PathLike

from batdetect2.postprocess.types import (
    ClipDetections,
    ClipDetectionsTensor,
    Detection,
)

__all__ = [
    "ClipDetectionsTransform",
    "DetectionTransform",
    "OutputFormatterProtocol",
    "OutputTransformProtocol",
]

T = TypeVar("T")


class OutputFormatterProtocol(Protocol, Generic[T]):
    def format(self, predictions: Sequence[ClipDetections]) -> list[T]: ...

    def save(
        self,
        predictions: Sequence[T],
        path: PathLike,
        audio_dir: PathLike | None = None,
    ) -> None: ...

    def load(self, path: PathLike) -> list[T]: ...


DetectionTransform = Callable[[Detection], Detection | None]
ClipDetectionsTransform = Callable[[ClipDetections], ClipDetections]


class OutputTransformProtocol(Protocol):
    def to_detections(
        self,
        detections: ClipDetectionsTensor,
        start_time: float = 0,
    ) -> list[Detection]: ...

    def to_clip_detections(
        self,
        detections: ClipDetectionsTensor,
        clip: data.Clip,
    ) -> ClipDetections: ...

    def transform_detections(
        self,
        detections: Sequence[Detection],
    ) -> list[Detection]: ...

    def transform_clip_detections(
        self,
        prediction: ClipDetections,
    ) -> ClipDetections: ...
