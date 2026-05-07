from collections.abc import Callable
from typing import Protocol

import numpy as np
from soundevent import data

__all__ = [
    "Position",
    "ROIMapperProtocol",
    "ROITargetMapper",
    "Size",
    "SoundEventDecoder",
    "SoundEventEncoder",
    "SoundEventFilter",
    "TargetProtocol",
]

SoundEventEncoder = Callable[[data.SoundEventAnnotation], str | None]
SoundEventDecoder = Callable[[str], list[data.Tag]]
SoundEventFilter = Callable[[data.SoundEventAnnotation], bool]

Position = tuple[float, float]
Size = np.ndarray


class TargetProtocol(Protocol):
    class_names: list[str]
    detection_class_tags: list[data.Tag]
    detection_class_name: str

    @classmethod
    def from_config(cls, config: dict) -> "TargetProtocol": ...

    def get_config(self) -> dict: ...

    def filter(self, sound_event: data.SoundEventAnnotation) -> bool: ...

    def encode_class(
        self,
        sound_event: data.SoundEventAnnotation,
    ) -> str | None: ...

    def decode_class(self, class_label: str) -> list[data.Tag]: ...


class ROIMapperProtocol(Protocol):
    dimension_names: list[str]

    def encode(
        self,
        sound_event: data.SoundEvent,
        class_name: str | None = None,
    ) -> tuple[Position, Size]: ...

    def decode(
        self,
        position: Position,
        size: Size,
        class_name: str | None = None,
    ) -> data.Geometry: ...

    def encode_roi(
        self,
        sound_event: data.SoundEventAnnotation,
    ) -> tuple[Position, Size]: ...

    def decode_roi(
        self,
        position: Position,
        size: Size,
        class_name: str | None = None,
    ) -> data.Geometry: ...


class ROITargetMapper(Protocol):
    dimension_names: list[str]

    def encode(
        self, sound_event: data.SoundEvent
    ) -> tuple[Position, Size]: ...

    def decode(self, position: Position, size: Size) -> data.Geometry: ...
