from collections.abc import Sequence
from typing import Generic, Protocol, TypeVar

from soundevent.data import PathLike

from batdetect2.postprocess.types import ClipDetections

__all__ = [
    "OutputFormatterProtocol",
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
