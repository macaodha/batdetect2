from typing import Generic, List, Protocol, Sequence, TypeVar

from soundevent.data import PathLike

from batdetect2.typing.postprocess import ClipDetections

__all__ = [
    "OutputFormatterProtocol",
]

T = TypeVar("T")


class OutputFormatterProtocol(Protocol, Generic[T]):
    def format(self, predictions: Sequence[ClipDetections]) -> List[T]: ...

    def save(
        self,
        predictions: Sequence[T],
        path: PathLike,
        audio_dir: PathLike | None = None,
    ) -> None: ...

    def load(self, path: PathLike) -> List[T]: ...
