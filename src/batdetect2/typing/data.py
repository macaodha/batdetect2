from typing import Generic, List, Optional, Protocol, Sequence, TypeVar

from soundevent.data import PathLike

from batdetect2.typing.postprocess import BatDetect2Prediction

__all__ = [
    "OutputFormatterProtocol",
]

T = TypeVar("T")


class OutputFormatterProtocol(Protocol, Generic[T]):
    def format(
        self, predictions: Sequence[BatDetect2Prediction]
    ) -> List[T]: ...

    def save(
        self,
        predictions: Sequence[T],
        path: PathLike,
        audio_dir: PathLike | None = None,
    ) -> None: ...

    def load(self, path: PathLike) -> List[T]: ...
