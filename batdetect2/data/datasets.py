from typing import Callable, Generic, Iterable, List, TypeVar

from soundevent import data
from torch.utils.data import Dataset

__all__ = [
    "ClipDataset",
]


E = TypeVar("E")


class ClipDataset(Dataset, Generic[E]):
    clips: List[data.Clip]

    transform: Callable[[data.Clip], E]

    def __init__(
        self,
        clips: Iterable[data.Clip],
        transform: Callable[[data.Clip], E],
        name: str = "ClipDataset",
    ):
        self.clips = list(clips)
        self.transform = transform
        self.name = name

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> E:
        return self.transform(self.clips[idx])
