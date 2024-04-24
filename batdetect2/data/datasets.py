from typing import Callable, Generic, Iterable, List, TypeVar

from soundevent import data
from torch.utils.data import Dataset

__all__ = [
    "ClipAnnotationDataset",
    "ClipDataset",
]


E = TypeVar("E")


class ClipAnnotationDataset(Dataset, Generic[E]):

    clip_annotations: List[data.ClipAnnotation]

    transform: Callable[[data.ClipAnnotation], E]

    def __init__(
        self,
        clip_annotations: Iterable[data.ClipAnnotation],
        transform: Callable[[data.ClipAnnotation], E],
        name: str = "ClipAnnotationDataset",
    ):
        self.clip_annotations = list(clip_annotations)
        self.transform = transform
        self.name = name

    def __len__(self) -> int:
        return len(self.clip_annotations)

    def __getitem__(self, idx: int) -> E:
        return self.transform(self.clip_annotations[idx])


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
