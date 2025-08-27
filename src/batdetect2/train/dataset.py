from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from soundevent import data
from torch.utils.data import Dataset

from batdetect2.train.augmentations import Augmentation
from batdetect2.train.preprocess import (
    list_preprocessed_files,
    load_preprocessed_example,
)
from batdetect2.typing import ClipperProtocol, TrainExample
from batdetect2.typing.train import PreprocessedExample

__all__ = [
    "LabeledDataset",
]


class LabeledDataset(Dataset):
    def __init__(
        self,
        filenames: Sequence[data.PathLike],
        clipper: ClipperProtocol,
        augmentation: Optional[Augmentation] = None,
    ):
        self.filenames = filenames
        self.clipper = clipper
        self.augmentation = augmentation

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx) -> TrainExample:
        example = self.get_example(idx)

        example, start_time, end_time = self.clipper(example)

        if self.augmentation:
            example = self.augmentation(example)

        return TrainExample(
            spec=example.spectrogram,
            detection_heatmap=example.detection_heatmap,
            class_heatmap=example.class_heatmap,
            size_heatmap=example.size_heatmap,
            idx=torch.tensor(idx),
            start_time=torch.tensor(start_time),
            end_time=torch.tensor(end_time),
        )

    @classmethod
    def from_directory(
        cls,
        directory: data.PathLike,
        clipper: ClipperProtocol,
        extension: str = ".npz",
        augmentation: Optional[Augmentation] = None,
    ):
        return cls(
            filenames=list_preprocessed_files(directory, extension),
            clipper=clipper,
            augmentation=augmentation,
        )

    def get_random_example(self) -> Tuple[PreprocessedExample, float, float]:
        idx = np.random.randint(0, len(self))
        dataset = self.get_example(idx)
        dataset, start_time, end_time = self.clipper(dataset)
        return dataset, start_time, end_time

    def get_example(self, idx) -> PreprocessedExample:
        return load_preprocessed_example(self.filenames[idx])

    def get_clip_annotation(self, idx) -> data.ClipAnnotation:
        item = np.load(self.filenames[idx], allow_pickle=True, mmap_mode="r+")
        return item["clip_annotation"].tolist()
