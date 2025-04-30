from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import xarray as xr
from soundevent import data
from torch.utils.data import Dataset

from batdetect2.train.augmentations import Augmentation
from batdetect2.train.types import ClipperProtocol, TrainExample

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
        dataset = self.get_dataset(idx)
        dataset, start_time, end_time = self.clipper.extract_clip(dataset)

        if self.augmentation:
            dataset = self.augmentation(dataset)

        return TrainExample(
            spec=self.to_tensor(dataset["spectrogram"]).unsqueeze(0),
            detection_heatmap=self.to_tensor(dataset["detection"]),
            class_heatmap=self.to_tensor(dataset["class"]),
            size_heatmap=self.to_tensor(dataset["size"]),
            idx=torch.tensor(idx),
            start_time=torch.tensor(start_time),
            end_time=torch.tensor(end_time),
        )

    @classmethod
    def from_directory(
        cls,
        directory: data.PathLike,
        clipper: ClipperProtocol,
        extension: str = ".nc",
        augmentation: Optional[Augmentation] = None,
    ):
        return cls(
            filenames=list_preprocessed_files(directory, extension),
            clipper=clipper,
            augmentation=augmentation,
        )

    def get_random_example(self) -> Tuple[xr.Dataset, float, float]:
        idx = np.random.randint(0, len(self))
        dataset = self.get_dataset(idx)

        dataset, start_time, end_time = self.clipper.extract_clip(dataset)

        return dataset, start_time, end_time

    def get_dataset(self, idx) -> xr.Dataset:
        return xr.open_dataset(self.filenames[idx])

    def get_clip_annotation(self, idx) -> data.ClipAnnotation:
        return data.ClipAnnotation.model_validate_json(
            self.get_dataset(idx).attrs["clip_annotation"]
        )

    def to_tensor(
        self,
        array: xr.DataArray,
        dtype=np.float32,
    ) -> torch.Tensor:
        return torch.nan_to_num(
            torch.tensor(array.values.astype(dtype)),
            nan=0,
        )


def list_preprocessed_files(
    directory: data.PathLike, extension: str = ".nc"
) -> List[Path]:
    return list(Path(directory).glob(f"*{extension}"))


class RandomExampleSource:
    def __init__(
        self,
        filenames: List[data.PathLike],
        clipper: ClipperProtocol,
    ):
        self.filenames = filenames
        self.clipper = clipper

    def __call__(self):
        index = int(np.random.randint(len(self.filenames)))
        filename = self.filenames[index]
        dataset = xr.open_dataset(filename)
        example, _, _ = self.clipper.extract_clip(dataset)
        return example
