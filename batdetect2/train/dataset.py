import os
from pathlib import Path
from typing import NamedTuple, Optional, Sequence, Union

import numpy as np
import torch
import xarray as xr
from pydantic import Field
from soundevent import data
from torch.utils.data import Dataset

from batdetect2.configs import BaseConfig
from batdetect2.train.augmentations import (
    Augmentation,
    AugmentationsConfig,
    select_subclip,
)
from batdetect2.train.preprocess import PreprocessorProtocol
from batdetect2.utils.tensors import adjust_width

__all__ = [
    "TrainExample",
    "LabeledDataset",
]


PathLike = Union[Path, str, os.PathLike]


class TrainExample(NamedTuple):
    spec: torch.Tensor
    detection_heatmap: torch.Tensor
    class_heatmap: torch.Tensor
    size_heatmap: torch.Tensor
    idx: torch.Tensor


class SubclipConfig(BaseConfig):
    duration: Optional[float] = None
    width: int = 512
    random: bool = False


class DatasetConfig(BaseConfig):
    subclip: SubclipConfig = Field(default_factory=SubclipConfig)
    augmentation: AugmentationsConfig = Field(
        default_factory=AugmentationsConfig
    )


class LabeledDataset(Dataset):
    def __init__(
        self,
        preprocessor: PreprocessorProtocol,
        filenames: Sequence[PathLike],
        subclip: Optional[SubclipConfig] = None,
        augmentation: Optional[Augmentation] = None,
    ):
        self.preprocessor = preprocessor
        self.filenames = filenames
        self.subclip = subclip
        self.augmentation = augmentation

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx) -> TrainExample:
        dataset = self.get_dataset(idx)

        if self.subclip:
            dataset = select_subclip(
                dataset,
                duration=self.subclip.duration,
                width=self.subclip.width,
                random=self.subclip.random,
            )

        if self.augmentation:
            dataset = self.augmentation(dataset)

        return TrainExample(
            spec=self.to_tensor(dataset["spectrogram"]).unsqueeze(0),
            detection_heatmap=self.to_tensor(dataset["detection"]),
            class_heatmap=self.to_tensor(dataset["class"]),
            size_heatmap=self.to_tensor(dataset["size"]),
            idx=torch.tensor(idx),
        )

    @classmethod
    def from_directory(
        cls,
        directory: PathLike,
        preprocessor: PreprocessorProtocol,
        extension: str = ".nc",
        subclip: Optional[SubclipConfig] = None,
        augmentation: Optional[Augmentation] = None,
    ):
        return cls(
            preprocessor=preprocessor,
            filenames=get_preprocessed_files(directory, extension),
            subclip=subclip,
            augmentation=augmentation,
        )

    def get_random_example(self) -> xr.Dataset:
        idx = np.random.randint(0, len(self))
        dataset = self.get_dataset(idx)

        if self.subclip:
            dataset = select_subclip(
                dataset,
                duration=self.subclip.duration,
                width=self.subclip.width,
                random=self.subclip.random,
            )

        return dataset

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
        tensor = torch.tensor(array.values.astype(dtype))

        if not self.subclip:
            return tensor

        width = self.subclip.width
        return adjust_width(tensor, width)


def get_preprocessed_files(
    directory: PathLike, extension: str = ".nc"
) -> Sequence[Path]:
    return list(Path(directory).glob(f"*{extension}"))
