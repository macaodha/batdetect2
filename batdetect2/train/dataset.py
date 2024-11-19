import os
from pathlib import Path
from typing import NamedTuple, Optional, Sequence, Union

import numpy as np
import torch
import xarray as xr
from soundevent import data
from torch.utils.data import Dataset

from batdetect2.train.augmentations import AugmentationsConfig, augment_example
from batdetect2.train.preprocess import PreprocessingConfig

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


def get_files(directory: PathLike, extension: str = ".nc") -> Sequence[Path]:
    return list(Path(directory).glob(f"*{extension}"))


class LabeledDataset(Dataset):
    def __init__(
        self,
        filenames: Sequence[PathLike],
        augment: bool = False,
        preprocessing_config: Optional[PreprocessingConfig] = None,
        augmentation_config: Optional[AugmentationsConfig] = None,
    ):
        self.filenames = filenames
        self.augment = augment
        self.preprocessing_config = (
            preprocessing_config or PreprocessingConfig()
        )
        self.agumentation_config = augmentation_config or AugmentationsConfig()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx) -> TrainExample:
        dataset = self.get_dataset(idx)

        if self.augment:
            dataset = augment_example(
                dataset,
                self.agumentation_config,
                preprocessing_config=self.preprocessing_config,
                others=self.get_random_example,
            )

        return TrainExample(
            spec=torch.tensor(
                dataset["spectrogram"].values.astype(np.float32)
            ).unsqueeze(0),
            detection_heatmap=torch.tensor(
                dataset["detection"].values.astype(np.float32)
            ),
            class_heatmap=torch.tensor(
                dataset["class"].values.astype(np.float32)
            ),
            size_heatmap=torch.tensor(
                dataset["size"].values.astype(np.float32)
            ),
            idx=torch.tensor(idx),
        )

    @classmethod
    def from_directory(cls, directory: PathLike, extension: str = ".nc"):
        return cls(get_files(directory, extension))

    def get_random_example(self) -> xr.Dataset:
        idx = np.random.randint(0, len(self))
        return self.get_dataset(idx)

    def get_dataset(self, idx) -> xr.Dataset:
        return xr.open_dataset(self.filenames[idx])

    def get_clip_annotation(self, idx) -> data.ClipAnnotation:
        return data.ClipAnnotation.model_validate_json(
            self.get_dataset(idx).attrs["clip_annotation"]
        )
