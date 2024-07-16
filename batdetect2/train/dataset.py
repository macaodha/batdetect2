import os
from pathlib import Path
from typing import Callable, Dict, NamedTuple, Optional, Sequence, Union

import torch
import xarray as xr
from soundevent import data
from torch.utils.data import Dataset

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
        transform: Optional[Callable[[xr.Dataset], xr.Dataset]] = None,
    ):
        self.filenames = filenames
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx) -> TrainExample:
        data = self.load(idx)
        return TrainExample(
            spec=data["spectrogram"],
            detection_heatmap=data["detection"],
            class_heatmap=data["class"],
            size_heatmap=data["size"],
            idx=torch.tensor(idx),
        )

    @classmethod
    def from_directory(cls, directory: PathLike, extension: str = ".nc"):
        return cls(get_files(directory, extension))

    def load(self, idx) -> Dict[str, torch.Tensor]:
        dataset = self.get_dataset(idx)
        return {
            "spectrogram": torch.tensor(
                dataset["spectrogram"].values
            ).unsqueeze(0),
            "detection": torch.tensor(dataset["detection"].values),
            "class": torch.tensor(dataset["class"].values),
            "size": torch.tensor(dataset["size"].values),
        }

    def apply_augmentation(self, dataset: xr.Dataset) -> xr.Dataset:
        if self.transform is not None:
            return self.transform(dataset)

        return dataset

    def get_dataset(self, idx):
        return xr.open_dataset(self.filenames[idx])

    def get_spectrogram(self, idx):
        return xr.open_dataset(self.filenames[idx])["spectrogram"]

    def get_detection_mask(self, idx):
        return xr.open_dataset(self.filenames[idx])["detection"]

    def get_class_mask(self, idx):
        return xr.open_dataset(self.filenames[idx])["class"]

    def get_size_mask(self, idx):
        return xr.open_dataset(self.filenames[idx])["size"]

    def get_clip_annotation(self, idx):
        filename = self.filenames[idx]
        dataset = xr.open_dataset(filename)
        clip_annotation = dataset.attrs["clip_annotation"]
        return data.ClipAnnotation.model_validate_json(clip_annotation)

    def get_preprocessing_configuration(self, idx):
        config = xr.open_dataset(self.filenames[idx]).attrs["configuration"]
        return PreprocessingConfig.model_validate_json(config)
