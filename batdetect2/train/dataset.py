import os
from typing import NamedTuple
from pathlib import Path
from typing import Sequence, Union, Dict
from soundevent import data

from torch.utils.data import Dataset
import torch
import xarray as xr

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
    def __init__(self, filenames: Sequence[PathLike]):
        self.filenames = filenames

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx) -> TrainExample:
        data = self.load(self.filenames[idx])
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

    def load(self, filename: PathLike) -> Dict[str, torch.Tensor]:
        dataset = xr.open_dataset(filename)
        spectrogram = torch.tensor(dataset["spectrogram"].values).unsqueeze(0)
        return {
            "spectrogram": spectrogram,
            "detection": torch.tensor(dataset["detection"].values),
            "class": torch.tensor(dataset["class"].values),
            "size": torch.tensor(dataset["size"].values),
        }

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
