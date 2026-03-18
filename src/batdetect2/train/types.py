from collections.abc import Callable
from typing import TYPE_CHECKING, NamedTuple, Protocol

import torch
from soundevent import data

if TYPE_CHECKING:
    from batdetect2.models.types import ModelOutput

__all__ = [
    "Augmentation",
    "ClipLabeller",
    "Heatmaps",
    "Losses",
    "LossProtocol",
    "TrainExample",
]


class Heatmaps(NamedTuple):
    detection: torch.Tensor
    classes: torch.Tensor
    size: torch.Tensor


class PreprocessedExample(NamedTuple):
    audio: torch.Tensor
    spectrogram: torch.Tensor
    detection_heatmap: torch.Tensor
    class_heatmap: torch.Tensor
    size_heatmap: torch.Tensor

    def copy(self):
        return PreprocessedExample(
            audio=self.audio.clone(),
            spectrogram=self.spectrogram.clone(),
            detection_heatmap=self.detection_heatmap.clone(),
            size_heatmap=self.size_heatmap.clone(),
            class_heatmap=self.class_heatmap.clone(),
        )


ClipLabeller = Callable[[data.ClipAnnotation, torch.Tensor], Heatmaps]


Augmentation = Callable[
    [torch.Tensor, data.ClipAnnotation],
    tuple[torch.Tensor, data.ClipAnnotation],
]


class TrainExample(NamedTuple):
    spec: torch.Tensor
    detection_heatmap: torch.Tensor
    class_heatmap: torch.Tensor
    size_heatmap: torch.Tensor
    idx: torch.Tensor
    start_time: torch.Tensor
    end_time: torch.Tensor


class Losses(NamedTuple):
    detection: torch.Tensor
    size: torch.Tensor
    classification: torch.Tensor
    total: torch.Tensor


class LossProtocol(Protocol):
    def __call__(self, pred: "ModelOutput", gt: TrainExample) -> Losses: ...
