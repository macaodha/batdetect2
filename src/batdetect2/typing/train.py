from typing import Callable, NamedTuple, Protocol, Tuple

import torch
from soundevent import data

from batdetect2.typing.models import ModelOutput

__all__ = [
    "Augmentation",
    "ClipLabeller",
    "ClipperProtocol",
    "Heatmaps",
    "LossProtocol",
    "Losses",
    "TrainExample",
]


class Heatmaps(NamedTuple):
    """Structure holding the generated heatmap targets."""

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
"""Type alias for the final clip labelling function.

This function takes the complete annotations for a clip and the corresponding
spectrogram, applies all configured filtering, transformation, and encoding
steps, and returns the final `Heatmaps` used for model training.
"""


Augmentation = Callable[
    [torch.Tensor, data.ClipAnnotation],
    Tuple[torch.Tensor, data.ClipAnnotation],
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
    """Structure to hold the computed loss values.

    Allows returning individual loss components along with the total weighted
    loss for monitoring and analysis during training.

    Attributes
    ----------
    detection : torch.Tensor
        Scalar tensor representing the calculated detection loss component
        (before weighting).
    size : torch.Tensor
        Scalar tensor representing the calculated size regression loss component
        (before weighting).
    classification : torch.Tensor
        Scalar tensor representing the calculated classification loss component
        (before weighting).
    total : torch.Tensor
        Scalar tensor representing the final combined loss, computed as the
        weighted sum of the detection, size, and classification components.
        This is the value typically used for backpropagation.
    """

    detection: torch.Tensor
    size: torch.Tensor
    classification: torch.Tensor
    total: torch.Tensor


class LossProtocol(Protocol):
    def __call__(self, pred: ModelOutput, gt: TrainExample) -> Losses: ...


class ClipperProtocol(Protocol):
    def __call__(
        self,
        clip_annotation: data.ClipAnnotation,
    ) -> data.ClipAnnotation: ...

    def get_subclip(self, clip: data.Clip) -> data.Clip: ...
