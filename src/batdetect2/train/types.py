from typing import Callable, NamedTuple, Protocol, Tuple

import torch
import xarray as xr
from soundevent import data

from batdetect2.models import ModelOutput

__all__ = [
    "Heatmaps",
    "ClipLabeller",
    "Augmentation",
    "LossProtocol",
    "TrainExample",
]


class Heatmaps(NamedTuple):
    """Structure holding the generated heatmap targets.

    Attributes
    ----------
    detection : xr.DataArray
        Heatmap indicating the probability of sound event presence. Typically
        smoothed with a Gaussian kernel centered on event reference points.
        Shape matches the input spectrogram. Values normalized [0, 1].
    classes : xr.DataArray
        Heatmap indicating the probability of specific class presence. Has an
        additional 'category' dimension corresponding to the target class
        names. Each category slice is typically smoothed with a Gaussian
        kernel. Values normalized [0, 1] per category.
    size : xr.DataArray
        Heatmap encoding the size (width, height) of detected events. Has an
        additional 'dimension' coordinate ('width', 'height'). Values represent
        scaled dimensions placed at the event reference points.
    """

    detection: xr.DataArray
    classes: xr.DataArray
    size: xr.DataArray


ClipLabeller = Callable[[data.ClipAnnotation, xr.DataArray], Heatmaps]
"""Type alias for the final clip labelling function.

This function takes the complete annotations for a clip and the corresponding
spectrogram, applies all configured filtering, transformation, and encoding
steps, and returns the final `Heatmaps` used for model training.
"""

Augmentation = Callable[[xr.Dataset], xr.Dataset]


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
    def extract_clip(
        self, example: xr.Dataset
    ) -> Tuple[xr.Dataset, float, float]: ...
