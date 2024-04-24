from abc import ABC, abstractmethod
from typing import NamedTuple

import torch
import torch.nn as nn


class ModelOutput(NamedTuple):
    """Output of the detection model.

    Each of the tensors has a shape of

        `(batch_size, num_channels, spec_height, spec_width)`.

    Where `spec_height` and `spec_width` are the height and width of the
    input spectrograms.

    They contain localised information of:

    1. The probability of a bounding box detection at the given location.
    2. The predicted size of the bounding box at the given location.
    3. The probabilities of each class at the given location before softmax.
    4. Features used to make the predictions at the given location.
    """

    detection_probs: torch.Tensor
    """Tensor with predict detection probabilities."""

    size_preds: torch.Tensor
    """Tensor with predicted bounding box sizes."""

    class_probs: torch.Tensor
    """Tensor with predicted class probabilities."""

    features: torch.Tensor
    """Tensor with intermediate features."""


class EncoderModel(ABC, nn.Module):

    input_height: int
    """Height of the input spectrogram."""

    num_filts: int
    """Dimension of the feature tensor."""

    @abstractmethod
    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """Forward pass of the encoder model."""


class DetectionModel(ABC, nn.Module):
    @abstractmethod
    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """Forward pass of the detection model."""
