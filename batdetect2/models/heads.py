"""Prediction Head modules for BatDetect2 models.

This module defines simple `torch.nn.Module` subclasses that serve as
prediction heads, typically attached to the output feature map of a backbone
network

Each head is responsible for generating one specific type of output required
by the BatDetect2 task:
- `DetectorHead`: Predicts the probability of sound event presence.
- `ClassifierHead`: Predicts the probability distribution over target classes.
- `BBoxHead`: Predicts the size (width, height) of the sound event's bounding
box.

These heads use 1x1 convolutions to map the backbone feature channels
to the desired number of output channels for each prediction task at each
spatial location, followed by an appropriate activation function (e.g., sigmoid
for detection, softmax for classification, none for size regression).
"""

import torch
from torch import nn

__all__ = [
    "ClassifierHead",
    "DetectorHead",
    "BBoxHead",
]


class ClassifierHead(nn.Module):
    """Prediction head for multi-class classification probabilities.

    Takes an input feature map and produces a probability map where each
    channel corresponds to a specific target class. It uses a 1x1 convolution
    to map input channels to `num_classes + 1` outputs (one for each target
    class plus an assumed background/generic class), applies softmax across the
    channels, and returns the probabilities for the specific target classes
    (excluding the last background/generic channel).

    Parameters
    ----------
    num_classes : int
        The number of specific target classes the model should predict
        (excluding any background or generic category). Must be positive.
    in_channels : int
        Number of channels in the input feature map tensor from the backbone.
        Must be positive.

    Attributes
    ----------
    num_classes : int
        Number of specific output classes.
    in_channels : int
        Number of input channels expected.
    classifier : nn.Conv2d
        The 1x1 convolutional layer used for prediction.
        Output channels = num_classes + 1.

    Raises
    ------
    ValueError
        If `num_classes` or `in_channels` are not positive.
    """

    def __init__(self, num_classes: int, in_channels: int):
        """Initialize the ClassifierHead."""
        super().__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.classifier = nn.Conv2d(
            self.in_channels,
            self.num_classes + 1,
            kernel_size=1,
            padding=0,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Compute class probabilities from input features.

        Parameters
        ----------
        features : torch.Tensor
            Input feature map tensor from the backbone, typically with shape
            `(B, C_in, H, W)`. `C_in` must match `self.in_channels`.

        Returns
        -------
        torch.Tensor
            Class probability map tensor with shape `(B, num_classes, H, W)`.
            Contains probabilities for the specific target classes after
            softmax, excluding the implicit background/generic class channel.
        """
        logits = self.classifier(features)
        probs = torch.softmax(logits, dim=1)
        return probs[:, :-1]


class DetectorHead(nn.Module):
    """Prediction head for sound event detection probability.

    Takes an input feature map and produces a single-channel heatmap where
    each value represents the probability ([0, 1]) of a relevant sound event
    (of any class) being present at that spatial location.

    Uses a 1x1 convolution to map input channels to 1 output channel, followed
    by a sigmoid activation function.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input feature map tensor from the backbone.
        Must be positive.

    Attributes
    ----------
    in_channels : int
        Number of input channels expected.
    detector : nn.Conv2d
        The 1x1 convolutional layer mapping to a single output channel.

    Raises
    ------
    ValueError
        If `in_channels` is not positive.
    """

    def __init__(self, in_channels: int):
        """Initialize the DetectorHead."""
        super().__init__()
        self.in_channels = in_channels

        self.detector = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=1,
            kernel_size=1,
            padding=0,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Compute detection probabilities from input features.

        Parameters
        ----------
        features : torch.Tensor
            Input feature map tensor from the backbone, typically with shape
            `(B, C_in, H, W)`. `C_in` must match `self.in_channels`.

        Returns
        -------
        torch.Tensor
            Detection probability heatmap tensor with shape `(B, 1, H, W)`.
            Values are in the range [0, 1] due to the sigmoid activation.

        Raises
        ------
        RuntimeError
            If input channel count does not match `self.in_channels`.
        """
        return torch.sigmoid(self.detector(features))


class BBoxHead(nn.Module):
    """Prediction head for bounding box size dimensions.

    Takes an input feature map and produces a two-channel map where each
    channel represents a predicted size dimension (typically width/duration and
    height/bandwidth) for a potential sound event at that spatial location.

    Uses a 1x1 convolution to map input channels to 2 output channels. No
    activation function is typically applied, as size prediction is often
    treated as a direct regression task. The output values usually represent
    *scaled* dimensions that need to be un-scaled during postprocessing.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input feature map tensor from the backbone.
        Must be positive.

    Attributes
    ----------
    in_channels : int
        Number of input channels expected.
    bbox : nn.Conv2d
        The 1x1 convolutional layer mapping to 2 output channels
        (width, height).

    Raises
    ------
    ValueError
        If `in_channels` is not positive.
    """

    def __init__(self, in_channels: int):
        """Initialize the BBoxHead."""
        super().__init__()
        self.in_channels = in_channels

        self.bbox = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=2,
            kernel_size=1,
            padding=0,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Compute predicted bounding box dimensions from input features.

        Parameters
        ----------
        features : torch.Tensor
            Input feature map tensor from the backbone, typically with shape
            `(B, C_in, H, W)`. `C_in` must match `self.in_channels`.

        Returns
        -------
        torch.Tensor
            Predicted size tensor with shape `(B, 2, H, W)`. Channel 0 usually
            represents scaled width, Channel 1 scaled height. These values
            need to be un-scaled during postprocessing.
        """
        return self.bbox(features)
