"""Prediction heads attached to the backbone feature map.

Each head is a lightweight ``torch.nn.Module`` that applies a 1×1
convolution to map backbone feature channels to one specific type of
output required by BatDetect2:

- ``DetectorHead``: single-channel detection probability heatmap (sigmoid
  activation).
- ``ClassifierHead``: multi-class probability map over the target bat
  species / call types (softmax activation).
- ``BBoxHead``: two-channel map of predicted call duration (time axis) and
  bandwidth (frequency axis) at each location (no activation; raw
  regression output).

All three heads share the same input feature map produced by the backbone,
so they can be evaluated in parallel in a single forward pass.
"""

import torch
from torch import nn

__all__ = [
    "ClassifierHead",
    "DetectorHead",
    "BBoxHead",
]


class ClassifierHead(nn.Module):
    """Prediction head for species / call-type classification probabilities.

    Takes a backbone feature map and produces a probability map where each
    channel corresponds to a target class. Internally the 1×1 convolution
    maps ``in_channels`` to ``num_classes + 1`` logits (the extra channel
    represents a generic background / unknown category); a softmax is then
    applied across the channel dimension and the background channel is
    discarded before returning.

    Parameters
    ----------
    num_classes : int
        Number of target classes (bat species or call types) to predict,
        excluding the background category. Must be positive.
    in_channels : int
        Number of channels in the backbone feature map. Must be positive.

    Attributes
    ----------
    num_classes : int
        Number of specific output classes (background excluded).
    in_channels : int
        Number of input channels expected.
    classifier : nn.Conv2d
        1×1 convolution with ``num_classes + 1`` output channels.
    """

    def __init__(self, num_classes: int, in_channels: int):
        """Initialise the ClassifierHead."""
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
        """Compute per-class probabilities from backbone features.

        Parameters
        ----------
        features : torch.Tensor
            Backbone feature map, shape ``(B, C_in, H, W)``.

        Returns
        -------
        torch.Tensor
            Class probability map, shape ``(B, num_classes, H, W)``.
            Values are softmax probabilities in the range [0, 1] and
            sum to less than 1 per location (the background probability
            is discarded).
        """
        logits = self.classifier(features)
        probs = torch.softmax(logits, dim=1)
        return probs[:, :-1]


class DetectorHead(nn.Module):
    """Prediction head for detection probability (is a call present here?).

    Produces a single-channel heatmap where each value indicates the
    probability ([0, 1]) that a bat call of *any* species is present at
    that time–frequency location in the spectrogram.

    Applies a 1×1 convolution mapping ``in_channels`` → 1, followed by
    sigmoid activation.

    Parameters
    ----------
    in_channels : int
        Number of channels in the backbone feature map. Must be positive.

    Attributes
    ----------
    in_channels : int
        Number of input channels expected.
    detector : nn.Conv2d
        1×1 convolution with a single output channel.
    """

    def __init__(self, in_channels: int):
        """Initialise the DetectorHead."""
        super().__init__()
        self.in_channels = in_channels

        self.detector = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=1,
            kernel_size=1,
            padding=0,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Compute detection probabilities from backbone features.

        Parameters
        ----------
        features : torch.Tensor
            Backbone feature map, shape ``(B, C_in, H, W)``.

        Returns
        -------
        torch.Tensor
            Detection probability heatmap, shape ``(B, 1, H, W)``.
            Values are in the range [0, 1].
        """
        return torch.sigmoid(self.detector(features))


class BBoxHead(nn.Module):
    """Prediction head for bounding box size (duration and bandwidth).

    Produces a two-channel map where channel 0 predicts the scaled duration
    (time-axis extent) and channel 1 predicts the scaled bandwidth
    (frequency-axis extent) of the call at each spectrogram location.

    Applies a 1×1 convolution mapping ``in_channels`` → 2 with no
    activation function (raw regression output). The predicted values are
    in a scaled space and must be converted to real units (seconds and Hz)
    during postprocessing.

    Parameters
    ----------
    in_channels : int
        Number of channels in the backbone feature map. Must be positive.

    Attributes
    ----------
    in_channels : int
        Number of input channels expected.
    bbox : nn.Conv2d
        1×1 convolution with 2 output channels (duration, bandwidth).
    """

    def __init__(self, in_channels: int, num_sizes: int = 2):
        """Initialise the BBoxHead."""
        super().__init__()
        self.in_channels = in_channels
        self.num_sizes = num_sizes

        self.bbox = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.num_sizes,
            kernel_size=1,
            padding=0,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict call duration and bandwidth from backbone features.

        Parameters
        ----------
        features : torch.Tensor
            Backbone feature map, shape ``(B, C_in, H, W)``.

        Returns
        -------
        torch.Tensor
            Size prediction tensor, shape ``(B, 2, H, W)``. Channel 0 is
            the predicted scaled duration; channel 1 is the predicted
            scaled bandwidth. Values must be rescaled to real units during
            postprocessing.
        """
        return self.bbox(features)
