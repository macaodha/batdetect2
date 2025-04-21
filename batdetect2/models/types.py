"""Defines shared interfaces (ABCs) and data structures for models.

This module centralizes the definitions of core data structures, like the
standard model output container (`ModelOutput`), and establishes abstract base
classes (ABCs) using `abc.ABC` and `torch.nn.Module`. These define contracts
for fundamental model components, ensuring modularity and consistent
interaction within the `batdetect2.models` package.

Key components:
- `ModelOutput`: Standard structure for outputs from detection models.
- `BackboneModel`: Generic interface for any feature extraction backbone.
- `EncoderDecoderModel`: Specialized interface for backbones with distinct
  encoder-decoder stages (e.g., U-Net), providing access to intermediate
  features.
- `DetectionModel`: Interface for the complete end-to-end detection model.
"""

from abc import ABC, abstractmethod
from typing import NamedTuple

import torch
import torch.nn as nn

__all__ = [
    "ModelOutput",
    "BackboneModel",
    "DetectionModel",
]


class ModelOutput(NamedTuple):
    """Standard container for the outputs of a BatDetect2 detection model.

    This structure groups the different prediction tensors produced by the
    model for a batch of input spectrograms. All tensors typically share the
    same spatial dimensions (height H, width W) corresponding to the model's
    output resolution, and the same batch size (N).

    Attributes
    ----------
    detection_probs : torch.Tensor
        Tensor containing the probability of sound event presence at each
        location in the output grid.
        Shape: `(N, 1, H, W)`
    size_preds : torch.Tensor
        Tensor containing the predicted size dimensions
        (e.g., width and height) for a potential bounding box at each location.
        Shape: `(N, 2, H, W)` (Channel 0 typically width, Channel 1 height)
    class_probs : torch.Tensor
        Tensor containing the predicted probabilities (or logits, depending on
        the final activation) for each target class at each location.
        The number of channels corresponds to the number of specific classes
        defined in the `Targets` configuration.
        Shape: `(N, num_classes, H, W)`
    features : torch.Tensor
        Tensor containing features extracted by the model's backbone. These
        might be used for downstream tasks or analysis. The number of channels
        depends on the specific model architecture.
        Shape: `(N, num_features, H, W)`
    """

    detection_probs: torch.Tensor
    size_preds: torch.Tensor
    class_probs: torch.Tensor
    features: torch.Tensor


class BackboneModel(ABC, nn.Module):
    """Abstract Base Class for generic feature extraction backbone models.

    Defines the minimal interface for a feature extractor network within a
    BatDetect2 model. Its primary role is to process an input spectrogram
    tensor and produce a spatially rich feature map tensor, which is then
    typically consumed by separate prediction heads (for detection,
    classification, size).

    This base class is agnostic to the specific internal architecture (e.g.,
    it could be a simple CNN, a U-Net, a Transformer, etc.). Concrete
    implementations must inherit from this class and `torch.nn.Module`,
    implement the `forward` method, and define the required attributes.

    Attributes
    ----------
    input_height : int
        Expected height (number of frequency bins) of the input spectrogram
        tensor that the backbone is designed to process.
    out_channels : int
        Number of channels in the final feature map tensor produced by the
        backbone's `forward` method.
    """

    input_height: int
    """Expected input spectrogram height (frequency bins)."""

    out_channels: int
    """Number of output channels in the final feature map."""

    @abstractmethod
    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """Perform the forward pass to extract features from the spectrogram.

        Parameters
        ----------
        spec : torch.Tensor
            Input spectrogram tensor, typically with shape
            `(batch_size, 1, frequency_bins, time_bins)`.
            `frequency_bins` should match `self.input_height`.

        Returns
        -------
        torch.Tensor
            Output feature map tensor, typically with shape
            `(batch_size, self.out_channels, output_height, output_width)`.
            The spatial dimensions (`output_height`, `output_width`) depend
            on the specific backbone architecture (e.g., they might match the
            input or be downsampled).
        """
        raise NotImplementedError


class EncoderDecoderModel(BackboneModel):
    """Abstract Base Class for Encoder-Decoder style backbone models.

    This class specializes `BackboneModel` for architectures that have distinct
    encoder stages (downsampling path), a bottleneck, and decoder stages
    (upsampling path).

    It provides separate abstract methods for the `encode` and `decode` steps,
    allowing access to the intermediate "bottleneck" features produced by the
    encoder. This can be useful for tasks like transfer learning or specialized
    analyses.

    Attributes
    ----------
    input_height : int
        (Inherited from BackboneModel) Expected input spectrogram height.
    out_channels : int
        (Inherited from BackboneModel) Number of output channels in the final
        feature map produced by the decoder/forward pass.
    bottleneck_channels : int
        Number of channels in the feature map produced by the encoder at its
        deepest point (the bottleneck), before the decoder starts.
    """

    bottleneck_channels: int
    """Number of channels at the encoder's bottleneck."""

    @abstractmethod
    def encode(self, spec: torch.Tensor) -> torch.Tensor:
        """Process the input spectrogram through the encoder part.

        Takes the input spectrogram and passes it through the downsampling path
        of the network up to the bottleneck layer.

        Parameters
        ----------
        spec : torch.Tensor
            Input spectrogram tensor, typically with shape
            `(batch_size, 1, frequency_bins, time_bins)`.

        Returns
        -------
        torch.Tensor
            The encoded feature map from the bottleneck layer, typically with
            shape `(batch_size, self.bottleneck_channels, bottleneck_height,
            bottleneck_width)`. The spatial dimensions are usually downsampled
            relative to the input.
        """
        ...

    @abstractmethod
    def decode(self, encoded: torch.Tensor) -> torch.Tensor:
        """Process the bottleneck features through the decoder part.

        Takes the encoded feature map from the bottleneck and passes it through
        the upsampling path (potentially using skip connections from the
        encoder) to produce the final output feature map.

        Parameters
        ----------
        encoded : torch.Tensor
            The bottleneck feature map tensor produced by the `encode` method.

        Returns
        -------
        torch.Tensor
            The final output feature map tensor, typically with shape
            `(batch_size, self.out_channels, output_height, output_width)`.
            This should match the output shape of the `forward` method.
        """
        ...


class DetectionModel(ABC, nn.Module):
    """Abstract Base Class for complete BatDetect2 detection models.

    Defines the interface for the overall model that takes an input spectrogram
    and produces all necessary outputs for detection, classification, and size
    prediction, packaged within a `ModelOutput` object.

    Concrete implementations typically combine a `BackboneModel` for feature
    extraction with specific prediction heads for each output type. They must
    inherit from this class and `torch.nn.Module`, and implement the `forward`
    method.
    """

    @abstractmethod
    def forward(self, spec: torch.Tensor) -> ModelOutput:
        """Perform the forward pass of the full detection model.

        Processes the input spectrogram through the backbone and prediction
        heads to generate all required output tensors.

        Parameters
        ----------
        spec : torch.Tensor
            Input spectrogram tensor, typically with shape
            `(batch_size, 1, frequency_bins, time_bins)`.

        Returns
        -------
        ModelOutput
            A NamedTuple containing the prediction tensors: `detection_probs`,
            `size_preds`, `class_probs`, and `features`.
        """
