"""Defines and builds the neural network models used in BatDetect2.

This package (`batdetect2.models`) contains the PyTorch implementations of the
deep neural network architectures used for detecting and classifying bat calls
from spectrograms. It provides modular components and configuration-driven
assembly, allowing for experimentation and use of different architectural
variants.

Key Submodules:
- `.types`: Defines core data structures (`ModelOutput`) and abstract base
  classes (`BackboneModel`, `DetectionModel`) establishing interfaces.
- `.blocks`: Provides reusable neural network building blocks.
- `.encoder`: Defines and builds the downsampling path (encoder) of the network.
- `.bottleneck`: Defines and builds the central bottleneck component.
- `.decoder`: Defines and builds the upsampling path (decoder) of the network.
- `.backbone`: Assembles the encoder, bottleneck, and decoder into a complete
  feature extraction backbone (e.g., a U-Net like structure).
- `.heads`: Defines simple prediction heads (detection, classification, size)
  that attach to the backbone features.
- `.detectors`: Assembles the backbone and prediction heads into the final,
  end-to-end `Detector` model.

This module re-exports the most important classes, configurations, and builder
functions from these submodules for convenient access. The primary entry point
for creating a standard BatDetect2 model instance is the `build_model` function
provided here.
"""

from typing import Optional

from batdetect2.models.backbones import (
    Backbone,
    BackboneConfig,
    build_backbone,
    load_backbone_config,
)
from batdetect2.models.blocks import (
    ConvConfig,
    FreqCoordConvDownConfig,
    FreqCoordConvUpConfig,
    StandardConvDownConfig,
    StandardConvUpConfig,
)
from batdetect2.models.bottleneck import (
    Bottleneck,
    BottleneckConfig,
    build_bottleneck,
)
from batdetect2.models.decoder import (
    DEFAULT_DECODER_CONFIG,
    DecoderConfig,
    build_decoder,
)
from batdetect2.models.detectors import (
    Detector,
    build_detector,
)
from batdetect2.models.encoder import (
    DEFAULT_ENCODER_CONFIG,
    EncoderConfig,
    build_encoder,
)
from batdetect2.models.heads import BBoxHead, ClassifierHead, DetectorHead
from batdetect2.models.types import BackboneModel, DetectionModel, ModelOutput

__all__ = [
    "BBoxHead",
    "Backbone",
    "BackboneConfig",
    "BackboneModel",
    "BackboneModel",
    "Bottleneck",
    "BottleneckConfig",
    "ClassifierHead",
    "ConvConfig",
    "DEFAULT_DECODER_CONFIG",
    "DEFAULT_ENCODER_CONFIG",
    "DecoderConfig",
    "DetectionModel",
    "Detector",
    "DetectorHead",
    "EncoderConfig",
    "FreqCoordConvDownConfig",
    "FreqCoordConvUpConfig",
    "ModelOutput",
    "StandardConvDownConfig",
    "StandardConvUpConfig",
    "build_backbone",
    "build_bottleneck",
    "build_decoder",
    "build_detector",
    "build_encoder",
    "build_model",
    "load_backbone_config",
]


def build_model(
    num_classes: int,
    config: Optional[BackboneConfig] = None,
) -> DetectionModel:
    """Build the complete BatDetect2 detection model.

    This high-level factory function constructs the standard BatDetect2 model
    architecture. It first builds the feature extraction backbone (typically an
    encoder-bottleneck-decoder structure) based on the provided
    `BackboneConfig` (or defaults if None), and then attaches the standard
    prediction heads (`DetectorHead`, `ClassifierHead`, `BBoxHead`) using the
    `build_detector` function.

    Parameters
    ----------
    num_classes : int
        The number of specific target classes the model should predict
        (required for the `ClassifierHead`). Must be positive.
    config : BackboneConfig, optional
        Configuration object specifying the architecture of the backbone
        (encoder, bottleneck, decoder). If None, default configurations defined
        within the respective builder functions (`build_encoder`, etc.) will be
        used to construct a default backbone architecture.

    Returns
    -------
    DetectionModel
        An initialized `Detector` model instance.

    Raises
    ------
    ValueError
        If `num_classes` is not positive, or if errors occur during the
        construction of the backbone or detector components (e.g., incompatible
        configurations, invalid parameters).
    """
    backbone = build_backbone(config or BackboneConfig())
    return build_detector(num_classes, backbone)
