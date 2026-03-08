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

from typing import List

import torch

from batdetect2.models.backbones import (
    UNetBackbone,
    UNetBackboneConfig,
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
from batdetect2.models.detectors import Detector, build_detector
from batdetect2.models.encoder import (
    DEFAULT_ENCODER_CONFIG,
    EncoderConfig,
    build_encoder,
)
from batdetect2.models.heads import BBoxHead, ClassifierHead, DetectorHead
from batdetect2.typing import (
    ClipDetectionsTensor,
    DetectionModel,
    PostprocessorProtocol,
    PreprocessorProtocol,
    TargetProtocol,
)

__all__ = [
    "BBoxHead",
    "UNetBackbone",
    "BackboneConfig",
    "Bottleneck",
    "BottleneckConfig",
    "ClassifierHead",
    "ConvConfig",
    "DEFAULT_DECODER_CONFIG",
    "DEFAULT_ENCODER_CONFIG",
    "DecoderConfig",
    "Detector",
    "DetectorHead",
    "EncoderConfig",
    "FreqCoordConvDownConfig",
    "FreqCoordConvUpConfig",
    "StandardConvDownConfig",
    "StandardConvUpConfig",
    "build_backbone",
    "build_bottleneck",
    "build_decoder",
    "build_encoder",
    "build_detector",
    "load_backbone_config",
    "Model",
    "build_model",
]


class Model(torch.nn.Module):
    detector: DetectionModel
    preprocessor: PreprocessorProtocol
    postprocessor: PostprocessorProtocol
    targets: TargetProtocol

    def __init__(
        self,
        detector: DetectionModel,
        preprocessor: PreprocessorProtocol,
        postprocessor: PostprocessorProtocol,
        targets: TargetProtocol,
    ):
        super().__init__()
        self.detector = detector
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.targets = targets

    def forward(self, wav: torch.Tensor) -> List[ClipDetectionsTensor]:
        spec = self.preprocessor(wav)
        outputs = self.detector(spec)
        return self.postprocessor(outputs)


def build_model(
    config: BackboneConfig | None = None,
    targets: TargetProtocol | None = None,
    preprocessor: PreprocessorProtocol | None = None,
    postprocessor: PostprocessorProtocol | None = None,
):
    from batdetect2.postprocess import build_postprocessor
    from batdetect2.preprocess import build_preprocessor
    from batdetect2.targets import build_targets

    config = config or UNetBackboneConfig()
    targets = targets or build_targets()
    preprocessor = preprocessor or build_preprocessor()
    postprocessor = postprocessor or build_postprocessor(
        preprocessor=preprocessor,
    )
    detector = build_detector(
        num_classes=len(targets.class_names),
        config=config,
    )
    return Model(
        detector=detector,
        postprocessor=postprocessor,
        preprocessor=preprocessor,
        targets=targets,
    )
