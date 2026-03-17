"""Neural network model definitions and builders for BatDetect2.

This package contains the PyTorch implementations of the deep neural network
architectures used to detect and classify bat echolocation calls in
spectrograms. Components are designed to be combined through configuration
objects, making it easy to experiment with different architectures.

Key submodules
--------------
- ``blocks``: Reusable convolutional building blocks (downsampling,
  upsampling, attention, coord-conv variants).
- ``encoder``: The downsampling path; reduces spatial resolution whilst
  extracting increasingly abstract features.
- ``bottleneck``: The central component connecting encoder to decoder;
  optionally applies self-attention along the time axis.
- ``decoder``: The upsampling path; reconstructs high-resolution feature
  maps using bottleneck output and skip connections from the encoder.
- ``backbones``: Assembles encoder, bottleneck, and decoder into a complete
  U-Net-style feature extraction backbone.
- ``heads``: Lightweight 1×1 convolutional heads that produce detection,
  classification, and bounding-box size predictions from backbone features.
- ``detectors``: Combines a backbone with prediction heads into the final
  end-to-end ``Detector`` model.

The primary entry point for building a full, ready-to-use BatDetect2 model
is the ``build_model`` factory function exported from this module.
"""

import torch

from batdetect2.models.backbones import (
    BackboneConfig,
    UNetBackbone,
    UNetBackboneConfig,
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
    """End-to-end BatDetect2 model wrapping preprocessing and postprocessing.

    Combines a preprocessor, a detection model, and a postprocessor into a
    single PyTorch module. Calling ``forward`` on a raw waveform tensor
    returns a list of detection tensors ready for downstream use.

    This class is the top-level object produced by ``build_model``. Most
    users will not need to construct it directly.

    Attributes
    ----------
    detector : DetectionModel
        The neural network that processes spectrograms and produces raw
        detection, classification, and bounding-box outputs.
    preprocessor : PreprocessorProtocol
        Converts a raw waveform tensor into a spectrogram tensor accepted by
        ``detector``.
    postprocessor : PostprocessorProtocol
        Converts the raw ``ModelOutput`` from ``detector`` into a list of
        per-clip detection tensors.
    targets : TargetProtocol
        Describes the set of target classes; used when building heads and
        during training target construction.
    """

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

    def forward(self, wav: torch.Tensor) -> list[ClipDetectionsTensor]:
        """Run the full detection pipeline on a waveform tensor.

        Converts the waveform to a spectrogram, passes it through the
        detector, and postprocesses the raw outputs into detection tensors.

        Parameters
        ----------
        wav : torch.Tensor
            Raw audio waveform tensor. The exact expected shape depends on
            the preprocessor, but is typically ``(batch, samples)`` or
            ``(batch, channels, samples)``.

        Returns
        -------
        list[ClipDetectionsTensor]
            One detection tensor per clip in the batch. Each tensor encodes
            the detected events (locations, class scores, sizes) for that
            clip.
        """
        spec = self.preprocessor(wav)
        outputs = self.detector(spec)
        return self.postprocessor(outputs)


def build_model(
    config: BackboneConfig | None = None,
    targets: TargetProtocol | None = None,
    preprocessor: PreprocessorProtocol | None = None,
    postprocessor: PostprocessorProtocol | None = None,
) -> "Model":
    """Build a complete, ready-to-use BatDetect2 model.

    Assembles a ``Model`` instance from optional configuration and component
    overrides. Any argument left as ``None`` will be replaced by a sensible
    default built with the project's own builder functions.

    Parameters
    ----------
    config : BackboneConfig, optional
        Configuration describing the backbone architecture (encoder,
        bottleneck, decoder). Defaults to ``UNetBackboneConfig()`` if not
        provided.
    targets : TargetProtocol, optional
        Describes the target bat species or call types to detect. Determines
        the number of output classes. Defaults to the standard BatDetect2
        target set.
    preprocessor : PreprocessorProtocol, optional
        Converts raw audio waveforms to spectrograms. Defaults to the
        standard BatDetect2 preprocessor.
    postprocessor : PostprocessorProtocol, optional
        Converts raw model outputs to detection tensors. Defaults to the
        standard BatDetect2 postprocessor. If a custom ``preprocessor`` is
        given without a matching ``postprocessor``, the default postprocessor
        will be built using the provided preprocessor so that frequency and
        time scaling remain consistent.

    Returns
    -------
    Model
        A fully assembled ``Model`` instance ready for inference or training.
    """
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
