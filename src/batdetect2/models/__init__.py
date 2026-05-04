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
from pydantic import Field

from batdetect2.audio.loader import TARGET_SAMPLERATE_HZ
from batdetect2.core.configs import BaseConfig
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
from batdetect2.models.types import DetectionModel
from batdetect2.postprocess.config import PostprocessConfig
from batdetect2.postprocess.types import (
    ClipDetectionsTensor,
    PostprocessorProtocol,
)
from batdetect2.preprocess.config import PreprocessingConfig
from batdetect2.preprocess.types import PreprocessorProtocol
from batdetect2.targets.types import ROIMapperProtocol, TargetProtocol

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
    "ModelConfig",
    "build_model",
    "build_model_with_new_targets",
]


class ModelConfig(BaseConfig):
    """Complete configuration describing a BatDetect2 model.

    Bundles every parameter that defines a model's behaviour: the input
    sample rate, backbone architecture, preprocessing pipeline,
    postprocessing pipeline, and detection targets.

    Attributes
    ----------
    samplerate : int
        Expected input audio sample rate in Hz.  Audio must be resampled
        to this rate before being passed to the model.  Defaults to
        ``TARGET_SAMPLERATE_HZ`` (256 000 Hz).
    architecture : BackboneConfig
        Configuration for the encoder-decoder backbone network.  Defaults
        to ``UNetBackboneConfig()``.
    preprocess : PreprocessingConfig
        Parameters for the audio-to-spectrogram preprocessing pipeline
        (STFT, frequency crop, transforms, resize).  Defaults to
        ``PreprocessingConfig()``.
    postprocess : PostprocessConfig
        Parameters for converting raw model outputs into detections (NMS
        kernel, thresholds, top-k limit).  Defaults to
        ``PostprocessConfig()``.
    """

    samplerate: int = Field(default=TARGET_SAMPLERATE_HZ, gt=0)
    architecture: BackboneConfig = Field(default_factory=UNetBackboneConfig)
    preprocess: PreprocessingConfig = Field(
        default_factory=PreprocessingConfig
    )
    postprocess: PostprocessConfig = Field(default_factory=PostprocessConfig)


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
    class_names : list[str]
        Class names corresponding to the model classification outputs.
    dimension_names : list[str]
        Size-dimension names corresponding to the model size outputs.
    """

    detector: DetectionModel
    preprocessor: PreprocessorProtocol
    postprocessor: PostprocessorProtocol
    class_names: list[str]
    dimension_names: list[str]

    def __init__(
        self,
        detector: DetectionModel,
        preprocessor: PreprocessorProtocol,
        postprocessor: PostprocessorProtocol,
        class_names: list[str],
        dimension_names: list[str],
    ):
        super().__init__()
        self.detector = detector
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.class_names = class_names
        self.dimension_names = dimension_names

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
    config: ModelConfig | dict | None = None,
    class_names: list[str] | None = None,
    dimension_names: list[str] | None = None,
    preprocessor: PreprocessorProtocol | None = None,
    postprocessor: PostprocessorProtocol | None = None,
) -> Model:
    """Build a complete, ready-to-use BatDetect2 model.

    Assembles a ``Model`` instance from a ``ModelConfig`` and optional
    component overrides.  Any component argument left as ``None`` is built
    from the configuration.  Passing a pre-built component overrides the
    corresponding config fields for that component only.

    Parameters
    ----------
    config : ModelConfig, optional
        Full model configuration (samplerate, architecture, preprocessing,
        postprocessing).  Defaults to ``ModelConfig()`` if not provided.
    class_names : list[str], optional
        Class names used to size the classifier head. Required when building
        a new model.
    dimension_names : list[str], optional
        Dimension names used to size the bbox head. Required when building a
        new model.
    preprocessor : PreprocessorProtocol, optional
        Pre-built preprocessor.  If given, overrides
        ``config.preprocess`` and ``config.samplerate`` for the
        preprocessing step.
    postprocessor : PostprocessorProtocol, optional
        Pre-built postprocessor.  If given, overrides
        ``config.postprocess``.  When omitted and a custom
        ``preprocessor`` is supplied, the default postprocessor is built
        using that preprocessor so that frequency and time scaling remain
        consistent.

    Returns
    -------
    Model
        A fully assembled ``Model`` instance ready for inference or
        training.
    """
    from batdetect2.postprocess import build_postprocessor
    from batdetect2.preprocess import build_preprocessor

    config = config or ModelConfig()

    if isinstance(config, dict):
        config = ModelConfig.model_validate(config)

    if class_names is None:
        raise ValueError("class_names must be provided when building a model.")

    if dimension_names is None:
        raise ValueError(
            "dimension_names must be provided when building a model."
        )

    preprocessor = preprocessor or build_preprocessor(
        config=config.preprocess,
        input_samplerate=config.samplerate,
    )
    postprocessor = postprocessor or build_postprocessor(
        preprocessor=preprocessor,
        config=config.postprocess,
    )
    detector = build_detector(
        num_classes=len(class_names),
        num_sizes=len(dimension_names),
        config=config.architecture,
    )
    return Model(
        detector=detector,
        postprocessor=postprocessor,
        preprocessor=preprocessor,
        class_names=class_names,
        dimension_names=dimension_names,
    )


def build_model_with_new_targets(
    model: Model,
    targets: TargetProtocol,
    roi_mapper: ROIMapperProtocol,
) -> Model:
    """Build a new model with a different target set."""
    detector = build_detector(
        num_classes=len(targets.class_names),
        num_sizes=len(roi_mapper.dimension_names),
        backbone=model.detector.backbone,
    )

    return Model(
        detector=detector,
        postprocessor=model.postprocessor,
        preprocessor=model.preprocessor,
        class_names=targets.class_names,
        dimension_names=roi_mapper.dimension_names,
    )
