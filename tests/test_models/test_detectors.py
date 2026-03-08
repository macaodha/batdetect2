import numpy as np
import pytest
import torch

from batdetect2.models.config import BackboneConfig
from batdetect2.models.detectors import Detector, build_detector
from batdetect2.models.heads import BBoxHead, ClassifierHead
from batdetect2.typing.models import ModelOutput


@pytest.fixture
def dummy_spectrogram() -> torch.Tensor:
    """Provides a dummy spectrogram tensor (B, C, H, W)."""
    return torch.randn(2, 1, 256, 128)


def test_build_detector_default():
    """Test building the default detector without a config."""
    num_classes = 5
    model = build_detector(num_classes=num_classes)

    assert isinstance(model, Detector)
    assert model.num_classes == num_classes
    assert isinstance(model.classifier_head, ClassifierHead)
    assert isinstance(model.bbox_head, BBoxHead)


def test_build_detector_custom_config():
    """Test building a detector with a custom BackboneConfig."""
    num_classes = 3
    config = BackboneConfig(in_channels=2, input_height=128)

    model = build_detector(num_classes=num_classes, config=config)

    assert isinstance(model, Detector)
    assert model.backbone.input_height == 128
    assert model.backbone.encoder.in_channels == 2


def test_detector_forward_pass_shapes(dummy_spectrogram):
    """Test that the forward pass produces correctly shaped outputs."""
    num_classes = 4
    # Build model matching the dummy input shape
    config = BackboneConfig(in_channels=1, input_height=256)
    model = build_detector(num_classes=num_classes, config=config)

    # Process the spectrogram through the model
    # PyTorch expects shape (Batch, Channels, Height, Width)
    output = model(dummy_spectrogram)

    # Verify the output is a NamedTuple ModelOutput
    assert isinstance(output, ModelOutput)

    batch_size = dummy_spectrogram.size(0)
    input_height = dummy_spectrogram.size(2)
    input_width = dummy_spectrogram.size(3)

    # Check detection probabilities shape: (B, 1, H, W)
    assert output.detection_probs.shape == (
        batch_size,
        1,
        input_height,
        input_width,
    )

    # Check size predictions shape: (B, 2, H, W)
    assert output.size_preds.shape == (
        batch_size,
        2,
        input_height,
        input_width,
    )

    # Check class probabilities shape: (B, num_classes, H, W)
    assert output.class_probs.shape == (
        batch_size,
        num_classes,
        input_height,
        input_width,
    )

    # Check features shape: (B, out_channels, H, W)
    out_channels = model.backbone.out_channels
    assert output.features.shape == (
        batch_size,
        out_channels,
        input_height,
        input_width,
    )


def test_detector_forward_pass_with_preprocessor(sample_preprocessor):
    """Test the full pipeline from audio to model output."""
    # Generate random audio: 1 second at 256kHz
    samplerate = 256000
    duration = 1.0
    audio = np.random.randn(int(samplerate * duration)).astype(np.float32)

    # Create tensor: (Batch=1, Channels=1, Samples) - Preprocessor expects batched 1D waveforms
    audio_tensor = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0)

    # Preprocess -> Output shape: (Batch=1, Channels=1, Height, Width)
    spec = sample_preprocessor(audio_tensor)

    # Just to be safe, make sure it has 4 dimensions if the preprocessor didn't add batch
    if spec.ndim == 3:
        spec = spec.unsqueeze(0)

    # Build model matching the preprocessor's output shape
    # The preprocessor output is (B, C, H, W) -> spec.shape[1] is C, spec.shape[2] is H
    config = BackboneConfig(
        in_channels=spec.shape[1], input_height=spec.shape[2]
    )
    model = build_detector(num_classes=3, config=config)

    # Process
    output = model(spec)

    # Assert
    assert isinstance(output, ModelOutput)
    assert output.detection_probs.shape[0] == 1  # Batch size 1
    assert output.class_probs.shape[1] == 3  # 3 classes
