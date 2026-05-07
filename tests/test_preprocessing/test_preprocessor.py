"""Integration and unit tests for the Preprocessor pipeline.

Covers :mod:`batdetect2.preprocess.preprocessor` — construction,
pipeline output shape/dtype, the ``process_numpy`` helper, attribute
values, output frame rate, and a round-trip YAML → config → build test.
"""

import pathlib

import numpy as np
import torch

from batdetect2.preprocess.audio import FixDurationConfig
from batdetect2.preprocess.config import PreprocessingConfig
from batdetect2.preprocess.preprocessor import (
    Preprocessor,
    build_preprocessor,
    compute_output_samplerate,
)
from batdetect2.preprocess.spectrogram import (
    FrequencyConfig,
    PcenConfig,
    ResizeConfig,
    SpectralMeanSubtractionConfig,
    STFTConfig,
)

SAMPLERATE = 256_000
CLIP_SAMPLES = int(SAMPLERATE * 0.256)


def make_sine_wav(
    samplerate: int = SAMPLERATE,
    duration: float = 0.256,
    freq: float = 40_000.0,
) -> torch.Tensor:
    """Return a single-channel sine-wave tensor."""
    t = torch.linspace(0.0, duration, int(samplerate * duration))
    return torch.sin(2 * torch.pi * freq * t)


def test_build_preprocessor_returns_protocol():
    """build_preprocessor should return a Preprocessor instance."""
    preprocessor = build_preprocessor()
    assert isinstance(preprocessor, Preprocessor)


def test_build_preprocessor_with_default_config():
    """build_preprocessor() with no arguments should not raise."""
    preprocessor = build_preprocessor(input_samplerate=SAMPLERATE)
    assert preprocessor is not None


def test_build_preprocessor_with_explicit_config():
    config = PreprocessingConfig(
        stft=STFTConfig(window_duration=0.002, window_overlap=0.75),
        frequencies=FrequencyConfig(min_freq=10_000, max_freq=120_000),
        size=ResizeConfig(height=128, resize_factor=0.5),
        spectrogram_transforms=[PcenConfig(), SpectralMeanSubtractionConfig()],
    )
    preprocessor = build_preprocessor(config, input_samplerate=SAMPLERATE)
    assert isinstance(preprocessor, Preprocessor)


def test_preprocessor_output_is_2d():
    """The preprocessor output should be a 2-D tensor (freq_bins × time_frames)."""
    preprocessor = build_preprocessor(input_samplerate=SAMPLERATE)
    wav = make_sine_wav()
    result = preprocessor(wav)
    assert result.ndim == 2


def test_preprocessor_output_height_matches_config():
    """Output height should match the ResizeConfig.height setting."""
    config = PreprocessingConfig(
        size=ResizeConfig(height=64, resize_factor=0.5)
    )
    preprocessor = build_preprocessor(config, input_samplerate=SAMPLERATE)
    wav = make_sine_wav()
    result = preprocessor(wav)
    assert result.shape[0] == 64


def test_preprocessor_output_dtype_float32():
    """Output tensor should be float32."""
    preprocessor = build_preprocessor(input_samplerate=SAMPLERATE)
    wav = make_sine_wav()
    result = preprocessor(wav)
    assert result.dtype == torch.float32


def test_preprocessor_output_is_finite():
    """Output spectrogram should contain no NaN or Inf values."""
    preprocessor = build_preprocessor(input_samplerate=SAMPLERATE)
    wav = make_sine_wav()
    result = preprocessor(wav)
    assert torch.isfinite(result).all()


def test_preprocessor_process_numpy_accepts_ndarray():
    """process_numpy should accept a NumPy array and return a NumPy array."""
    preprocessor = build_preprocessor(input_samplerate=SAMPLERATE)
    wav_np = make_sine_wav().numpy()
    result = preprocessor.process_numpy(wav_np)
    assert isinstance(result, np.ndarray)


def test_preprocessor_process_numpy_matches_forward():
    """process_numpy and forward should give numerically identical results."""
    preprocessor = build_preprocessor(input_samplerate=SAMPLERATE)
    wav = make_sine_wav()
    result_pt = preprocessor(wav).numpy()
    result_np = preprocessor.process_numpy(wav.numpy())
    np.testing.assert_array_almost_equal(result_pt, result_np)


def test_preprocessor_min_max_freq_attributes():
    """min_freq and max_freq should match the FrequencyConfig values."""
    config = PreprocessingConfig(
        frequencies=FrequencyConfig(min_freq=15_000, max_freq=100_000)
    )
    preprocessor = build_preprocessor(config, input_samplerate=SAMPLERATE)
    assert preprocessor.min_freq == 15_000
    assert preprocessor.max_freq == 100_000


def test_preprocessor_input_samplerate_attribute():
    preprocessor = build_preprocessor(input_samplerate=SAMPLERATE)
    assert preprocessor.input_samplerate == SAMPLERATE


def test_compute_output_samplerate_defaults():
    """At default settings, output_samplerate should equal 1000 fps."""
    config = PreprocessingConfig()
    rate = compute_output_samplerate(config, input_samplerate=SAMPLERATE)
    assert abs(rate - 1000.0) < 1e-6


def test_preprocessor_output_samplerate_attribute_matches_compute():
    config = PreprocessingConfig()
    preprocessor = build_preprocessor(config, input_samplerate=SAMPLERATE)
    expected = compute_output_samplerate(config, input_samplerate=SAMPLERATE)
    assert abs(preprocessor.output_samplerate - expected) < 1e-6


def test_generate_spectrogram_shape():
    """generate_spectrogram should return the full STFT without crop or resize."""
    config = PreprocessingConfig()
    preprocessor = build_preprocessor(config, input_samplerate=SAMPLERATE)
    wav = make_sine_wav()
    spec = preprocessor.generate_spectrogram(wav)
    # Full STFT: n_fft//2 + 1 = 257 bins at defaults
    assert spec.shape[0] == 257


def test_generate_spectrogram_larger_than_forward():
    """Raw spectrogram should have more frequency bins than the processed output."""
    preprocessor = build_preprocessor(input_samplerate=SAMPLERATE)
    wav = make_sine_wav()
    raw = preprocessor.generate_spectrogram(wav)
    processed = preprocessor(wav)
    assert raw.shape[0] > processed.shape[0]


def test_preprocessor_with_fix_duration_audio_transform():
    """A FixDuration audio transform should produce consistent output shapes."""
    config = PreprocessingConfig(
        audio_transforms=[FixDurationConfig(duration=0.256)],
    )
    preprocessor = build_preprocessor(config, input_samplerate=SAMPLERATE)
    for n_samples in [CLIP_SAMPLES - 1000, CLIP_SAMPLES, CLIP_SAMPLES + 1000]:
        wav = torch.randn(n_samples)
        result = preprocessor(wav)
        assert result.ndim == 2


def test_preprocessor_yaml_roundtrip(tmp_path: pathlib.Path):
    """PreprocessingConfig serialised to YAML and reloaded should produce
    a functionally identical preprocessor."""
    config = PreprocessingConfig(
        stft=STFTConfig(window_duration=0.002, window_overlap=0.75),
        frequencies=FrequencyConfig(min_freq=10_000, max_freq=120_000),
        size=ResizeConfig(height=128, resize_factor=0.5),
    )

    yaml_path = tmp_path / "preprocess_config.yaml"
    yaml_path.write_text(config.to_yaml_string())

    loaded_config = PreprocessingConfig.load(yaml_path)

    preprocessor = build_preprocessor(
        loaded_config, input_samplerate=SAMPLERATE
    )
    wav = make_sine_wav()
    result = preprocessor(wav)

    assert result.ndim == 2
    assert result.shape[0] == 128
    assert torch.isfinite(result).all()
