"""Tests for spectrogram-level preprocessing transforms.

Covers :mod:`batdetect2.preprocess.spectrogram` — STFT configuration,
frequency cropping, PCEN, spectral mean subtraction, amplitude scaling,
peak normalisation, and resizing.
"""

import torch

from batdetect2.preprocess.spectrogram import (
    PCEN,
    FrequencyConfig,
    FrequencyCrop,
    PcenConfig,
    PeakNormalize,
    PeakNormalizeConfig,
    ResizeConfig,
    ResizeSpec,
    ScaleAmplitude,
    ScaleAmplitudeConfig,
    SpectralMeanSubtraction,
    SpectralMeanSubtractionConfig,
    STFTConfig,
    _spec_params_from_config,
    build_spectrogram_builder,
    build_spectrogram_crop,
    build_spectrogram_resizer,
    build_spectrogram_transform,
)

SAMPLERATE = 256_000


# ---------------------------------------------------------------------------
# STFTConfig / _spec_params_from_config
# ---------------------------------------------------------------------------


def test_stft_config_defaults_give_correct_params():
    """Default STFTConfig at 256 kHz should give n_fft=512, hop_length=128."""
    config = STFTConfig()
    n_fft, hop_length = _spec_params_from_config(config, samplerate=SAMPLERATE)
    assert n_fft == 512
    assert hop_length == 128


def test_stft_config_custom_params():
    """Custom window duration and overlap should produce the expected sizes."""
    config = STFTConfig(window_duration=0.004, window_overlap=0.5)
    n_fft, hop_length = _spec_params_from_config(config, samplerate=SAMPLERATE)
    assert n_fft == 1024
    assert hop_length == 512


# ---------------------------------------------------------------------------
# build_spectrogram_builder
# ---------------------------------------------------------------------------


def test_spectrogram_builder_output_shape():
    """Builder should produce a spectrogram with the expected number of bins."""
    config = STFTConfig()
    n_fft, _ = _spec_params_from_config(config, samplerate=SAMPLERATE)
    expected_freq_bins = n_fft // 2 + 1  # 257 at defaults

    builder = build_spectrogram_builder(config, samplerate=SAMPLERATE)
    n_samples = SAMPLERATE  # 1 second of audio
    wav = torch.randn(n_samples)
    spec = builder(wav)

    assert spec.ndim == 2
    assert spec.shape[0] == expected_freq_bins


def test_spectrogram_builder_output_is_nonnegative():
    """Amplitude spectrogram values should all be >= 0."""
    config = STFTConfig()
    builder = build_spectrogram_builder(config, samplerate=SAMPLERATE)
    wav = torch.randn(SAMPLERATE)
    spec = builder(wav)
    assert (spec >= 0).all()


# ---------------------------------------------------------------------------
# FrequencyCrop
# ---------------------------------------------------------------------------


def test_frequency_crop_output_shape():
    """FrequencyCrop should reduce the number of frequency bins."""
    config = STFTConfig()
    n_fft, _ = _spec_params_from_config(config, samplerate=SAMPLERATE)

    crop = FrequencyCrop(
        samplerate=SAMPLERATE,
        n_fft=n_fft,
        min_freq=10_000,
        max_freq=120_000,
    )
    spec = torch.ones(n_fft // 2 + 1, 100)
    cropped = crop(spec)

    assert cropped.ndim == 2
    # Must be smaller than the full spectrogram
    assert cropped.shape[0] < spec.shape[0]
    assert cropped.shape[1] == 100  # time axis unchanged


def test_frequency_crop_build_from_config():
    """build_spectrogram_crop should return a working FrequencyCrop."""
    freq_config = FrequencyConfig(min_freq=10_000, max_freq=120_000)
    stft_config = STFTConfig()
    crop = build_spectrogram_crop(
        freq_config, stft=stft_config, samplerate=SAMPLERATE
    )
    assert isinstance(crop, FrequencyCrop)


def test_frequency_crop_no_crop_when_bounds_are_none():
    """FrequencyCrop with no bounds should return the full spectrogram."""
    config = STFTConfig()
    n_fft, _ = _spec_params_from_config(config, samplerate=SAMPLERATE)

    crop = FrequencyCrop(samplerate=SAMPLERATE, n_fft=n_fft)
    spec = torch.ones(n_fft // 2 + 1, 100)
    cropped = crop(spec)

    assert cropped.shape == spec.shape


# ---------------------------------------------------------------------------
# PCEN
# ---------------------------------------------------------------------------


def test_pcen_output_shape_preserved():
    """PCEN should not change the shape of the spectrogram."""
    config = PcenConfig()
    pcen = PCEN.from_config(config, samplerate=SAMPLERATE)
    spec = torch.rand(64, 200) + 1e-6  # positive values
    result = pcen(spec)
    assert result.shape == spec.shape


def test_pcen_output_is_finite():
    """PCEN applied to a well-formed spectrogram should produce no NaN or Inf."""
    config = PcenConfig()
    pcen = PCEN.from_config(config, samplerate=SAMPLERATE)
    spec = torch.rand(64, 200) + 1e-6
    result = pcen(spec)
    assert torch.isfinite(result).all()


def test_pcen_output_dtype_matches_input():
    """PCEN should return a tensor with the same dtype as the input."""
    config = PcenConfig()
    pcen = PCEN.from_config(config, samplerate=SAMPLERATE)
    spec = torch.rand(64, 200, dtype=torch.float32)
    result = pcen(spec)
    assert result.dtype == spec.dtype


# ---------------------------------------------------------------------------
# SpectralMeanSubtraction
# ---------------------------------------------------------------------------


def test_spectral_mean_subtraction_output_nonnegative():
    """SpectralMeanSubtraction clamps output to >= 0."""
    module = SpectralMeanSubtraction()
    spec = torch.rand(64, 200)
    result = module(spec)
    assert (result >= 0).all()


def test_spectral_mean_subtraction_shape_preserved():
    module = SpectralMeanSubtraction()
    spec = torch.rand(64, 200)
    result = module(spec)
    assert result.shape == spec.shape


def test_spectral_mean_subtraction_reduces_time_mean():
    """After subtraction the time-axis mean per bin should be <= 0 (pre-clamp)."""
    module = SpectralMeanSubtraction()
    # Constant spectrogram: mean subtraction should produce all zeros before clamp
    spec = torch.ones(32, 100) * 3.0
    result = module(spec)
    assert (result == 0).all()


def test_spectral_mean_subtraction_from_config():
    config = SpectralMeanSubtractionConfig()
    module = SpectralMeanSubtraction.from_config(config, samplerate=SAMPLERATE)
    assert isinstance(module, SpectralMeanSubtraction)


# ---------------------------------------------------------------------------
# PeakNormalize (spectrogram-level)
# ---------------------------------------------------------------------------


def test_peak_normalize_spec_max_is_one():
    """PeakNormalize should scale the spectrogram peak to 1."""
    module = PeakNormalize()
    spec = torch.rand(64, 200) * 5.0
    result = module(spec)
    assert abs(result.abs().max().item() - 1.0) < 1e-6


def test_peak_normalize_spec_handles_zero():
    """PeakNormalize on a zero spectrogram should not raise."""
    module = PeakNormalize()
    spec = torch.zeros(64, 200)
    result = module(spec)
    assert (result == 0).all()


def test_peak_normalize_from_config():
    config = PeakNormalizeConfig()
    module = PeakNormalize.from_config(config, samplerate=SAMPLERATE)
    assert isinstance(module, PeakNormalize)


# ---------------------------------------------------------------------------
# ScaleAmplitude
# ---------------------------------------------------------------------------


def test_scale_amplitude_db_output_is_finite():
    """AmplitudeToDB scaling should produce finite values for positive input."""
    module = ScaleAmplitude(scale="db")
    spec = torch.rand(64, 200) + 1e-4
    result = module(spec)
    assert torch.isfinite(result).all()


def test_scale_amplitude_power_output_equals_square():
    """ScaleAmplitude('power') should square every element."""
    module = ScaleAmplitude(scale="power")
    spec = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
    result = module(spec)
    expected = spec**2
    assert torch.allclose(result, expected)


def test_scale_amplitude_from_config():
    config = ScaleAmplitudeConfig(scale="db")
    module = ScaleAmplitude.from_config(config, samplerate=SAMPLERATE)
    assert isinstance(module, ScaleAmplitude)
    assert module.scale == "db"


# ---------------------------------------------------------------------------
# ResizeSpec
# ---------------------------------------------------------------------------


def test_resize_spec_output_shape():
    """ResizeSpec should produce the target height and scaled width."""
    module = ResizeSpec(height=64, time_factor=0.5)
    spec = torch.rand(1, 128, 200)  # (batch, freq, time)
    result = module(spec)
    assert result.shape == (1, 64, 100)


def test_resize_spec_2d_input():
    """ResizeSpec should handle 2-D input (no batch or channel dimensions)."""
    module = ResizeSpec(height=64, time_factor=0.5)
    spec = torch.rand(128, 200)
    result = module(spec)
    assert result.shape == (64, 100)


def test_resize_spec_output_is_finite():
    module = ResizeSpec(height=128, time_factor=0.5)
    spec = torch.rand(128, 200)
    result = module(spec)
    assert torch.isfinite(result).all()


def test_resize_spec_from_config():
    config = ResizeConfig(height=64, resize_factor=0.25)
    module = build_spectrogram_resizer(config)
    assert isinstance(module, ResizeSpec)
    assert module.height == 64
    assert module.time_factor == 0.25


# ---------------------------------------------------------------------------
# build_spectrogram_transform dispatch
# ---------------------------------------------------------------------------


def test_build_spectrogram_transform_pcen():
    config = PcenConfig()
    module = build_spectrogram_transform(config, samplerate=SAMPLERATE)
    assert isinstance(module, PCEN)


def test_build_spectrogram_transform_spectral_mean_subtraction():
    config = SpectralMeanSubtractionConfig()
    module = build_spectrogram_transform(config, samplerate=SAMPLERATE)
    assert isinstance(module, SpectralMeanSubtraction)


def test_build_spectrogram_transform_scale_amplitude():
    config = ScaleAmplitudeConfig(scale="db")
    module = build_spectrogram_transform(config, samplerate=SAMPLERATE)
    assert isinstance(module, ScaleAmplitude)


def test_build_spectrogram_transform_peak_normalize():
    config = PeakNormalizeConfig()
    module = build_spectrogram_transform(config, samplerate=SAMPLERATE)
    assert isinstance(module, PeakNormalize)
