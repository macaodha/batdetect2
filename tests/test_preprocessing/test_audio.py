"""Tests for audio-level preprocessing transforms.

Covers :mod:`batdetect2.preprocess.audio` and the shared helper functions
in :mod:`batdetect2.preprocess.common`.
"""

import pathlib
import uuid

import numpy as np
import pytest
import soundfile as sf
import torch
from soundevent import data

from batdetect2.audio import AudioConfig
from batdetect2.preprocess.audio import (
    CenterAudio,
    CenterAudioConfig,
    FixDuration,
    FixDurationConfig,
    ScaleAudio,
    ScaleAudioConfig,
    build_audio_transform,
)
from batdetect2.preprocess.common import center_tensor, peak_normalize

SAMPLERATE = 256_000


def create_dummy_wave(
    samplerate: int,
    duration: float,
    num_channels: int = 1,
    freq: float = 440.0,
    amplitude: float = 0.5,
    dtype: type = np.float32,
) -> np.ndarray:
    """Generate a simple sine-wave waveform as a NumPy array."""
    t = np.linspace(
        0.0, duration, int(samplerate * duration), endpoint=False, dtype=dtype
    )
    wave = amplitude * np.sin(2 * np.pi * freq * t)
    if num_channels > 1:
        wave = np.stack([wave] * num_channels, axis=0)
    return wave.astype(dtype)


@pytest.fixture
def dummy_wav_path(tmp_path: pathlib.Path) -> pathlib.Path:
    """Create a dummy 2-channel WAV file and return its path."""
    samplerate = 48000
    duration = 2.0
    num_channels = 2
    wave_data = create_dummy_wave(samplerate, duration, num_channels)
    file_path = tmp_path / f"{uuid.uuid4()}.wav"
    sf.write(file_path, wave_data.T, samplerate, format="WAV", subtype="FLOAT")
    return file_path


@pytest.fixture
def dummy_recording(dummy_wav_path: pathlib.Path) -> data.Recording:
    """Create a Recording object pointing to the dummy WAV file."""
    return data.Recording.from_file(dummy_wav_path)


@pytest.fixture
def dummy_clip(dummy_recording: data.Recording) -> data.Clip:
    """Create a Clip object from the dummy recording."""
    return data.Clip(
        recording=dummy_recording,
        start_time=0.5,
        end_time=1.5,
    )


@pytest.fixture
def default_audio_config() -> AudioConfig:
    return AudioConfig()


def test_center_tensor_zero_mean():
    """Output tensor should have a mean very close to zero."""
    wav = torch.tensor([1.0, 2.0, 3.0, 4.0])
    result = center_tensor(wav)
    assert result.mean().abs().item() < 1e-5


def test_center_tensor_preserves_shape():
    wav = torch.randn(3, 1000)
    result = center_tensor(wav)
    assert result.shape == wav.shape


def test_peak_normalize_max_is_one():
    """After peak normalisation, the maximum absolute value should be 1."""
    wav = torch.tensor([0.1, -0.4, 0.2, 0.8, -0.3])
    result = peak_normalize(wav)
    assert abs(result.abs().max().item() - 1.0) < 1e-6


def test_peak_normalize_zero_tensor_unchanged():
    """A zero tensor should be returned unchanged (no division by zero)."""
    wav = torch.zeros(100)
    result = peak_normalize(wav)
    assert (result == 0).all()


def test_peak_normalize_preserves_shape():
    wav = torch.randn(2, 512)
    result = peak_normalize(wav)
    assert result.shape == wav.shape


def test_center_audio_forward_zero_mean():
    module = CenterAudio()
    wav = torch.tensor([1.0, 3.0, 5.0])
    result = module(wav)
    assert result.mean().abs().item() < 1e-5


def test_center_audio_from_config():
    config = CenterAudioConfig()
    module = CenterAudio.from_config(config, samplerate=SAMPLERATE)
    assert isinstance(module, CenterAudio)


def test_scale_audio_peak_normalises_to_one():
    """ScaleAudio.forward should scale the peak absolute value to 1."""
    module = ScaleAudio()
    wav = torch.tensor([0.0, 0.25, 0.1])
    result = module(wav)
    assert abs(result.abs().max().item() - 1.0) < 1e-6


def test_scale_audio_handles_zero_tensor():
    """ScaleAudio should not raise on a zero tensor."""
    module = ScaleAudio()
    wav = torch.zeros(100)
    result = module(wav)
    assert (result == 0).all()


def test_scale_audio_from_config():
    config = ScaleAudioConfig()
    module = ScaleAudio.from_config(config, samplerate=SAMPLERATE)
    assert isinstance(module, ScaleAudio)


def test_fix_duration_truncates_long_input():
    """Waveform longer than target should be truncated to the target length."""
    target_samples = int(SAMPLERATE * 0.5)
    module = FixDuration(samplerate=SAMPLERATE, duration=0.5)
    wav = torch.randn(target_samples + 1000)
    result = module(wav)
    assert result.shape[-1] == target_samples


def test_fix_duration_pads_short_input():
    """Waveform shorter than target should be zero-padded to the target length."""
    target_samples = int(SAMPLERATE * 0.5)
    module = FixDuration(samplerate=SAMPLERATE, duration=0.5)
    short_wav = torch.randn(target_samples - 100)
    result = module(short_wav)
    assert result.shape[-1] == target_samples
    # Padded region should be zero
    assert (result[target_samples - 100 :] == 0).all()


def test_fix_duration_passthrough_exact_length():
    """Waveform with exactly the right length should be returned unchanged."""
    target_samples = int(SAMPLERATE * 0.5)
    module = FixDuration(samplerate=SAMPLERATE, duration=0.5)
    wav = torch.randn(target_samples)
    result = module(wav)
    assert result.shape[-1] == target_samples
    assert torch.equal(result, wav)


def test_fix_duration_from_config():
    """FixDurationConfig should produce a FixDuration with the correct length."""
    config = FixDurationConfig(duration=0.256)
    module = FixDuration.from_config(config, samplerate=SAMPLERATE)
    assert isinstance(module, FixDuration)
    assert module.length == int(SAMPLERATE * 0.256)


def test_build_audio_transform_center_audio():
    config = CenterAudioConfig()
    module = build_audio_transform(config, samplerate=SAMPLERATE)
    assert isinstance(module, CenterAudio)


def test_build_audio_transform_scale_audio():
    config = ScaleAudioConfig()
    module = build_audio_transform(config, samplerate=SAMPLERATE)
    assert isinstance(module, ScaleAudio)


def test_build_audio_transform_fix_duration():
    config = FixDurationConfig(duration=0.5)
    module = build_audio_transform(config, samplerate=SAMPLERATE)
    assert isinstance(module, FixDuration)
