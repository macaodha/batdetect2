import pathlib
import uuid

import numpy as np
import pytest
import soundfile as sf
from soundevent import data

from batdetect2.audio import AudioConfig


def create_dummy_wave(
    samplerate: int,
    duration: float,
    num_channels: int = 1,
    freq: float = 440.0,
    amplitude: float = 0.5,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Generates a simple numpy waveform."""
    t = np.linspace(
        0.0, duration, int(samplerate * duration), endpoint=False, dtype=dtype
    )
    wave = amplitude * np.sin(2 * np.pi * freq * t)
    if num_channels > 1:
        wave = np.stack([wave] * num_channels, axis=0)
    return wave.astype(dtype)


@pytest.fixture
def dummy_wav_path(tmp_path: pathlib.Path) -> pathlib.Path:
    """Creates a dummy WAV file and returns its path."""
    samplerate = 48000
    duration = 2.0
    num_channels = 2
    wave_data = create_dummy_wave(samplerate, duration, num_channels)
    file_path = tmp_path / f"{uuid.uuid4()}.wav"
    sf.write(file_path, wave_data.T, samplerate, format="WAV", subtype="FLOAT")
    return file_path


@pytest.fixture
def dummy_recording(dummy_wav_path: pathlib.Path) -> data.Recording:
    """Creates a Recording object pointing to the dummy WAV file."""
    return data.Recording.from_file(dummy_wav_path)


@pytest.fixture
def dummy_clip(dummy_recording: data.Recording) -> data.Clip:
    """Creates a Clip object from the dummy recording."""
    return data.Clip(
        recording=dummy_recording,
        start_time=0.5,
        end_time=1.5,
    )


@pytest.fixture
def default_audio_config() -> AudioConfig:
    return AudioConfig()
