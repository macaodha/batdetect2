"""Handles loading and initial preprocessing of audio waveforms."""

from typing import Annotated, List, Literal, Optional, Union

import numpy as np
import torch
from numpy.typing import DTypeLike
from pydantic import Field
from scipy.signal import resample, resample_poly
from soundevent import audio, data
from soundfile import LibsndfileError

from batdetect2.configs import BaseConfig
from batdetect2.preprocess.common import CenterTensor, PeakNormalize
from batdetect2.typing import AudioLoader

__all__ = [
    "ResampleConfig",
    "AudioConfig",
    "SoundEventAudioLoader",
    "build_audio_loader",
    "load_file_audio",
    "load_recording_audio",
    "load_clip_audio",
    "resample_audio",
    "TARGET_SAMPLERATE_HZ",
    "SCALE_RAW_AUDIO",
    "DEFAULT_DURATION",
]

TARGET_SAMPLERATE_HZ = 256_000
"""Default target sample rate in Hz used if resampling is enabled."""

SCALE_RAW_AUDIO = False
"""Default setting for whether to perform peak normalization."""

DEFAULT_DURATION = None
"""Default setting for target audio duration in seconds."""


class ResampleConfig(BaseConfig):
    """Configuration for audio resampling.

    Attributes
    ----------
    samplerate : int, default=256000
        The target sample rate in Hz to resample the audio to. Must be > 0.
    method : str, default="poly"
        The resampling algorithm to use. Options:
        - "poly": Polyphase resampling using `scipy.signal.resample_poly`.
                  Generally fast.
        - "fourier": Resampling via Fourier method using
                     `scipy.signal.resample`. May handle non-integer
                     resampling factors differently.
    """

    enabled: bool = True
    method: str = "poly"


class SoundEventAudioLoader:
    """Concrete implementation of the `AudioLoader`."""

    def __init__(
        self,
        samplerate: int = TARGET_SAMPLERATE_HZ,
        config: Optional[ResampleConfig] = None,
    ):
        self.samplerate = samplerate
        self.config = config or ResampleConfig()

    def load_file(
        self,
        path: data.PathLike,
        audio_dir: Optional[data.PathLike] = None,
    ) -> np.ndarray:
        """Load and preprocess audio directly from a file path."""
        return load_file_audio(
            path,
            samplerate=self.samplerate,
            config=self.config,
            audio_dir=audio_dir,
        )

    def load_recording(
        self,
        recording: data.Recording,
        audio_dir: Optional[data.PathLike] = None,
    ) -> np.ndarray:
        """Load and preprocess the entire audio for a Recording object."""
        return load_recording_audio(
            recording,
            samplerate=self.samplerate,
            config=self.config,
            audio_dir=audio_dir,
        )

    def load_clip(
        self,
        clip: data.Clip,
        audio_dir: Optional[data.PathLike] = None,
    ) -> np.ndarray:
        """Load and preprocess the audio segment defined by a Clip object."""
        return load_clip_audio(
            clip,
            samplerate=self.samplerate,
            config=self.config,
            audio_dir=audio_dir,
        )


def load_file_audio(
    path: data.PathLike,
    samplerate: Optional[int] = None,
    config: Optional[ResampleConfig] = None,
    audio_dir: Optional[data.PathLike] = None,
    dtype: DTypeLike = np.float32,  # type: ignore
) -> np.ndarray:
    """Load and preprocess audio from a file path using specified config."""
    try:
        recording = data.Recording.from_file(path)
    except LibsndfileError as e:
        raise FileNotFoundError(
            f"Could not load the recording at path: {path}. Error: {e}"
        ) from e

    return load_recording_audio(
        recording,
        samplerate=samplerate,
        config=config,
        dtype=dtype,
        audio_dir=audio_dir,
    )


def load_recording_audio(
    recording: data.Recording,
    samplerate: Optional[int] = None,
    config: Optional[ResampleConfig] = None,
    audio_dir: Optional[data.PathLike] = None,
    dtype: DTypeLike = np.float32,  # type: ignore
) -> np.ndarray:
    """Load and preprocess the entire audio content of a recording using config."""
    clip = data.Clip(
        recording=recording,
        start_time=0,
        end_time=recording.duration,
    )
    return load_clip_audio(
        clip,
        samplerate=samplerate,
        config=config,
        dtype=dtype,
        audio_dir=audio_dir,
    )


def load_clip_audio(
    clip: data.Clip,
    samplerate: Optional[int] = None,
    config: Optional[ResampleConfig] = None,
    audio_dir: Optional[data.PathLike] = None,
    dtype: DTypeLike = np.float32,  # type: ignore
) -> np.ndarray:
    """Load and preprocess a specific audio clip segment based on config."""
    try:
        wav = (
            audio.load_clip(clip, audio_dir=audio_dir)
            .sel(channel=0)
            .astype(dtype)
        )
    except LibsndfileError as e:
        raise FileNotFoundError(
            f"Could not load the recording at path: {clip.recording.path}. "
            f"Error: {e}"
        ) from e

    if not config or not config.enabled or samplerate is None:
        return wav.data.astype(dtype)

    sr = int(1 / wav.time.attrs["step"])
    return resample_audio(
        wav.data,
        sr=sr,
        samplerate=samplerate,
        method=config.method,
    )


def resample_audio(
    wav: np.ndarray,
    sr: int,
    samplerate: int = TARGET_SAMPLERATE_HZ,
    method: str = "poly",
) -> np.ndarray:
    """Resample an audio waveform DataArray to a target sample rate."""
    if sr == samplerate:
        return wav

    if method == "poly":
        return resample_audio_poly(
            wav,
            sr_orig=sr,
            sr_new=samplerate,
        )
    elif method == "fourier":
        return resample_audio_fourier(
            wav,
            sr_orig=sr,
            sr_new=samplerate,
        )
    else:
        raise NotImplementedError(
            f"Resampling method '{method}' not implemented"
        )


def resample_audio_poly(
    array: np.ndarray,
    sr_orig: int,
    sr_new: int,
    axis: int = -1,
) -> np.ndarray:
    """Resample a numpy array using `scipy.signal.resample_poly`.

    This method is often preferred for signals when the ratio of new
    to old sample rates can be expressed as a rational number. It uses
    polyphase filtering.

    Parameters
    ----------
    array : np.ndarray
        The input array to resample.
    sr_orig : int
        The original sample rate in Hz.
    sr_new : int
        The target sample rate in Hz.
    axis : int, default=-1
        The axis of `array` along which to resample.

    Returns
    -------
    np.ndarray
        The array resampled to the target sample rate.

    Raises
    ------
    ValueError
        If sample rates are not positive.
    """
    gcd = np.gcd(sr_orig, sr_new)
    return resample_poly(
        array,
        sr_new // gcd,
        sr_orig // gcd,
        axis=axis,
    )


def resample_audio_fourier(
    array: np.ndarray,
    sr_orig: int,
    sr_new: int,
    axis: int = -1,
) -> np.ndarray:
    """Resample a numpy array using `scipy.signal.resample`.

    This method uses FFTs to resample the signal.

    Parameters
    ----------
    array : np.ndarray
        The input array to resample.
    num : int
        The desired number of samples in the output array along `axis`.
    axis : int, default=-1
        The axis of `array` along which to resample.

    Returns
    -------
    np.ndarray
        The array resampled to have `num` samples along `axis`.

    Raises
    ------
    ValueError
        If `num` is negative.
    """
    ratio = sr_new / sr_orig
    return resample(  # type: ignore
        array,
        int(array.shape[axis] * ratio),
        axis=axis,
    )


class CenterAudioConfig(BaseConfig):
    name: Literal["center_audio"] = "center_audio"


class ScaleAudioConfig(BaseConfig):
    name: Literal["scale_audio"] = "scale_audio"


class FixDurationConfig(BaseConfig):
    name: Literal["fix_duration"] = "fix_duration"
    duration: float = 0.5


class FixDuration(torch.nn.Module):
    def __init__(self, samplerate: int, duration: float):
        super().__init__()
        self.samplerate = samplerate
        self.duration = duration
        self.length = int(samplerate * duration)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        length = wav.shape[-1]

        if length == self.length:
            return wav

        if length > self.length:
            return wav[: self.length]

        return torch.nn.functional.pad(wav, (0, self.length - length))


AudioTransform = Annotated[
    Union[
        FixDurationConfig,
        ScaleAudioConfig,
        CenterAudioConfig,
    ],
    Field(discriminator="name"),
]


class AudioConfig(BaseConfig):
    """Configuration for loading and initial audio preprocessing."""

    samplerate: int = Field(default=TARGET_SAMPLERATE_HZ, gt=0)
    resample: Optional[ResampleConfig] = Field(default_factory=ResampleConfig)
    transforms: List[AudioTransform] = Field(default_factory=list)


def build_audio_loader(
    config: Optional[AudioConfig] = None,
) -> AudioLoader:
    """Factory function to create an AudioLoader based on configuration."""
    config = config or AudioConfig()
    return SoundEventAudioLoader(
        samplerate=config.samplerate,
        config=config.resample,
    )


def build_audio_transform_step(
    config: AudioTransform,
    samplerate: int,
) -> torch.nn.Module:
    if config.name == "fix_duration":
        return FixDuration(samplerate=samplerate, duration=config.duration)

    if config.name == "scale_audio":
        return PeakNormalize()

    if config.name == "center_audio":
        return CenterTensor()

    raise NotImplementedError(
        f"Audio preprocessing step {config.name} not implemented"
    )


def build_audio_pipeline(config: AudioConfig) -> torch.nn.Module:
    return torch.nn.Sequential(
        *[
            build_audio_transform_step(step, samplerate=config.samplerate)
            for step in config.transforms
        ]
    )
