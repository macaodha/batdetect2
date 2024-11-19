"""Module containing functions for preprocessing audio clips."""

from typing import Optional

import xarray as xr
from pydantic import BaseModel, Field
from soundevent import data

from batdetect2.preprocess.audio import (
    AudioConfig,
    ResampleConfig,
    load_clip_audio,
)
from batdetect2.preprocess.spectrogram import (
    AmplitudeScaleConfig,
    FFTConfig,
    FrequencyConfig,
    LogScaleConfig,
    PcenScaleConfig,
    Scales,
    SpecSizeConfig,
    SpectrogramConfig,
    compute_spectrogram,
)

__all__ = [
    "AudioConfig",
    "ResampleConfig",
    "SpectrogramConfig",
    "FFTConfig",
    "FrequencyConfig",
    "PcenScaleConfig",
    "LogScaleConfig",
    "AmplitudeScaleConfig",
    "Scales",
    "SpecSizeConfig",
    "PreprocessingConfig",
    "preprocess_audio_clip",
]


class PreprocessingConfig(BaseModel):
    """Configuration for preprocessing data."""

    audio: AudioConfig = Field(default_factory=AudioConfig)
    spectrogram: SpectrogramConfig = Field(default_factory=SpectrogramConfig)


def preprocess_audio_clip(
    clip: data.Clip,
    config: Optional[PreprocessingConfig] = None,
    audio_dir: Optional[data.PathLike] = None,
) -> xr.DataArray:
    """Preprocesses audio clip to generate spectrogram.

    Parameters
    ----------
    clip
        The audio clip to preprocess.
    config
        Configuration for preprocessing.

    Returns
    -------
    xr.DataArray
        Preprocessed spectrogram.

    """
    config = config or PreprocessingConfig()
    wav = load_clip_audio(clip, config=config.audio, audio_dir=audio_dir)
    return compute_spectrogram(wav, config=config.spectrogram)
