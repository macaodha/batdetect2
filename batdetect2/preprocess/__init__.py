"""Module containing functions for preprocessing audio clips."""

from typing import Optional

import xarray as xr
from soundevent import data

from batdetect2.preprocess.audio import (
    AudioConfig,
    ResampleConfig,
    load_clip_audio,
)
from batdetect2.preprocess.config import (
    PreprocessingConfig,
    load_preprocessing_config,
)
from batdetect2.preprocess.spectrogram import (
    AmplitudeScaleConfig,
    FrequencyConfig,
    LogScaleConfig,
    PcenScaleConfig,
    Scales,
    SpecSizeConfig,
    SpectrogramConfig,
    STFTConfig,
    compute_spectrogram,
)

__all__ = [
    "AmplitudeScaleConfig",
    "AudioConfig",
    "FrequencyConfig",
    "LogScaleConfig",
    "PcenScaleConfig",
    "PreprocessingConfig",
    "ResampleConfig",
    "STFTConfig",
    "Scales",
    "SpecSizeConfig",
    "SpectrogramConfig",
    "load_preprocessing_config",
    "preprocess_audio_clip",
]


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
