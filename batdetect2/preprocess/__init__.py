"""Module containing functions for preprocessing audio clips."""

from typing import Optional, Union

import numpy as np
import xarray as xr
from pydantic import Field
from soundevent import data

from batdetect2.configs import BaseConfig, load_config
from batdetect2.preprocess.audio import (
    DEFAULT_DURATION,
    SCALE_RAW_AUDIO,
    TARGET_SAMPLERATE_HZ,
    AudioConfig,
    ResampleConfig,
    adjust_audio_duration,
    build_audio_loader,
    convert_to_xr,
    load_clip_audio,
    load_file_audio,
    load_recording_audio,
    resample_audio,
)
from batdetect2.preprocess.spectrogram import (
    MAX_FREQ,
    MIN_FREQ,
    ConfigurableSpectrogramBuilder,
    FrequencyConfig,
    PcenConfig,
    SpecSizeConfig,
    SpectrogramConfig,
    STFTConfig,
    build_spectrogram_builder,
    compute_spectrogram,
    get_spectrogram_resolution,
)
from batdetect2.preprocess.types import (
    AudioLoader,
    Preprocessor,
    SpectrogramBuilder,
)

__all__ = [
    "AudioConfig",
    "AudioLoader",
    "ConfigurableSpectrogramBuilder",
    "DEFAULT_DURATION",
    "FrequencyConfig",
    "FrequencyConfig",
    "MAX_FREQ",
    "MIN_FREQ",
    "PcenConfig",
    "PcenConfig",
    "PreprocessingConfig",
    "ResampleConfig",
    "SCALE_RAW_AUDIO",
    "STFTConfig",
    "STFTConfig",
    "SpecSizeConfig",
    "SpecSizeConfig",
    "SpectrogramBuilder",
    "SpectrogramConfig",
    "SpectrogramConfig",
    "TARGET_SAMPLERATE_HZ",
    "adjust_audio_duration",
    "build_audio_loader",
    "build_spectrogram_builder",
    "compute_spectrogram",
    "convert_to_xr",
    "get_spectrogram_resolution",
    "load_clip_audio",
    "load_file_audio",
    "load_preprocessing_config",
    "load_recording_audio",
    "resample_audio",
]


class PreprocessingConfig(BaseConfig):
    """Configuration for preprocessing data."""

    audio: AudioConfig = Field(default_factory=AudioConfig)
    spectrogram: SpectrogramConfig = Field(default_factory=SpectrogramConfig)


class StandardPreprocessor(Preprocessor):
    audio_loader: AudioLoader
    spectrogram_builder: SpectrogramBuilder
    default_samplerate: int

    def __init__(
        self,
        audio_loader: AudioLoader,
        spectrogram_builder: SpectrogramBuilder,
        default_samplerate: int,
    ) -> None:
        self.audio_loader = audio_loader
        self.spectrogram_builder = spectrogram_builder
        self.default_samplerate = default_samplerate

    def load_file_audio(
        self,
        path: data.PathLike,
        audio_dir: Optional[data.PathLike] = None,
    ) -> xr.DataArray:
        return self.audio_loader.load_file(
            path,
            audio_dir=audio_dir,
        )

    def load_recording_audio(
        self,
        recording: data.Recording,
        audio_dir: Optional[data.PathLike] = None,
    ) -> xr.DataArray:
        return self.audio_loader.load_recording(
            recording,
            audio_dir=audio_dir,
        )

    def load_clip_audio(
        self,
        clip: data.Clip,
        audio_dir: Optional[data.PathLike] = None,
    ) -> xr.DataArray:
        return self.audio_loader.load_clip(
            clip,
            audio_dir=audio_dir,
        )

    def preprocess_file(
        self,
        path: data.PathLike,
        audio_dir: Optional[data.PathLike] = None,
    ) -> xr.DataArray:
        wav = self.load_file_audio(path, audio_dir=audio_dir)
        return self.spectrogram_builder(
            wav,
            samplerate=self.default_samplerate,
        )

    def preprocess_recording(
        self,
        recording: data.Recording,
        audio_dir: Optional[data.PathLike] = None,
    ) -> xr.DataArray:
        wav = self.load_recording_audio(recording, audio_dir=audio_dir)
        return self.spectrogram_builder(
            wav,
            samplerate=self.default_samplerate,
        )

    def preprocess_clip(
        self,
        clip: data.Clip,
        audio_dir: Optional[data.PathLike] = None,
    ) -> xr.DataArray:
        wav = self.load_clip_audio(clip, audio_dir=audio_dir)
        return self.spectrogram_builder(
            wav,
            samplerate=self.default_samplerate,
        )

    def compute_spectrogram(
        self, wav: Union[xr.DataArray, np.ndarray]
    ) -> xr.DataArray:
        return self.spectrogram_builder(
            wav,
            samplerate=self.default_samplerate,
        )


def load_preprocessing_config(
    path: data.PathLike,
    field: Optional[str] = None,
) -> PreprocessingConfig:
    return load_config(path, schema=PreprocessingConfig, field=field)


def build_preprocessor_from_config(
    config: PreprocessingConfig,
) -> Preprocessor:
    default_samplerate = (
        config.audio.resample.samplerate
        if config.audio.resample
        else TARGET_SAMPLERATE_HZ
    )
    return StandardPreprocessor(
        audio_loader=build_audio_loader(config.audio),
        spectrogram_builder=build_spectrogram_builder(config.spectrogram),
        default_samplerate=default_samplerate,
    )
