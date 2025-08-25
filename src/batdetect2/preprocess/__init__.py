"""Main entry point for the BatDetect2 Preprocessing subsystem.

This package (`batdetect2.preprocessing`) defines and orchestrates the pipeline
for converting raw audio input (from files or data objects) into processed
spectrograms suitable for input to BatDetect2 models. This ensures consistent
data handling between model training and inference.

The preprocessing pipeline consists of two main stages, configured via nested
data structures:
1.  **Audio Processing (`.audio`)**: Loads audio waveforms and applies initial
    processing like resampling, duration adjustment, centering, and scaling.
    Configured via `AudioConfig`.
2.  **Spectrogram Generation (`.spectrogram`)**: Computes the spectrogram from
    the processed waveform using STFT, followed by frequency cropping, optional
    PCEN, amplitude scaling (dB, power, linear), optional denoising, optional
    resizing, and optional peak normalization. Configured via
    `SpectrogramConfig`.

This module provides the primary interface:

- `PreprocessingConfig`: A unified configuration object holding `AudioConfig`
  and `SpectrogramConfig`.
- `load_preprocessing_config`: Function to load the unified configuration.
- `Preprocessor`: A protocol defining the interface for the end-to-end pipeline.
- `StandardPreprocessor`: The default implementation of the `Preprocessor`.
- `build_preprocessor`: A factory function to create a `StandardPreprocessor`
  instance from a `PreprocessingConfig`.

"""

from typing import Optional

import torch
from loguru import logger
from pydantic import Field
from soundevent.data import PathLike

from batdetect2.configs import BaseConfig, load_config
from batdetect2.preprocess.audio import (
    DEFAULT_DURATION,
    SCALE_RAW_AUDIO,
    TARGET_SAMPLERATE_HZ,
    AudioConfig,
    ResampleConfig,
    build_audio_loader,
    build_audio_pipeline,
)
from batdetect2.preprocess.spectrogram import (
    MAX_FREQ,
    MIN_FREQ,
    FrequencyConfig,
    PcenConfig,
    SpectrogramConfig,
    SpectrogramPipeline,
    STFTConfig,
    build_spectrogram_builder,
    build_spectrogram_pipeline,
)
from batdetect2.typing import PreprocessorProtocol

__all__ = [
    "AudioConfig",
    "DEFAULT_DURATION",
    "FrequencyConfig",
    "MAX_FREQ",
    "MIN_FREQ",
    "PcenConfig",
    "PreprocessingConfig",
    "ResampleConfig",
    "SCALE_RAW_AUDIO",
    "STFTConfig",
    "SpectrogramConfig",
    "TARGET_SAMPLERATE_HZ",
    "build_audio_loader",
    "build_spectrogram_builder",
    "load_preprocessing_config",
]


class PreprocessingConfig(BaseConfig):
    """Unified configuration for the audio preprocessing pipeline.

    Aggregates the configuration for both the initial audio processing stage
    and the subsequent spectrogram generation stage.

    Attributes
    ----------
    audio : AudioConfig
        Configuration settings for the audio loading and initial waveform
        processing steps (e.g., resampling, duration adjustment, scaling).
        Defaults to default `AudioConfig` settings if omitted.
    spectrogram : SpectrogramConfig
        Configuration settings for the spectrogram generation process
        (e.g., STFT parameters, frequency cropping, scaling, denoising,
        resizing). Defaults to default `SpectrogramConfig` settings if omitted.
    """

    audio: AudioConfig = Field(default_factory=AudioConfig)
    spectrogram: SpectrogramConfig = Field(default_factory=SpectrogramConfig)


def load_preprocessing_config(
    path: PathLike,
    field: Optional[str] = None,
) -> PreprocessingConfig:
    return load_config(path, schema=PreprocessingConfig, field=field)


class StandardPreprocessor(torch.nn.Module, PreprocessorProtocol):
    """Standard implementation of the `Preprocessor` protocol."""

    samplerate: int
    max_freq: float
    min_freq: float

    def __init__(
        self,
        audio_pipeline: torch.nn.Module,
        spectrogram_pipeline: SpectrogramPipeline,
        samplerate: int,
        max_freq: float,
        min_freq: float,
    ) -> None:
        super().__init__()
        self.audio_pipeline = audio_pipeline
        self.spectrogram_pipeline = spectrogram_pipeline
        self.samplerate = samplerate
        self.max_freq = max_freq
        self.min_freq = min_freq

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        wav = self.audio_pipeline(wav)
        return self.spectrogram_pipeline(wav)


def build_preprocessor(
    config: Optional[PreprocessingConfig] = None,
) -> PreprocessorProtocol:
    """Factory function to build the standard preprocessor from configuration."""
    config = config or PreprocessingConfig()
    logger.opt(lazy=True).debug(
        "Building preprocessor with config: \n{}",
        lambda: config.to_yaml_string(),
    )

    samplerate = config.audio.samplerate

    min_freq = config.spectrogram.frequencies.min_freq
    max_freq = config.spectrogram.frequencies.max_freq

    return StandardPreprocessor(
        audio_pipeline=build_audio_pipeline(config.audio),
        spectrogram_pipeline=build_spectrogram_pipeline(
            samplerate, config.spectrogram
        ),
        samplerate=samplerate,
        min_freq=min_freq,
        max_freq=max_freq,
    )


def get_default_preprocessor():
    return build_preprocessor()
