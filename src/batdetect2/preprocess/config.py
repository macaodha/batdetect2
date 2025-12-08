from typing import List, Optional

from pydantic import Field
from soundevent.data import PathLike

from batdetect2.core.configs import BaseConfig, load_config
from batdetect2.preprocess.audio import AudioTransform
from batdetect2.preprocess.spectrogram import (
    FrequencyConfig,
    PcenConfig,
    ResizeConfig,
    SpectralMeanSubstractionConfig,
    SpectrogramTransform,
    STFTConfig,
)

__all__ = [
    "load_preprocessing_config",
    "AudioTransform",
    "PreprocessingConfig",
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

    audio_transforms: List[AudioTransform] = Field(default_factory=list)

    spectrogram_transforms: List[SpectrogramTransform] = Field(
        default_factory=lambda: [
            PcenConfig(),
            SpectralMeanSubstractionConfig(),
        ]
    )

    stft: STFTConfig = Field(default_factory=STFTConfig)

    frequencies: FrequencyConfig = Field(default_factory=FrequencyConfig)

    size: ResizeConfig = Field(default_factory=ResizeConfig)


def load_preprocessing_config(
    path: PathLike,
    field: str | None = None,
) -> PreprocessingConfig:
    return load_config(path, schema=PreprocessingConfig, field=field)
