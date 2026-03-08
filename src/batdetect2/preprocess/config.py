"""Configuration for the full batdetect2 preprocessing pipeline.

This module defines :class:`PreprocessingConfig`, which aggregates all
configuration needed to convert a raw audio waveform into a normalised
spectrogram ready for the detection model.
"""

from typing import List

from pydantic import Field
from soundevent.data import PathLike

from batdetect2.core.configs import BaseConfig, load_config
from batdetect2.preprocess.audio import AudioTransform
from batdetect2.preprocess.spectrogram import (
    FrequencyConfig,
    PcenConfig,
    ResizeConfig,
    SpectralMeanSubtractionConfig,
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

    Aggregates the parameters for every stage of the pipeline:
    audio-level transforms, STFT computation, frequency cropping,
    spectrogram-level transforms, and the final resize step.

    Attributes
    ----------
    audio_transforms : list of AudioTransform, default=[]
        Ordered list of transforms applied to the raw audio waveform
        before the STFT is computed. Each entry is a configuration
        object for one of the supported audio transforms
        (``"center_audio"``, ``"scale_audio"``, or ``"fix_duration"``).
    spectrogram_transforms : list of SpectrogramTransform
        Ordered list of transforms applied to the cropped spectrogram
        after the STFT and frequency crop steps. Defaults to
        ``[PcenConfig(), SpectralMeanSubtractionConfig()]``, which
        applies PCEN followed by spectral mean subtraction.
    stft : STFTConfig
        Parameters for the Short-Time Fourier Transform (window
        duration, overlap, and window function).
    frequencies : FrequencyConfig
        Frequency range (in Hz) to retain after the STFT.
    size : ResizeConfig
        Target height (number of frequency bins) and time-axis scaling
        factor for the final resize step.
    """

    audio_transforms: List[AudioTransform] = Field(default_factory=list)

    spectrogram_transforms: List[SpectrogramTransform] = Field(
        default_factory=lambda: [
            PcenConfig(),
            SpectralMeanSubtractionConfig(),
        ]
    )

    stft: STFTConfig = Field(default_factory=STFTConfig)

    frequencies: FrequencyConfig = Field(default_factory=FrequencyConfig)

    size: ResizeConfig = Field(default_factory=ResizeConfig)


def load_preprocessing_config(
    path: PathLike,
    field: str | None = None,
) -> PreprocessingConfig:
    """Load a ``PreprocessingConfig`` from a YAML file.

    Parameters
    ----------
    path : PathLike
        Path to the YAML configuration file.
    field : str, optional
        If provided, read the config from a nested field within the
        YAML document (e.g. ``"preprocessing"`` to read from a top-level
        ``preprocessing:`` key).

    Returns
    -------
    PreprocessingConfig
        The deserialised preprocessing configuration.
    """
    return load_config(path, schema=PreprocessingConfig, field=field)
