from collections.abc import Sequence
from typing import Annotated, List, Literal, Optional, Union

from pydantic import Field
from soundevent.data import PathLike

from batdetect2.core.configs import BaseConfig, load_config

__all__ = [
    "load_preprocessing_config",
    "CenterAudioConfig",
    "ScaleAudioConfig",
    "FixDurationConfig",
    "ResampleConfig",
    "AudioTransform",
    "AudioConfig",
    "STFTConfig",
    "FrequencyConfig",
    "PcenConfig",
    "ScaleAmplitudeConfig",
    "SpectralMeanSubstractionConfig",
    "ResizeConfig",
    "PeakNormalizeConfig",
    "SpectrogramTransform",
    "SpectrogramConfig",
    "PreprocessingConfig",
    "TARGET_SAMPLERATE_HZ",
    "MIN_FREQ",
    "MAX_FREQ",
]

TARGET_SAMPLERATE_HZ = 256_000
"""Default target sample rate in Hz used if resampling is enabled."""

MIN_FREQ = 10_000
"""Default minimum frequency (Hz) for spectrogram frequency cropping."""

MAX_FREQ = 120_000
"""Default maximum frequency (Hz) for spectrogram frequency cropping."""


class CenterAudioConfig(BaseConfig):
    name: Literal["center_audio"] = "center_audio"


class ScaleAudioConfig(BaseConfig):
    name: Literal["scale_audio"] = "scale_audio"


class FixDurationConfig(BaseConfig):
    name: Literal["fix_duration"] = "fix_duration"
    duration: float = 0.5


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


class STFTConfig(BaseConfig):
    """Configuration for the Short-Time Fourier Transform (STFT).

    Attributes
    ----------
    window_duration : float, default=0.002
        Duration of the STFT window in seconds (e.g., 0.002 for 2ms). Must be
        > 0. Determines frequency resolution (longer window = finer frequency
        resolution).
    window_overlap : float, default=0.75
        Fraction of overlap between consecutive STFT windows (e.g., 0.75
        for 75%). Must be >= 0 and < 1. Determines time resolution
        (higher overlap = finer time resolution).
    window_fn : str, default="hann"
        Name of the window function to apply before FFT calculation. Common
        options include "hann", "hamming", "blackman". See
        `scipy.signal.get_window`.
    """

    window_duration: float = Field(default=0.002, gt=0)
    window_overlap: float = Field(default=0.75, ge=0, lt=1)
    window_fn: str = "hann"


class FrequencyConfig(BaseConfig):
    """Configuration for frequency axis parameters.

    Attributes
    ----------
    max_freq : int, default=120000
        Maximum frequency in Hz to retain in the spectrogram after STFT.
        Frequencies above this value will be cropped. Must be > 0.
    min_freq : int, default=10000
        Minimum frequency in Hz to retain in the spectrogram after STFT.
        Frequencies below this value will be cropped. Must be >= 0.
    """

    max_freq: int = Field(default=120_000, ge=0)
    min_freq: int = Field(default=10_000, ge=0)


class PcenConfig(BaseConfig):
    """Configuration for Per-Channel Energy Normalization (PCEN)."""

    name: Literal["pcen"] = "pcen"
    time_constant: float = 0.4
    gain: float = 0.98
    bias: float = 2
    power: float = 0.5


class ScaleAmplitudeConfig(BaseConfig):
    name: Literal["scale_amplitude"] = "scale_amplitude"
    scale: Literal["power", "db"] = "db"


class SpectralMeanSubstractionConfig(BaseConfig):
    name: Literal["spectral_mean_substraction"] = "spectral_mean_substraction"


class ResizeConfig(BaseConfig):
    name: Literal["resize_spec"] = "resize_spec"
    height: int = 128
    resize_factor: float = 0.5


class PeakNormalizeConfig(BaseConfig):
    name: Literal["peak_normalize"] = "peak_normalize"


SpectrogramTransform = Annotated[
    Union[
        PcenConfig,
        ScaleAmplitudeConfig,
        SpectralMeanSubstractionConfig,
        PeakNormalizeConfig,
    ],
    Field(discriminator="name"),
]


class SpectrogramConfig(BaseConfig):
    stft: STFTConfig = Field(default_factory=STFTConfig)
    frequencies: FrequencyConfig = Field(default_factory=FrequencyConfig)
    size: ResizeConfig = Field(default_factory=ResizeConfig)
    transforms: Sequence[SpectrogramTransform] = Field(
        default_factory=lambda: [
            PcenConfig(),
            SpectralMeanSubstractionConfig(),
        ]
    )


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
