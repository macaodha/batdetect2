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
    build_audio_loader,
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
    get_spectrogram_resolution,
)
from batdetect2.preprocess.types import (
    AudioLoader,
    PreprocessorProtocol,
    SpectrogramBuilder,
)

__all__ = [
    "AudioConfig",
    "AudioLoader",
    "ConfigurableSpectrogramBuilder",
    "DEFAULT_DURATION",
    "FrequencyConfig",
    "MAX_FREQ",
    "MIN_FREQ",
    "PcenConfig",
    "PreprocessingConfig",
    "ResampleConfig",
    "SCALE_RAW_AUDIO",
    "STFTConfig",
    "SpecSizeConfig",
    "SpectrogramBuilder",
    "SpectrogramConfig",
    "StandardPreprocessor",
    "TARGET_SAMPLERATE_HZ",
    "build_audio_loader",
    "build_preprocessor",
    "build_spectrogram_builder",
    "get_spectrogram_resolution",
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


class StandardPreprocessor(PreprocessorProtocol):
    """Standard implementation of the `Preprocessor` protocol.

    Orchestrates the audio loading and spectrogram generation pipeline using
    an `AudioLoader` and a `SpectrogramBuilder` internally, which are
    configured according to a `PreprocessingConfig`.

    This class is typically instantiated using the `build_preprocessor`
    factory function.

    Attributes
    ----------
    audio_loader : AudioLoader
        The configured audio loader instance used for waveform loading and
        initial processing.
    spectrogram_builder : SpectrogramBuilder
        The configured spectrogram builder instance used for generating
        spectrograms from waveforms.
    default_samplerate : int
        The sample rate (in Hz) assumed for input waveforms when they are
        provided as raw NumPy arrays without coordinate information (e.g.,
        when calling `compute_spectrogram` directly with `np.ndarray`).
        This value is derived from the `AudioConfig` (target resample rate
        or default if resampling is off) and also serves as documentation
        for the pipeline's intended operating sample rate. Note that when
        processing `xr.DataArray` inputs that have coordinate information
        (the standard internal workflow), the sample rate embedded in the
        coordinates takes precedence over this default value during
        spectrogram calculation.
    """

    audio_loader: AudioLoader
    spectrogram_builder: SpectrogramBuilder
    default_samplerate: int

    def __init__(
        self,
        audio_loader: AudioLoader,
        spectrogram_builder: SpectrogramBuilder,
        default_samplerate: int,
    ) -> None:
        """Initialize the StandardPreprocessor.

        Parameters
        ----------
        audio_loader : AudioLoader
            An initialized audio loader conforming to the AudioLoader protocol.
        spectrogram_builder : SpectrogramBuilder
            An initialized spectrogram builder conforming to the
            SpectrogramBuilder protocol.
        default_samplerate : int
            The sample rate to assume for NumPy array inputs and potentially
            reflecting the target rate of the audio config.
        """
        self.audio_loader = audio_loader
        self.spectrogram_builder = spectrogram_builder
        self.default_samplerate = default_samplerate

    def load_file_audio(
        self,
        path: data.PathLike,
        audio_dir: Optional[data.PathLike] = None,
    ) -> xr.DataArray:
        """Load and preprocess *only* the audio waveform from a file path.

        Delegates to the internal `audio_loader`.

        Parameters
        ----------
        path : PathLike
            Path to the audio file.
        audio_dir : PathLike, optional
            A directory prefix if `path` is relative.

        Returns
        -------
        xr.DataArray
            The loaded and preprocessed audio waveform (typically first
            channel).
        """
        return self.audio_loader.load_file(
            path,
            audio_dir=audio_dir,
        )

    def load_recording_audio(
        self,
        recording: data.Recording,
        audio_dir: Optional[data.PathLike] = None,
    ) -> xr.DataArray:
        """Load and preprocess *only* the audio waveform for a Recording.

        Delegates to the internal `audio_loader`.

        Parameters
        ----------
        recording : data.Recording
            The Recording object.
        audio_dir : PathLike, optional
            Directory containing the audio file.

        Returns
        -------
        xr.DataArray
            The loaded and preprocessed audio waveform (typically first
            channel).
        """
        return self.audio_loader.load_recording(
            recording,
            audio_dir=audio_dir,
        )

    def load_clip_audio(
        self,
        clip: data.Clip,
        audio_dir: Optional[data.PathLike] = None,
    ) -> xr.DataArray:
        """Load and preprocess *only* the audio waveform for a Clip.

        Delegates to the internal `audio_loader`.

        Parameters
        ----------
        clip : data.Clip
            The Clip object defining the segment.
        audio_dir : PathLike, optional
            Directory containing the audio file.

        Returns
        -------
        xr.DataArray
            The loaded and preprocessed audio waveform segment (typically first
            channel).
        """
        return self.audio_loader.load_clip(
            clip,
            audio_dir=audio_dir,
        )

    def preprocess_file(
        self,
        path: data.PathLike,
        audio_dir: Optional[data.PathLike] = None,
    ) -> xr.DataArray:
        """Load audio from a file and compute the final processed spectrogram.

        Performs the full pipeline:

            Load -> Preprocess Audio -> Compute Spectrogram.

        Parameters
        ----------
        path : PathLike
            Path to the audio file.
        audio_dir : PathLike, optional
            A directory prefix if `path` is relative.

        Returns
        -------
        xr.DataArray
            The final processed spectrogram.
        """
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
        """Load audio for a Recording and compute the processed spectrogram.

        Performs the full pipeline for the entire duration of the recording.

        Parameters
        ----------
        recording : data.Recording
            The Recording object.
        audio_dir : PathLike, optional
            Directory containing the audio file.

        Returns
        -------
        xr.DataArray
            The final processed spectrogram.
        """
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
        """Load audio for a Clip and compute the final processed spectrogram.

        Performs the full pipeline for the specified clip segment.

        Parameters
        ----------
        clip : data.Clip
            The Clip object defining the audio segment.
        audio_dir : PathLike, optional
            Directory containing the audio file.

        Returns
        -------
        xr.DataArray
            The final processed spectrogram.
        """
        wav = self.load_clip_audio(clip, audio_dir=audio_dir)
        return self.spectrogram_builder(
            wav,
            samplerate=self.default_samplerate,
        )

    def compute_spectrogram(
        self, wav: Union[xr.DataArray, np.ndarray]
    ) -> xr.DataArray:
        """Compute the spectrogram from a pre-loaded audio waveform.

        Applies the configured spectrogram generation steps
        (STFT, scaling, etc.) using the internal `spectrogram_builder`.

        If `wav` is a NumPy array, the `default_samplerate` stored in this
        preprocessor instance will be used. If `wav` is an xarray DataArray
        with time coordinates, the sample rate derived from those coordinates
        will take precedence over `default_samplerate`.

        Parameters
        ----------
        wav : Union[xr.DataArray, np.ndarray]
            The input audio waveform. If numpy array, `default_samplerate`
            stored in this object will be assumed.

        Returns
        -------
        xr.DataArray
            The computed spectrogram.
        """
        return self.spectrogram_builder(
            wav,
            samplerate=self.default_samplerate,
        )


def load_preprocessing_config(
    path: data.PathLike,
    field: Optional[str] = None,
) -> PreprocessingConfig:
    """Load the unified preprocessing configuration from a file.

    Reads a configuration file (YAML) and validates it against the
    `PreprocessingConfig` schema, potentially extracting data from a nested
    field.

    Parameters
    ----------
    path : PathLike
        Path to the configuration file.
    field : str, optional
        Dot-separated path to a nested section within the file containing the
        preprocessing configuration (e.g., "train.preprocessing"). If None, the
        entire file content is validated as the PreprocessingConfig.

    Returns
    -------
    PreprocessingConfig
        Loaded and validated preprocessing configuration object.

    Raises
    ------
    FileNotFoundError
        If the config file path does not exist.
    yaml.YAMLError
        If the file content is not valid YAML.
    pydantic.ValidationError
        If the loaded config data does not conform to PreprocessingConfig.
    KeyError, TypeError
        If `field` specifies an invalid path.
    """
    return load_config(path, schema=PreprocessingConfig, field=field)


def build_preprocessor(
    config: Optional[PreprocessingConfig] = None,
) -> PreprocessorProtocol:
    """Factory function to build the standard preprocessor from configuration.

    Creates instances of the required `AudioLoader` and `SpectrogramBuilder`
    based on the provided `PreprocessingConfig` (or defaults if config is None),
    determines the effective default sample rate, and initializes the
    `StandardPreprocessor`.

    Parameters
    ----------
    config : PreprocessingConfig, optional
        The unified preprocessing configuration object. If None, default
        configurations for audio and spectrogram processing will be used.

    Returns
    -------
    Preprocessor
        An initialized `StandardPreprocessor` instance ready to process audio
        according to the configuration.
    """
    config = config or PreprocessingConfig()

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
