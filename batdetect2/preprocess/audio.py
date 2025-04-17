"""Handles loading and initial preprocessing of audio waveforms.

This module provides components for loading audio data associated with
`soundevent` objects (Clips, Recordings, or raw files) and applying
fundamental waveform processing steps. These steps typically include:

1.  Loading the raw audio data.
2.  Adjusting the audio clip to a fixed duration (optional).
3.  Resampling the audio to a target sample rate (optional).
4.  Centering the waveform (DC offset removal) (optional).
5.  Scaling the waveform amplitude (optional).

The processing pipeline is configurable via the `AudioConfig` data structure,
allowing for reproducible preprocessing consistent between model training and
inference. It uses the `soundevent` library for audio loading and basic array
operations, and `scipy` for resampling implementations.

The primary interface is the `AudioLoader` protocol, with
`ConfigurableAudioLoader` providing a concrete implementation driven by the
`AudioConfig`.
"""

from typing import Optional

import numpy as np
import xarray as xr
from numpy.typing import DTypeLike
from pydantic import Field
from scipy.signal import resample, resample_poly
from soundevent import arrays, audio, data
from soundevent.arrays import operations as ops
from soundfile import LibsndfileError

from batdetect2.configs import BaseConfig
from batdetect2.preprocess.types import AudioLoader

__all__ = [
    "ResampleConfig",
    "AudioConfig",
    "ConfigurableAudioLoader",
    "build_audio_loader",
    "load_file_audio",
    "load_recording_audio",
    "load_clip_audio",
    "adjust_audio_duration",
    "resample_audio",
    "TARGET_SAMPLERATE_HZ",
    "SCALE_RAW_AUDIO",
    "DEFAULT_DURATION",
    "convert_to_xr",
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

    samplerate: int = Field(default=TARGET_SAMPLERATE_HZ, gt=0)
    method: str = "poly"


class AudioConfig(BaseConfig):
    """Configuration for loading and initial audio preprocessing.

    Defines the sequence of operations applied to raw audio waveforms after
    loading, controlling steps like resampling, scaling, centering, and
    duration adjustment.

    Attributes
    ----------
    resample : ResampleConfig, optional
        Configuration for resampling. If provided (or defaulted), audio will
        be resampled to the specified `samplerate` using the specified
        `method`. If set to `None` in the config file, resampling is skipped.
        Defaults to a ResampleConfig instance with standard settings.
    scale : bool, default=False
        If True, scales the audio waveform using peak normalization so that
        its maximum absolute amplitude is approximately 1.0. If False
        (default), no amplitude scaling is applied.
    center : bool, default=True
        If True (default), centers the waveform by subtracting its mean
        (DC offset removal). If False, the waveform is not centered.
    duration : float, optional
        If set to a float value (seconds), the loaded audio clip will be
        adjusted (cropped or padded with zeros) to exactly this duration.
        If None (default), the original duration is kept.
    """

    resample: Optional[ResampleConfig] = Field(default_factory=ResampleConfig)
    scale: bool = SCALE_RAW_AUDIO
    center: bool = True
    duration: Optional[float] = DEFAULT_DURATION


class ConfigurableAudioLoader:
    """Concrete implementation of the `AudioLoader` driven by `AudioConfig`.

    This class loads audio and applies preprocessing steps (resampling,
    scaling, centering, duration adjustment) based on the settings provided
    in an `AudioConfig` object during initialization. It delegates the actual
    work to module-level functions.
    """

    def __init__(
        self,
        config: AudioConfig,
    ):
        """Initialize the ConfigurableAudioLoader.

        Parameters
        ----------
        config : AudioConfig
            The configuration object specifying the desired preprocessing steps
            and parameters.
        """
        self.config = config

    def load_file(
        self,
        path: data.PathLike,
        audio_dir: Optional[data.PathLike] = None,
    ) -> xr.DataArray:
        """Load and preprocess audio directly from a file path.

        Implements the `AudioLoader.load_file` method by delegating to the
        `load_file_audio` function, passing the stored configuration.

        Parameters
        ----------
        path : PathLike
            Path to the audio file.
        audio_dir : PathLike, optional
            A directory prefix if `path` is relative.

        Returns
        -------
        xr.DataArray
            Loaded and preprocessed waveform (first channel).
        """
        return load_file_audio(path, config=self.config, audio_dir=audio_dir)

    def load_recording(
        self,
        recording: data.Recording,
        audio_dir: Optional[data.PathLike] = None,
    ) -> xr.DataArray:
        """Load and preprocess the entire audio for a Recording object.

        Implements the `AudioLoader.load_recording` method by delegating to the
        `load_recording_audio` function, passing the stored configuration.

        Parameters
        ----------
        recording : data.Recording
            The Recording object.
        audio_dir : PathLike, optional
            Directory containing the audio file.

        Returns
        -------
        xr.DataArray
            Loaded and preprocessed waveform (first channel).
        """
        return load_recording_audio(
            recording, config=self.config, audio_dir=audio_dir
        )

    def load_clip(
        self,
        clip: data.Clip,
        audio_dir: Optional[data.PathLike] = None,
    ) -> xr.DataArray:
        """Load and preprocess the audio segment defined by a Clip object.

        Implements the `AudioLoader.load_clip` method by delegating to the
        `load_clip_audio` function, passing the stored configuration.

        Parameters
        ----------
        clip : data.Clip
            The Clip object specifying the segment.
        audio_dir : PathLike, optional
            Directory containing the audio file.

        Returns
        -------
        xr.DataArray
            Loaded and preprocessed waveform segment (first channel).
        """
        return load_clip_audio(clip, config=self.config, audio_dir=audio_dir)


def build_audio_loader(
    config: AudioConfig,
) -> AudioLoader:
    """Factory function to create an AudioLoader based on configuration.

    Instantiates and returns a `ConfigurableAudioLoader` initialized with
    the provided `AudioConfig`. The return type is `AudioLoader`, adhering
    to the protocol.

    Parameters
    ----------
    config : AudioConfig
        The configuration object specifying preprocessing steps.

    Returns
    -------
    AudioLoader
        An instance of `ConfigurableAudioLoader` ready to load and process audio
        according to the configuration.
    """
    return ConfigurableAudioLoader(config=config)


def load_file_audio(
    path: data.PathLike,
    config: Optional[AudioConfig] = None,
    audio_dir: Optional[data.PathLike] = None,
    dtype: DTypeLike = np.float32,  # type: ignore
) -> xr.DataArray:
    """Load and preprocess audio from a file path using specified config.

    Creates a `soundevent.data.Recording` object from the file path and then
    delegates the loading and processing to `load_recording_audio`.

    Parameters
    ----------
    path : PathLike
        Path to the audio file.
    config : AudioConfig, optional
        Audio processing configuration. If None, default settings defined
        in `AudioConfig` are used.
    audio_dir : PathLike, optional
        Directory prefix if `path` is relative.
    dtype : DTypeLike, default=np.float32
        Target NumPy data type for the loaded audio array.

    Returns
    -------
    xr.DataArray
        Loaded and preprocessed waveform (first channel only).
    """
    try:
        recording = data.Recording.from_file(path)
    except LibsndfileError as e:
        raise FileNotFoundError(
            f"Could not load the recording at path: {path}. Error: {e}"
        ) from e

    return load_recording_audio(
        recording,
        config=config,
        dtype=dtype,
        audio_dir=audio_dir,
    )


def load_recording_audio(
    recording: data.Recording,
    config: Optional[AudioConfig] = None,
    audio_dir: Optional[data.PathLike] = None,
    dtype: DTypeLike = np.float32,  # type: ignore
) -> xr.DataArray:
    """Load and preprocess the entire audio content of a recording using config.

    Creates a `soundevent.data.Clip` spanning the full duration of the
    recording and then delegates the loading and processing to `load_clip_audio`.

    Parameters
    ----------
    recording : data.Recording
        The Recording object containing metadata and file path.
    config : AudioConfig, optional
        Audio processing configuration. If None, default settings are used.
    audio_dir : PathLike, optional
        Directory containing the audio file, used if the path in `recording`
        is relative.
    dtype : DTypeLike, default=np.float32
        Target NumPy data type for the loaded audio array.

    Returns
    -------
    xr.DataArray
        Loaded and preprocessed waveform (first channel only).
    """
    clip = data.Clip(
        recording=recording,
        start_time=0,
        end_time=recording.duration,
    )
    return load_clip_audio(
        clip,
        config=config,
        dtype=dtype,
        audio_dir=audio_dir,
    )


def load_clip_audio(
    clip: data.Clip,
    config: Optional[AudioConfig] = None,
    audio_dir: Optional[data.PathLike] = None,
    dtype: DTypeLike = np.float32,  # type: ignore
) -> xr.DataArray:
    """Load and preprocess a specific audio clip segment based on config.

    This is the core function performing the configured processing pipeline:
    1. Loads the specified clip segment using `soundevent.audio.load_clip`.
    2. Selects the first audio channel.
    3. Resamples if `config.resample` is configured.
    4. Centers (DC offset removal) if `config.center` is True.
    5. Scales (peak normalization) if `config.scale` is True.
    6. Adjusts duration (crop/pad) if `config.duration` is set.

    Parameters
    ----------
    clip : data.Clip
        The Clip object defining the audio segment and source recording.
    config : AudioConfig, optional
        Audio processing configuration. If None, a default `AudioConfig` is
        used.
    audio_dir : PathLike, optional
        Directory containing the source audio file specified in the clip's
        recording.
    dtype : DTypeLike, default=np.float32
        Target NumPy data type for the processed audio array.

    Returns
    -------
    xr.DataArray
        The loaded and preprocessed waveform segment as an xarray DataArray
        with time coordinates.

    Raises
    ------
    FileNotFoundError
        If the underlying audio file cannot be found.
    Exception
        If audio loading or processing fails for other reasons (e.g., invalid
        format, resampling error).

    Notes
    -----
    - **Mono Processing:** This function currently loads and processes only the
      **first channel** (channel 0) of the audio file. Any other channels
      are ignored.
    """
    config = config or AudioConfig()

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

    if config.resample:
        wav = resample_audio(
            wav,
            samplerate=config.resample.samplerate,
            dtype=dtype,
        )

    if config.center:
        wav = ops.center(wav)

    if config.scale:
        wav = scale_audio(wav)

    if config.duration is not None:
        wav = adjust_audio_duration(wav, duration=config.duration)

    return wav.astype(dtype)


def scale_audio(
    wave: xr.DataArray,
) -> xr.DataArray:
    """
    Scale the audio waveform to have a maximum absolute value of 1.0.

    This function normalizes the waveform by dividing it by its maximum
    absolute value. If the maximum value is zero, the waveform is returned
    unchanged. Also known as peak normalization, this process ensures that the
    waveform's amplitude is within a standard range, which can be useful for
    audio processing and analysis.

    """
    max_val = np.max(np.abs(wave))

    if max_val == 0:
        return wave

    return ops.scale(wave, 1 / max_val)


def adjust_audio_duration(
    wave: xr.DataArray,
    duration: float,
) -> xr.DataArray:
    """Adjust the duration of an audio waveform array via cropping or padding.

    If the current duration is longer than the target, it crops the array
    from the beginning. If shorter, it pads the array with zeros at the end
    using `soundevent.arrays.extend_dim`.

    Parameters
    ----------
    wave : xr.DataArray
        The input audio waveform with a 'time' dimension and coordinates.
    duration : float
        The target duration in seconds.

    Returns
    -------
    xr.DataArray
        The waveform adjusted to the target duration. Returns the input
        unmodified if duration already matches or if the wave is empty.

    Raises
    ------
    ValueError
        If `duration` is negative.
    """
    start_time, end_time = arrays.get_dim_range(wave, dim="time")
    step = arrays.get_dim_step(wave, dim="time")
    current_duration = end_time - start_time + step

    if current_duration == duration:
        return wave

    with xr.set_options(keep_attrs=True):
        if current_duration > duration:
            return arrays.crop_dim(
                wave,
                dim="time",
                start=start_time,
                stop=start_time + duration - step / 2,
                right_closed=True,
            )

        return arrays.extend_dim(
            wave,
            dim="time",
            start=start_time,
            stop=start_time + duration - step / 2,
            eps=0,
            right_closed=True,
        )


def resample_audio(
    wav: xr.DataArray,
    samplerate: int = TARGET_SAMPLERATE_HZ,
    method: str = "poly",
    dtype: DTypeLike = np.float32,  # type: ignore
) -> xr.DataArray:
    """Resample an audio waveform DataArray to a target sample rate.

    Updates the 'time' coordinate axis according to the new sample rate and
    number of samples. Uses either polyphase (`scipy.signal.resample_poly`)
    or Fourier method (`scipy.signal.resample`) based on the `method`.

    Parameters
    ----------
    wav : xr.DataArray
        Input audio waveform with 'time' dimension and coordinates.
    samplerate : int, default=TARGET_SAMPLERATE_HZ
        Target sample rate in Hz.
    method : str, default="poly"
        Resampling algorithm: "poly" or "fourier".
    dtype : DTypeLike, default=np.float32
        Target data type for the resampled array.

    Returns
    -------
    xr.DataArray
        Resampled waveform with updated time coordinates. Returns the input
        unmodified (but dtype cast) if the sample rate is already correct or
        if the input array is empty.

    Raises
    ------
    ValueError
        If `wav` lacks a 'time' dimension, the original sample rate cannot
        be determined, `samplerate` is non-positive, or `method` is invalid.
    """
    if "time" not in wav.dims:
        raise ValueError("Audio must have a time dimension")

    time_axis: int = wav.get_axis_num("time")  # type: ignore
    step = arrays.get_dim_step(wav, dim="time")
    original_samplerate = int(1 / step)

    if original_samplerate == samplerate:
        return wav.astype(dtype)

    if method == "poly":
        resampled = resample_audio_poly(
            wav,
            sr_orig=original_samplerate,
            sr_new=samplerate,
            axis=time_axis,
        )
    elif method == "fourier":
        resampled = resample_audio_fourier(
            wav,
            sr_orig=original_samplerate,
            sr_new=samplerate,
            axis=time_axis,
        )
    else:
        raise NotImplementedError(
            f"Resampling method '{method}' not implemented"
        )

    start, stop = arrays.get_dim_range(wav, dim="time")
    times = np.linspace(
        start,
        stop + step,
        len(resampled),
        endpoint=False,
        dtype=dtype,
    )

    return xr.DataArray(
        data=resampled.astype(dtype),
        dims=wav.dims,
        coords={
            **wav.coords,
            "time": arrays.create_time_dim_from_array(
                times,
                samplerate=samplerate,
            ),
        },
        attrs={**wav.attrs, "samplerate": samplerate},
    )


def resample_audio_poly(
    array: xr.DataArray,
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
        array.values,
        sr_new // gcd,
        sr_orig // gcd,
        axis=axis,
    )


def resample_audio_fourier(
    array: xr.DataArray,
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
    return resample(array, int(array.shape[axis] * ratio), axis=axis)  # type: ignore


def convert_to_xr(
    wav: np.ndarray,
    samplerate: int,
    dtype: DTypeLike = np.float32,  # type: ignore
) -> xr.DataArray:
    """Convert a NumPy array to an xarray DataArray with time coordinates.

    Parameters
    ----------
    wav : np.ndarray
        The input waveform array. Expected to be 1D or 2D (with the first axis as
        the channel dimension).
    samplerate : int
        The sample rate in Hz.
    dtype : DTypeLike, default=np.float32
        Target data type for the xarray DataArray.

    Returns
    -------
    xr.DataArray
        The waveform as an xarray DataArray with time coordinates.

    Raises
    ------
    ValueError
        If the input array is not 1D or 2D, or if the sample rate is
        non-positive. If the input array is empty.
    """

    if wav.ndim == 2:
        wav = wav[0, :]

    if wav.ndim != 1:
        raise ValueError(
            "Audio must be 1D array or 2D channel where the first axis is the channel dimension"
        )

    if wav.size == 0:
        raise ValueError("Audio array is empty")

    if samplerate <= 0:
        raise ValueError("Sample rate must be positive")

    times = np.linspace(
        0,
        wav.shape[0] / samplerate,
        wav.shape[0],
        endpoint=False,
        dtype=dtype,
    )

    return xr.DataArray(
        data=wav.astype(dtype),
        dims=["time"],
        coords={
            "time": arrays.create_time_dim_from_array(
                times,
                samplerate=samplerate,
            ),
        },
        attrs={"samplerate": samplerate},
    )
