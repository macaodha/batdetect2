"""Computes spectrograms from audio waveforms with configurable parameters.

This module provides the functionality to convert preprocessed audio waveforms
(typically output from the `batdetect2.preprocessing.audio` module) into
spectrogram representations suitable for input into deep learning models like
BatDetect2.

It offers a configurable pipeline including:
1.  Short-Time Fourier Transform (STFT) calculation to get magnitude.
2.  Frequency axis cropping to a relevant range.
3.  Per-Channel Energy Normalization (PCEN) (optional).
4.  Amplitude scaling/representation (dB, power, or linear amplitude).
5.  Simple spectral mean subtraction denoising (optional).
6.  Resizing to target dimensions (optional).
7.  Final peak normalization (optional).

Configuration is managed via the `SpectrogramConfig` class, allowing for
reproducible spectrogram generation consistent between training and inference.
The core computation is performed by `compute_spectrogram`.
"""

from typing import Literal, Optional, Union

import librosa
import librosa.core.spectrum
import numpy as np
import xarray as xr
from numpy.typing import DTypeLike
from pydantic import Field
from soundevent import arrays, audio
from soundevent.arrays import operations as ops

from batdetect2.configs import BaseConfig
from batdetect2.preprocess.audio import convert_to_xr
from batdetect2.preprocess.types import SpectrogramBuilder

__all__ = [
    "STFTConfig",
    "FrequencyConfig",
    "SpecSizeConfig",
    "PcenConfig",
    "SpectrogramConfig",
    "ConfigurableSpectrogramBuilder",
    "build_spectrogram_builder",
    "compute_spectrogram",
    "get_spectrogram_resolution",
    "MIN_FREQ",
    "MAX_FREQ",
]


MIN_FREQ = 10_000
"""Default minimum frequency (Hz) for spectrogram frequency cropping."""

MAX_FREQ = 120_000
"""Default maximum frequency (Hz) for spectrogram frequency cropping."""


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
        Frequencies below this value will be cropped. Must be > 0.
    """

    max_freq: int = Field(default=120_000, gt=0)
    min_freq: int = Field(default=10_000, gt=0)


class SpecSizeConfig(BaseConfig):
    """Configuration for the final size and shape of the spectrogram.

    Attributes
    ----------
    height : int, default=128
        Target height of the spectrogram in pixels (frequency bins). The
        frequency axis will be resized (e.g., via interpolation) to match this
        height after frequency cropping and amplitude scaling. Must be > 0.
    resize_factor : float, optional
        Factor by which to resize the spectrogram along the time axis *after*
        STFT calculation. A value of 0.5 halves the number of time bins,
        2.0 doubles it. If None (default), no resizing along the time axis is
        performed relative to the STFT output width. Must be > 0 if provided.
    """

    height: int = 128
    resize_factor: Optional[float] = 0.5


class PcenConfig(BaseConfig):
    """Configuration for Per-Channel Energy Normalization (PCEN).

    PCEN is an adaptive gain control method that can help emphasize transients
    and suppress stationary noise. Applied after STFT and frequency cropping,
    but before final amplitude scaling (dB, power, amplitude).

    Attributes
    ----------
    time_constant : float, default=0.4
        Time constant (in seconds) for the PCEN smoothing filter. Controls
        how quickly the normalization adapts to energy changes.
    gain : float, default=0.98
        Gain factor (alpha). Controls the adaptive gain component.
    bias : float, default=2.0
        Bias factor (delta). Added before the exponentiation.
    power : float, default=0.5
        Exponent (r). Controls the compression characteristic.
    """

    time_constant: float = 0.4
    gain: float = 0.98
    bias: float = 2
    power: float = 0.5


class SpectrogramConfig(BaseConfig):
    """Unified configuration for spectrogram generation pipeline.

    Aggregates settings for all steps involved in converting a preprocessed
    audio waveform into a final spectrogram representation suitable for model input.

    Attributes
    ----------
    stft : STFTConfig
        Configuration for the initial Short-Time Fourier Transform.
        Defaults to standard settings via `STFTConfig`.
    frequencies : FrequencyConfig
        Configuration for cropping the frequency range after STFT.
        Defaults to standard settings via `FrequencyConfig`.
    pcen : PcenConfig, optional
        Configuration for applying Per-Channel Energy Normalization (PCEN). If
        provided, PCEN is applied after frequency cropping. If None or omitted
        (default), PCEN is skipped.
    scale : Literal["dB", "amplitude", "power"], default="amplitude"
        Determines the final amplitude representation *after* optional PCEN.
        - "amplitude": Use linear magnitude values (output of STFT or PCEN).
        - "power": Use power values (magnitude squared).
        - "dB": Use logarithmic (decibel-like) scaling applied to the magnitude
                (or PCEN output if enabled). Calculated as `log1p(C * S)`.
    size : SpecSizeConfig, optional
        Configuration for resizing the spectrogram dimensions
        (frequency height, optional time width factor). Applied after PCEN and
        scaling. If None (default), no resizing is performed.
    spectral_mean_substraction : bool, default=True
        If True (default), applies simple spectral mean subtraction denoising
        *after* PCEN and amplitude scaling, but *before* resizing.
    peak_normalize : bool, default=False
        If True, applies a final peak normalization to the spectrogram *after*
        all other steps (including resizing), scaling the overall maximum value
        to 1.0. If False (default), this final normalization is skipped.
    """

    stft: STFTConfig = Field(default_factory=STFTConfig)
    frequencies: FrequencyConfig = Field(default_factory=FrequencyConfig)
    pcen: Optional[PcenConfig] = Field(default_factory=PcenConfig)
    scale: Literal["dB", "amplitude", "power"] = "amplitude"
    size: Optional[SpecSizeConfig] = Field(default_factory=SpecSizeConfig)
    spectral_mean_substraction: bool = True
    peak_normalize: bool = False


class ConfigurableSpectrogramBuilder(SpectrogramBuilder):
    """Implementation of `SpectrogramBuilder` driven by `SpectrogramConfig`.

    This class computes spectrograms according to the parameters specified in a
    `SpectrogramConfig` object provided during initialization. It handles both
    numpy array and xarray DataArray inputs for the waveform.
    """

    def __init__(
        self,
        config: SpectrogramConfig,
        dtype: DTypeLike = np.float32,  # type: ignore
    ) -> None:
        """Initialize the ConfigurableSpectrogramBuilder.

        Parameters
        ----------
        config : SpectrogramConfig
            The configuration object specifying all spectrogram parameters.
        dtype : DTypeLike, default=np.float32
            The target NumPy data type for the computed spectrogram array.
        """
        self.config = config
        self.dtype = dtype

    def __call__(
        self,
        wav: Union[np.ndarray, xr.DataArray],
        samplerate: Optional[int] = None,
    ) -> xr.DataArray:
        """Generate a spectrogram from an audio waveform using the config.

        Implements the `SpectrogramBuilder` protocol. If the input `wav` is
        a numpy array, `samplerate` must be provided; the array will be
        converted to an xarray DataArray internally. If `wav` is already an
        xarray DataArray with time coordinates, `samplerate` is ignored.
        Delegates the main computation to `compute_spectrogram`.

        Parameters
        ----------
        wav : Union[np.ndarray, xr.DataArray]
            The input audio waveform.
        samplerate : int, optional
            The sample rate in Hz (required only if `wav` is np.ndarray).

        Returns
        -------
        xr.DataArray
            The computed spectrogram.

        Raises
        ------
        ValueError
            If `wav` is np.ndarray and `samplerate` is None.
        """
        if isinstance(wav, np.ndarray):
            if samplerate is None:
                raise ValueError(
                    "Samplerate must be provided when passing a numpy array."
                )
            wav = convert_to_xr(
                wav,
                samplerate=samplerate,
                dtype=self.dtype,
            )

        return compute_spectrogram(
            wav,
            config=self.config,
            dtype=self.dtype,
        )


def build_spectrogram_builder(
    config: SpectrogramConfig,
    dtype: DTypeLike = np.float32,  # type: ignore
) -> SpectrogramBuilder:
    """Factory function to create a SpectrogramBuilder based on configuration.

    Instantiates and returns a `ConfigurableSpectrogramBuilder` initialized
    with the provided `SpectrogramConfig`.

    Parameters
    ----------
    config : SpectrogramConfig
        The configuration object specifying spectrogram parameters.
    dtype : DTypeLike, default=np.float32
        The target NumPy data type for the computed spectrogram array.

    Returns
    -------
    SpectrogramBuilder
        An instance of `ConfigurableSpectrogramBuilder` ready to compute
        spectrograms according to the configuration.
    """
    return ConfigurableSpectrogramBuilder(config=config, dtype=dtype)


def compute_spectrogram(
    wav: xr.DataArray,
    config: Optional[SpectrogramConfig] = None,
    dtype: DTypeLike = np.float32,  # type: ignore
) -> xr.DataArray:
    """Compute a spectrogram from a waveform using specified configurations.

    Applies a sequence of operations based on the `config`:
    1. Compute STFT magnitude (`stft`).
    2. Crop frequency axis (`crop_spectrogram_frequencies`).
    3. Apply PCEN if configured (`apply_pcen`).
    4. Apply final amplitude scaling (dB, power, amplitude) (`scale_spectrogram`).
    5. Apply spectral mean subtraction denoising if enabled.
    6. Resize dimensions if specified (`resize_spectrogram`).
    7. Apply final peak normalization if enabled.

    Parameters
    ----------
    wav : xr.DataArray
        Input audio waveform with a 'time' dimension and coordinates from
        which the sample rate can be inferred.
    config : SpectrogramConfig, optional
        Configuration object specifying spectrogram parameters. If None,
        default settings from `SpectrogramConfig` are used.
    dtype : DTypeLike, default=np.float32
        Target NumPy data type for the final spectrogram array.

    Returns
    -------
    xr.DataArray
        The computed and processed spectrogram with 'time' and 'frequency'
        coordinates.

    Raises
    ------
    ValueError
        If `wav` lacks necessary 'time' coordinates or dimensions.
    Exception
        Can re-raise exceptions from underlying libraries (e.g., librosa, numpy)
        if invalid parameters or data are encountered.
    """
    config = config or SpectrogramConfig()

    spec = stft(
        wav,
        window_duration=config.stft.window_duration,
        window_overlap=config.stft.window_overlap,
        window_fn=config.stft.window_fn,
        dtype=dtype,
    )

    spec = crop_spectrogram_frequencies(
        spec,
        min_freq=config.frequencies.min_freq,
        max_freq=config.frequencies.max_freq,
    )

    if config.pcen:
        spec = apply_pcen(
            spec,
            time_constant=config.pcen.time_constant,
            gain=config.pcen.gain,
            power=config.pcen.power,
            bias=config.pcen.bias,
        )

    spec = scale_spectrogram(spec, scale=config.scale)

    if config.spectral_mean_substraction:
        spec = remove_spectral_mean(spec)

    if config.size:
        spec = resize_spectrogram(
            spec,
            height=config.size.height,
            resize_factor=config.size.resize_factor,
        )

    if config.peak_normalize:
        spec = ops.scale(spec, 1 / (10e-6 + np.max(spec)))

    return spec.astype(dtype)


def crop_spectrogram_frequencies(
    spec: xr.DataArray,
    min_freq: int = 10_000,
    max_freq: int = 120_000,
) -> xr.DataArray:
    """Crop the frequency axis of a spectrogram to a specified range.

    Uses `soundevent.arrays.crop_dim` to select the frequency bins
    corresponding to the range [`min_freq`, `max_freq`].

    Parameters
    ----------
    spec : xr.DataArray
        Input spectrogram with 'frequency' dimension and coordinates.
    min_freq : int, default=MIN_FREQ
        Minimum frequency (Hz) to keep.
    max_freq : int, default=MAX_FREQ
        Maximum frequency (Hz) to keep.

    Returns
    -------
    xr.DataArray
        Spectrogram cropped along the frequency axis. Preserves dtype.
    """
    return arrays.crop_dim(
        spec,
        dim="frequency",
        start=min_freq,
        stop=max_freq,
    ).astype(spec.dtype)


def stft(
    wave: xr.DataArray,
    window_duration: float,
    window_overlap: float,
    window_fn: str = "hann",
    dtype: DTypeLike = np.float32,  # type: ignore
) -> xr.DataArray:
    """Compute the Short-Time Fourier Transform (STFT) magnitude spectrogram.

    Calculates STFT parameters (N-FFT, hop length) based on the window
    duration, overlap, and waveform sample rate. Returns an xarray DataArray
    with correctly calculated 'time' and 'frequency' coordinates.

    Parameters
    ----------
    wave : xr.DataArray
        Input audio waveform with 'time' coordinates.
    window_duration : float
        Duration of the STFT window in seconds.
    window_overlap : float
        Fractional overlap between consecutive windows [0, 1).
    window_fn : str, default="hann"
        Name of the window function (e.g., "hann", "hamming").
    dtype : DTypeLike, default=np.float32
        Target data type for the spectrogram array.

    Returns
    -------
    xr.DataArray
        Magnitude spectrogram with 'time' and 'frequency' dimensions and
        coordinates. STFT parameters are stored in the `attrs`.

    Raises
    ------
    ValueError
        If sample rate cannot be determined from `wave` coordinates.
    """
    start_time, end_time = arrays.get_dim_range(wave, dim="time")
    step = arrays.get_dim_step(wave, dim="time")
    sampling_rate = 1 / step

    nfft = int(window_duration * sampling_rate)
    noverlap = int(window_overlap * nfft)
    hop_len = nfft - noverlap
    hop_duration = hop_len / sampling_rate

    spec, _ = librosa.core.spectrum._spectrogram(
        y=wave.data.astype(dtype),
        power=1,
        n_fft=nfft,
        hop_length=nfft - noverlap,
        center=False,
        window=window_fn,
    )

    return xr.DataArray(
        data=spec.astype(dtype),
        dims=["frequency", "time"],
        coords={
            "frequency": arrays.create_frequency_dim_from_array(
                np.linspace(
                    0,
                    sampling_rate / 2,
                    spec.shape[0],
                    endpoint=False,
                    dtype=dtype,
                ),
                step=sampling_rate / nfft,
            ),
            "time": arrays.create_time_dim_from_array(
                np.linspace(
                    start_time,
                    end_time - (window_duration - hop_duration),
                    spec.shape[1],
                    endpoint=False,
                    dtype=dtype,
                ),
                step=hop_duration,
            ),
        },
        attrs={
            **wave.attrs,
            "original_samplerate": sampling_rate,
            "nfft": nfft,
            "noverlap": noverlap,
        },
    )


def remove_spectral_mean(spec: xr.DataArray) -> xr.DataArray:
    """Apply simple spectral mean subtraction for denoising.

    Subtracts the mean value of each frequency bin (calculated across time)
    from that bin, then clips negative values to zero.

    Parameters
    ----------
    spec : xr.DataArray
        Input spectrogram with 'time' and 'frequency' dimensions.

    Returns
    -------
    xr.DataArray
        Denoised spectrogram with the same dimensions, coordinates, and dtype.
    """
    return xr.DataArray(
        data=(spec - spec.mean("time")).clip(0),
        dims=spec.dims,
        coords=spec.coords,
        attrs=spec.attrs,
    )


def scale_spectrogram(
    spec: xr.DataArray,
    scale: Literal["dB", "power", "amplitude"],
    dtype: DTypeLike = np.float32,  # type: ignore
) -> xr.DataArray:
    """Apply final amplitude scaling/representation to the spectrogram.

    Converts the input magnitude spectrogram based on the `scale` type:
    - "dB": Applies logarithmic scaling `log1p(C * S)`.
    - "power": Squares the magnitude values `S^2`.
    - "amplitude": Returns the input magnitude values `S` unchanged.

    Parameters
    ----------
    spec : xr.DataArray
        Input magnitude spectrogram (potentially after PCEN).
    scale : Literal["dB", "power", "amplitude"]
        The target amplitude representation.
    dtype : DTypeLike, default=np.float32
        Target data type for the output scaled spectrogram.

    Returns
    -------
    xr.DataArray
        Spectrogram with the specified amplitude scaling applied.
    """
    if scale == "dB":
        return scale_log(spec, dtype=dtype)

    if scale == "power":
        return spec**2

    return spec


def apply_pcen(
    spec: xr.DataArray,
    time_constant: float = 0.4,
    gain: float = 0.98,
    bias: float = 2,
    power: float = 0.5,
) -> xr.DataArray:
    """Apply Per-Channel Energy Normalization (PCEN) to a spectrogram.

    Parameters
    ----------
    spec : xr.DataArray
        Input magnitude spectrogram with required attributes like
        'processing_original_samplerate'.
    time_constant : float, default=0.4
        PCEN time constant in seconds.
    gain : float, default=0.98
        Gain factor (alpha).
    bias : float, default=2.0
        Bias factor (delta).
    power : float, default=0.5
        Exponent (r).
    dtype : DTypeLike, default=np.float32
        Target data type for the output spectrogram.

    Returns
    -------
    xr.DataArray
        PCEN-scaled spectrogram.

    Notes
    -----
    - The input spectrogram magnitude `spec` is multiplied by `2**31` before
      being passed to `audio.pcen`. This suggests the underlying implementation
      might expect values in a range typical of 16-bit or 32-bit signed integers,
      even though the input here might be float. This scaling factor should be
      verified against the specific `soundevent.audio.pcen` implementation
      details.
    """
    samplerate = spec.attrs["original_samplerate"]
    hop_length = spec.attrs["nfft"] - spec.attrs["noverlap"]
    t_frames = time_constant * samplerate / (float(hop_length) * 10)
    smoothing_constant = (np.sqrt(1 + 4 * t_frames**2) - 1) / (2 * t_frames**2)
    return audio.pcen(
        spec * (2**31),
        smooth=smoothing_constant,
        gain=gain,
        bias=bias,
        power=power,
    ).astype(spec.dtype)


def scale_log(
    spec: xr.DataArray,
    dtype: DTypeLike = np.float32,  # type: ignore
) -> xr.DataArray:
    """Apply logarithmic scaling to a magnitude spectrogram.

    Calculates `log(1 + C * S)`, where S is the input magnitude spectrogram
    and C is a scaling factor derived from the original STFT parameters
    (sample rate, N-FFT, window function) stored in `spec.attrs`.

    Parameters
    ----------
    spec : xr.DataArray
        Input magnitude spectrogram with required attributes like
        'processing_original_samplerate', 'processing_nfft'.
    dtype : DTypeLike, default=np.float32
        Target data type for the output spectrogram.

    Returns
    -------
    xr.DataArray
        Log-scaled spectrogram.

    Raises
    ------
    KeyError
        If required attributes are missing from `spec.attrs`.
    ValueError
        If attributes are non-numeric or window function is invalid.
    """
    samplerate = spec.attrs["original_samplerate"]
    nfft = spec.attrs["nfft"]
    log_scaling = 2 / (samplerate * (np.abs(np.hanning(nfft)) ** 2).sum())
    return xr.DataArray(
        data=np.log1p(log_scaling * spec).astype(dtype),
        dims=spec.dims,
        coords=spec.coords,
        attrs=spec.attrs,
    )


def resize_spectrogram(
    spec: xr.DataArray,
    height: int = 128,
    resize_factor: Optional[float] = 0.5,
) -> xr.DataArray:
    """Resize a spectrogram to target dimensions using interpolation.

    Resizes the frequency axis to `height` bins and optionally resizes the
    time axis by `resize_factor`.

    Parameters
    ----------
    spec : xr.DataArray
        Input spectrogram with 'time' and 'frequency' dimensions.
    height : int, default=128
        Target number of frequency bins (vertical dimension).
    resize_factor : float, optional
        Factor to resize the time dimension. If 1.0 or None, time dimension
        is unchanged. If 0.5, time dimension is halved, etc.

    Returns
    -------
    xr.DataArray
        Resized spectrogram. Coordinates are typically adjusted by the
        underlying resize operation if implemented in `ops.resize`.
        The dtype is currently hardcoded to float32 by ops.resize call.
    """
    resize_factor = resize_factor or 1
    current_width = spec.sizes["time"]
    return ops.resize(
        spec,
        time=int(resize_factor * current_width),
        frequency=height,
        dtype=np.float32,
    )


def get_spectrogram_resolution(
    config: SpectrogramConfig,
) -> tuple[float, float]:
    """Calculate the approximate resolution of the final spectrogram.

    Computes the width of each frequency bin (Hz/bin) and the duration
    of each time bin (seconds/bin) based on the configuration parameters.

    Parameters
    ----------
    config : SpectrogramConfig
        The spectrogram configuration object.
    samplerate : int, optional
        The sample rate of the audio *before* STFT. Required if needed to
        calculate hop duration accurately from STFT config, but the current
        implementation calculates hop_duration directly from STFT config times.

    Returns
    -------
    Tuple[float, float]
        A tuple containing:
        - frequency_resolution (float): Approximate Hz per frequency bin.
        - time_resolution (float): Approximate seconds per time bin.

    Raises
    ------
    ValueError
        If required configuration fields (like `config.size`) are missing
        or invalid.
    """
    max_freq = config.frequencies.max_freq
    min_freq = config.frequencies.min_freq

    if config.size is None:
        raise ValueError("Spectrogram size configuration is required.")

    spec_height = config.size.height
    resize_factor = config.size.resize_factor or 1
    freq_bin_width = (max_freq - min_freq) / spec_height
    hop_duration = config.stft.window_duration * (
        1 - config.stft.window_overlap
    )
    return freq_bin_width, hop_duration / resize_factor
