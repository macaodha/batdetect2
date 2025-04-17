"""Computes spectrograms from audio waveforms with configurable parameters.

This module provides the functionality to convert preprocessed audio waveforms
(typically output from the `batdetect2.preprocessing.audio` module) into
spectrogram representations suitable for input into deep learning models like
BatDetect2.

It offers a configurable pipeline including:
1.  Short-Time Fourier Transform (STFT) calculation.
2.  Frequency axis cropping to a relevant range.
3.  Amplitude scaling (e.g., Logarithmic, Per-Channel Energy Normalization -
    PCEN).
4.  Simple denoising (optional).
5.  Resizing to target dimensions (optional).
6.  Final peak normalization (optional).

Configuration is managed via the `SpectrogramConfig` class, allowing for
reproducible spectrogram generation consistent between training and inference.
The core computation is performed by `compute_spectrogram`.
"""

from typing import Literal, Optional, Protocol, Union

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

__all__ = [
    "SpectrogramBuilder",
    "STFTConfig",
    "FrequencyConfig",
    "SpecSizeConfig",
    "LogScaleConfig",
    "PcenScaleConfig",
    "AmplitudeScaleConfig",
    "Scales",
    "SpectrogramConfig",
    "ConfigurableSpectrogramBuilder",
    "build_spectrogram_builder",
    "compute_spectrogram",
    "get_spectrogram_resolution",
]


class SpectrogramBuilder(Protocol):
    """Defines the interface for a spectrogram generation component.

    A SpectrogramBuilder takes a waveform (as numpy array or xarray DataArray)
    and produces a spectrogram (as an xarray DataArray) based on its internal
    configuration or implementation.
    """

    def __call__(
        self,
        wav: Union[np.ndarray, xr.DataArray],
        samplerate: Optional[int] = None,
    ) -> xr.DataArray:
        """Generate a spectrogram from an audio waveform.

        Parameters
        ----------
        wav : Union[np.ndarray, xr.DataArray]
            The input audio waveform. If a numpy array, `samplerate` must
            also be provided. If an xarray DataArray, it must have a 'time'
            coordinate from which the sample rate can be inferred.
        samplerate : int, optional
            The sample rate of the audio in Hz. Required if `wav` is a
            numpy array. If `wav` is an xarray DataArray, this parameter is
            ignored as the sample rate is derived from the coordinates.

        Returns
        -------
        xr.DataArray
            The computed spectrogram as an xarray DataArray with 'time' and
            'frequency' coordinates.

        Raises
        ------
        ValueError
            If `wav` is a numpy array and `samplerate` is not provided, or
            if `wav` is an xarray DataArray without a valid 'time' coordinate.
        """
        ...


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
        2.0 doubles it. If None (default), no resizing along the time axis
        is performed relative to the STFT output width. Must be > 0 if provided.
    """

    height: int = 128
    resize_factor: Optional[float] = 0.5


class LogScaleConfig(BaseConfig):
    """Configuration marker for using Logarithmic Amplitude Scaling."""

    name: Literal["log"] = "log"


class PcenScaleConfig(BaseConfig):
    """Configuration for Per-Channel Energy Normalization (PCEN) scaling.

    PCEN is an adaptive gain control method often used for audio event
    detection.

    Attributes
    ----------
    name : Literal["pcen"]
        Discriminator field identifying this scaling type.
    time_constant : float, default=0.4
        Time constant (in seconds) for the PCEN smoothing filter. Controls how
        quickly the normalization adapts to energy changes.
    gain : float, default=0.98
        Gain factor (alpha in some formulations). Controls the AGC behavior.
    bias : float, default=2.0
        Bias factor (delta in some formulations). Added before the
        exponentiation.
    power : float, default=0.5
        Exponent (r in some formulations). Controls the compression
        characteristic.
    """

    name: Literal["pcen"] = "pcen"
    time_constant: float = 0.4
    gain: float = 0.98
    bias: float = 2
    power: float = 0.5


class AmplitudeScaleConfig(BaseConfig):
    """Configuration marker for using Linear Amplitude (no scaling applied).

    Note: The actual output is typically magnitude from STFT, not raw amplitude.
    This option essentially skips log or PCEN scaling.
    """

    name: Literal["amplitude"] = "amplitude"


Scales = Union[LogScaleConfig, PcenScaleConfig, AmplitudeScaleConfig]
"""Type alias for the different amplitude scaling configuration options."""


class SpectrogramConfig(BaseConfig):
    """Unified configuration for spectrogram generation.

    Aggregates settings for STFT, frequency selection, amplitude scaling,
    resizing, and optional post-processing steps like denoising and final
    normalization.

    Attributes
    ----------
    stft : STFTConfig
        Configuration for the Short-Time Fourier Transform. Defaults to standard
        settings via `STFTConfig`.
    frequencies : FrequencyConfig
        Configuration for cropping the frequency range. Defaults to standard
        settings via `FrequencyConfig`.
    scale : Scales
        Configuration for amplitude scaling. Determines whether to apply
        log scaling, PCEN, or leave as linear magnitude. Defaults to PCEN
        via `PcenScaleConfig`. Use the `name` field ("log", "pcen", "amplitude")
        in config files to select the type and provide relevant parameters.
    size : SpecSizeConfig, optional
        Configuration for resizing the final spectrogram dimensions (height in
        frequency bins, optional time resizing factor). If None or omitted,
        no resizing is performed after STFT and frequency cropping. Defaults
        to standard settings via `SpecSizeConfig`.
    denoise : bool, default=True
        If True (default), applies a simple spectral mean subtraction denoising
        step after amplitude scaling.
    max_scale : bool, default=False
        If True, applies a final peak normalization to the spectrogram *after*
        all other steps (including log/PCEN scaling and resizing), scaling the
        maximum value across the entire spectrogram to 1.0. If False (default),
        this final scaling is skipped. **Note:** Applying this after log or PCEN
        scaling will alter the characteristics of those scales.
    """

    stft: STFTConfig = Field(default_factory=STFTConfig)
    frequencies: FrequencyConfig = Field(default_factory=FrequencyConfig)
    scale: Scales = Field(
        default_factory=PcenScaleConfig,
        discriminator="name",
    )
    size: Optional[SpecSizeConfig] = Field(default_factory=SpecSizeConfig)
    denoise: bool = True
    max_scale: bool = False


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
    3. Apply amplitude scaling (log, PCEN, or none) (`scale_spectrogram`).
    4. Apply denoising if enabled (`denoise_spectrogram`).
    5. Resize dimensions if specified (`resize_spectrogram`).
    6. Apply final peak normalization if enabled (`max_scale`).

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

    spec = scale_spectrogram(spec, scale=config.scale)

    if config.denoise:
        spec = denoise_spectrogram(spec)

    if config.size:
        spec = resize_spectrogram(
            spec,
            height=config.size.height,
            resize_factor=config.size.resize_factor,
        )

    if config.max_scale:
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


def denoise_spectrogram(spec: xr.DataArray) -> xr.DataArray:
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
    scale: Scales,
    dtype: DTypeLike = np.float32,  # type: ignore
) -> xr.DataArray:
    """Apply configured amplitude scaling to the spectrogram.

    Dispatches to the appropriate scaling function (log, PCEN) based on the
    `scale` configuration object's `name` field. If `scale.name` is
    "amplitude", the spectrogram is returned unchanged (as it's already
    magnitude/amplitude).

    Parameters
    ----------
    spec : xr.DataArray
        Input magnitude spectrogram.
    scale : Scales
        The configuration object specifying the scaling method and parameters
        (instance of LogScaleConfig, PcenScaleConfig, or AmplitudeScaleConfig).
    dtype : DTypeLike, default=np.float32
        Target data type for the output scaled spectrogram.

    Returns
    -------
    xr.DataArray
        Spectrogram with the specified amplitude scaling applied.
    """
    if scale.name == "log":
        return scale_log(spec, dtype=dtype)

    if scale.name == "pcen":
        return scale_pcen(
            spec,
            time_constant=scale.time_constant,
            gain=scale.gain,
            power=scale.power,
            bias=scale.bias,
        )

    return spec


def scale_pcen(
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
