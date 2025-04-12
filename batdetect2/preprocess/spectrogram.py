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

__all__ = [
    "STFTConfig",
    "FrequencyConfig",
    "LogScaleConfig",
    "PcenScaleConfig",
    "AmplitudeScaleConfig",
    "Scales",
    "SpectrogramConfig",
    "compute_spectrogram",
]

MIN_FREQ = 10_000
MAX_FREQ = 120_000


class STFTConfig(BaseConfig):
    window_duration: float = Field(default=0.002, gt=0)
    window_overlap: float = Field(default=0.75, ge=0, lt=1)
    window_fn: str = "hann"


class FrequencyConfig(BaseConfig):
    max_freq: int = Field(default=120_000, gt=0)
    min_freq: int = Field(default=10_000, gt=0)


class SpecSizeConfig(BaseConfig):
    height: int = 128
    """Height of the spectrogram in pixels. This value determines the 
    number of frequency bands and corresponds to the vertical dimension 
    of the spectrogram."""

    resize_factor: Optional[float] = 0.5
    """Factor by which to resize the spectrogram along the time axis. 
    A value of 0.5 reduces the temporal dimension by half, while a 
    value of 2.0 doubles it. If None, no resizing is performed."""


class LogScaleConfig(BaseConfig):
    name: Literal["log"] = "log"


class PcenScaleConfig(BaseConfig):
    name: Literal["pcen"] = "pcen"
    time_constant: float = 0.4
    hop_length: int = 512
    gain: float = 0.98
    bias: float = 2
    power: float = 0.5


class AmplitudeScaleConfig(BaseConfig):
    name: Literal["amplitude"] = "amplitude"


Scales = Union[LogScaleConfig, PcenScaleConfig, AmplitudeScaleConfig]


class SpectrogramConfig(BaseConfig):
    stft: STFTConfig = Field(default_factory=STFTConfig)
    frequencies: FrequencyConfig = Field(default_factory=FrequencyConfig)
    scale: Scales = Field(
        default_factory=PcenScaleConfig,
        discriminator="name",
    )
    size: Optional[SpecSizeConfig] = Field(default_factory=SpecSizeConfig)
    denoise: bool = True
    max_scale: bool = False


def compute_spectrogram(
    wav: xr.DataArray,
    config: Optional[SpectrogramConfig] = None,
    dtype: DTypeLike = np.float32,  # type: ignore
) -> xr.DataArray:
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
    if scale.name == "log":
        return scale_log(spec, dtype=dtype)

    if scale.name == "pcen":
        return scale_pcen(
            spec,
            time_constant=scale.time_constant,
            hop_length=scale.hop_length,
            gain=scale.gain,
            power=scale.power,
            bias=scale.bias,
        )

    return spec


def scale_pcen(
    spec: xr.DataArray,
    time_constant: float = 0.4,
    hop_length: int = 512,
    gain: float = 0.98,
    bias: float = 2,
    power: float = 0.5,
) -> xr.DataArray:
    samplerate = spec.attrs["original_samplerate"]
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
    resize_factor = resize_factor or 1
    current_width = spec.sizes["time"]
    return ops.resize(
        spec,
        time=int(resize_factor * current_width),
        frequency=height,
        dtype=np.float32,
    )


def adjust_spectrogram_width(
    spec: xr.DataArray,
    divide_factor: int = 32,
    time_period: float = 0.001,
) -> xr.DataArray:
    time_width = spec.sizes["time"]

    if time_width % divide_factor == 0:
        return spec

    target_size = int(
        np.ceil(spec.sizes["time"] / divide_factor) * divide_factor
    )
    extra_duration = (target_size - time_width) * time_period
    _, stop = arrays.get_dim_range(spec, dim="time")
    resized = ops.extend_dim(
        spec,
        dim="time",
        stop=stop + extra_duration,
    )
    return resized


def duration_to_spec_width(
    duration: float,
    samplerate: int,
    window_duration: float,
    window_overlap: float,
) -> int:
    samples = int(duration * samplerate)
    fft_len = int(window_duration * samplerate)
    fft_overlap = int(window_overlap * fft_len)
    hop_len = fft_len - fft_overlap
    width = (samples - fft_len + hop_len) / hop_len
    return int(np.floor(width))


def spec_width_to_samples(
    width: int,
    samplerate: int,
    window_duration: float,
    window_overlap: float,
) -> int:
    fft_len = int(window_duration * samplerate)
    fft_overlap = int(window_overlap * fft_len)
    hop_len = fft_len - fft_overlap
    return width * hop_len + fft_len - hop_len


def get_spectrogram_resolution(
    config: SpectrogramConfig,
) -> tuple[float, float]:
    max_freq = config.frequencies.max_freq
    min_freq = config.frequencies.min_freq
    assert config.size is not None

    spec_height = config.size.height
    resize_factor = config.size.resize_factor or 1
    freq_bin_width = (max_freq - min_freq) / spec_height
    hop_duration = config.stft.window_duration * (
        1 - config.stft.window_overlap
    )
    return freq_bin_width, hop_duration / resize_factor
