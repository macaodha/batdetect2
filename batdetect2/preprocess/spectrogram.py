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
from batdetect2.preprocess.audio import DEFAULT_DURATION

FFT_WIN_LENGTH_S = 512 / 256000.0
FFT_OVERLAP = 0.75
MAX_FREQ_HZ = 120000
MIN_FREQ_HZ = 10000
SPEC_HEIGHT = 128
SPEC_WIDTH = 256
SPEC_SCALE = "pcen"
SPEC_TIME_PERIOD = DEFAULT_DURATION / SPEC_WIDTH
DENOISE_SPEC_AVG = True
MAX_SCALE_SPEC = False


class FFTConfig(BaseConfig):
    window_duration: float = Field(default=FFT_WIN_LENGTH_S, gt=0)
    window_overlap: float = Field(default=FFT_OVERLAP, ge=0, lt=1)
    window_fn: str = "hann"


class FrequencyConfig(BaseConfig):
    max_freq: int = Field(default=MAX_FREQ_HZ, gt=0)
    min_freq: int = Field(default=MIN_FREQ_HZ, gt=0)


class PcenConfig(BaseConfig):
    time_constant: float = 0.4
    hop_length: int = 512
    gain: float = 0.98
    bias: float = 2
    power: float = 0.5


class SpecSizeConfig(BaseConfig):
    height: int = SPEC_HEIGHT
    time_period: float = SPEC_TIME_PERIOD


class SpectrogramConfig(BaseConfig):
    fft: FFTConfig = Field(default_factory=FFTConfig)
    frequencies: FrequencyConfig = Field(default_factory=FrequencyConfig)
    scale: Union[Literal["log"], None, PcenConfig] = "log"
    denoise: bool = True
    resize: Optional[SpecSizeConfig] = Field(default_factory=SpecSizeConfig)
    max_scale: bool = MAX_SCALE_SPEC


def compute_spectrogram(
    wav: xr.DataArray,
    config: Optional[SpectrogramConfig] = None,
    dtype: DTypeLike = np.float32,
) -> xr.DataArray:
    config = config or SpectrogramConfig()

    spec = stft(
        wav,
        window_duration=config.fft.window_duration,
        window_overlap=config.fft.window_overlap,
        window_fn=config.fft.window_fn,
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

    if config.resize:
        spec = resize_spectrogram(spec, config=config.resize)

    if config.max_scale:
        spec = ops.scale(spec, 1 / (10e-6 + np.max(spec)))

    return spec.astype(dtype)


def crop_spectrogram_frequencies(
    spec: xr.DataArray,
    min_freq: int = MIN_FREQ_HZ,
    max_freq: int = MAX_FREQ_HZ,
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
    dtype: DTypeLike = np.float32,
) -> xr.DataArray:
    start_time, end_time = arrays.get_dim_range(wave, dim="time")
    step = arrays.get_dim_step(wave, dim="time")
    sampling_rate = 1 / step

    hop_len = window_duration * (1 - window_overlap)
    nfft = int(window_duration * sampling_rate)
    noverlap = int(window_overlap * nfft)

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
                    end_time - (window_duration - hop_len),
                    spec.shape[1],
                    endpoint=False,
                    dtype=dtype,
                ),
                step=hop_len,
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
    scale: Union[Literal["log"], None, PcenConfig],
    dtype: DTypeLike = np.float32,
) -> xr.DataArray:
    if scale == "log":
        return scale_log(spec, dtype=dtype)

    if isinstance(scale, PcenConfig):
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
    # NOTE: Not sure why the 10 is there
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
    dtype: DTypeLike = np.float32,
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
    config: SpecSizeConfig,
) -> xr.DataArray:
    duration = arrays.get_dim_width(spec, dim="time")
    return ops.resize(
        spec,
        time=int(np.ceil(duration / config.time_period)),
        frequency=config.height,
        dtype=np.float32,
    )
