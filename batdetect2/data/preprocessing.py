"""Module containing functions for preprocessing audio clips."""

from typing import Optional

import librosa
import librosa.core.spectrum
import numpy as np
import xarray as xr
from numpy.typing import DTypeLike
from pydantic import BaseModel, Field
from scipy.signal import resample_poly
from soundevent import audio, data, arrays
from soundevent.arrays import operations as ops

__all__ = [
    "PreprocessingConfig",
    "preprocess_audio_clip",
]


TARGET_SAMPLERATE_HZ = 256000
SCALE_RAW_AUDIO = False
FFT_WIN_LENGTH_S = 512 / 256000.0
FFT_OVERLAP = 0.75
MAX_FREQ_HZ = 120000
MIN_FREQ_HZ = 10000
DEFAULT_DURATION = 1
SPEC_HEIGHT = 128
SPEC_WIDTH = 256
SPEC_SCALE = "pcen"
SPEC_TIME_PERIOD = DEFAULT_DURATION / SPEC_WIDTH
DENOISE_SPEC_AVG = True
MAX_SCALE_SPEC = False


class PreprocessingConfig(BaseModel):
    """Configuration for preprocessing data."""

    target_samplerate: int = Field(default=TARGET_SAMPLERATE_HZ, gt=0)

    scale_audio: bool = Field(default=SCALE_RAW_AUDIO)

    fft_win_length: float = Field(default=FFT_WIN_LENGTH_S, gt=0)

    fft_overlap: float = Field(default=FFT_OVERLAP, ge=0, lt=1)

    max_freq: int = Field(default=MAX_FREQ_HZ, gt=0)

    min_freq: int = Field(default=MIN_FREQ_HZ, gt=0)

    spec_scale: str = Field(default=SPEC_SCALE)

    denoise_spec_avg: bool = DENOISE_SPEC_AVG

    max_scale_spec: bool = MAX_SCALE_SPEC

    duration: Optional[float] = DEFAULT_DURATION

    spec_height: int = SPEC_HEIGHT

    spec_time_period: float = SPEC_TIME_PERIOD


def preprocess_audio_clip(
    clip: data.Clip,
    config: PreprocessingConfig = PreprocessingConfig(),
) -> xr.DataArray:
    """Preprocesses audio clip to generate spectrogram.

    Parameters
    ----------
    clip
        The audio clip to preprocess.
    config
        Configuration for preprocessing.

    Returns
    -------
    xr.DataArray
        Preprocessed spectrogram.

    """
    wav = load_clip_audio(
        clip,
        target_sampling_rate=config.target_samplerate,
        scale=config.scale_audio,
    )

    spec = compute_spectrogram(
        wav,
        fft_win_length=config.fft_win_length,
        fft_overlap=config.fft_overlap,
        max_freq=config.max_freq,
        min_freq=config.min_freq,
        spec_scale=config.spec_scale,
        denoise_spec_avg=config.denoise_spec_avg,
        max_scale_spec=config.max_scale_spec,
    )

    if config.duration is not None:
        spec = adjust_spec_duration(clip, spec, config.duration)

    duration = arrays.get_dim_width(spec, dim="time")
    return ops.resize(
        spec,
        time=int(np.ceil(duration / config.spec_time_period)),
        frequency=config.spec_height,
    )


def adjust_spec_duration(
    clip: data.Clip,
    spec: xr.DataArray,
    duration: float,
) -> xr.DataArray:
    current_duration = clip.end_time - clip.start_time

    if current_duration == duration:
        return spec

    if current_duration > duration:
        return arrays.crop_dim(
            spec,
            dim="time",
            start=clip.start_time,
            stop=clip.start_time + duration,
        )

    return arrays.extend_dim(
        spec,
        dim="time",
        start=clip.start_time,
        stop=clip.start_time + duration,
    )


def load_clip_audio(
    clip: data.Clip,
    target_sampling_rate: int = TARGET_SAMPLERATE_HZ,
    scale: bool = SCALE_RAW_AUDIO,
    dtype: DTypeLike = np.float32,
) -> xr.DataArray:
    wav = audio.load_clip(clip).sel(channel=0).astype(dtype)

    wav = resample_audio(wav, target_sampling_rate, dtype=dtype)

    if scale:
        wav = ops.center(wav)
        wav = ops.scale(wav, 1 / (10e-6 + np.max(np.abs(wav))))

    return wav.astype(dtype)


def resample_audio(
    wav: xr.DataArray,
    target_samplerate: int = TARGET_SAMPLERATE_HZ,
    dtype: DTypeLike = np.float32,
) -> xr.DataArray:
    if "time" not in wav.dims:
        raise ValueError("Audio must have a time dimension")

    time_axis: int = wav.get_axis_num("time")  # type: ignore

    start, stop = arrays.get_dim_range(wav, dim="time")
    step = arrays.get_dim_step(wav, dim="time")
    original_samplerate = int(1 / step)

    if original_samplerate == target_samplerate:
        return wav.astype(dtype)

    gcd = np.gcd(original_samplerate, target_samplerate)
    resampled = resample_poly(
        wav.values,
        target_samplerate // gcd,
        original_samplerate // gcd,
        axis=time_axis,
    )

    resampled_times = np.linspace(
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
                resampled_times,
                samplerate=target_samplerate,
            ),
        },
        attrs=wav.attrs,
    )


def compute_spectrogram(
    wav: xr.DataArray,
    fft_win_length: float = FFT_WIN_LENGTH_S,
    fft_overlap: float = FFT_OVERLAP,
    max_freq: int = MAX_FREQ_HZ,
    min_freq: int = MIN_FREQ_HZ,
    spec_scale: str = SPEC_SCALE,
    denoise_spec_avg: bool = True,
    max_scale_spec: bool = False,
    dtype: DTypeLike = np.float32,
) -> xr.DataArray:
    spec = gen_mag_spectrogram(
        wav,
        window_len=fft_win_length,
        overlap_perc=fft_overlap,
        dtype=dtype,
    )

    spec = arrays.crop_dim(
        spec,
        dim="frequency",
        start=min_freq,
        stop=max_freq,
    ).astype(dtype)

    spec = scale_spectrogram(spec, scale=spec_scale)

    if denoise_spec_avg:
        spec = denoise_spectrogram(spec)

    if max_scale_spec:
        spec = ops.scale(spec, 1 / (10e-6 + np.max(spec)))

    return spec.astype(dtype)


def gen_mag_spectrogram(
    wave: xr.DataArray,
    window_len: float,
    overlap_perc: float,
    dtype: DTypeLike = np.float32,
) -> xr.DataArray:
    start_time, end_time = arrays.get_dim_range(wave, dim="time")
    step = arrays.get_dim_step(wave, dim="time")
    sampling_rate = 1 / step

    hop_len = window_len * (1 - overlap_perc)
    nfft = int(window_len * sampling_rate)
    noverlap = int(overlap_perc * nfft)

    # compute spec
    spec, _ = librosa.core.spectrum._spectrogram(
        y=wave.data,
        power=1,
        n_fft=nfft,
        hop_length=nfft - noverlap,
        center=False,
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
                    end_time - (window_len - hop_len),
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


def denoise_spectrogram(
    spec: xr.DataArray,
) -> xr.DataArray:
    return xr.DataArray(
        data=(spec - spec.mean("time")).clip(0),
        dims=spec.dims,
        coords=spec.coords,
        attrs=spec.attrs,
    )


def scale_spectrogram(
    spec: xr.DataArray,
    scale: str = SPEC_SCALE,
    dtype: DTypeLike = np.float32,
) -> xr.DataArray:
    samplerate = spec.attrs["original_samplerate"]

    if scale == "pcen":
        smoothing_constant = get_pcen_smoothing_constant(samplerate / 10)
        return audio.pcen(
            spec * (2**31),
            smooth=smoothing_constant,
        ).astype(dtype)

    if scale == "log":
        return log_scale(spec, dtype=dtype)

    return spec


def log_scale(
    spec: xr.DataArray,
    dtype: DTypeLike = np.float32,
) -> xr.DataArray:
    samplerate = spec.attrs["original_samplerate"]
    nfft = spec.attrs["nfft"]
    log_scaling = (
        2.0
        * (1.0 / samplerate)
        * (1.0 / (np.abs(np.hanning(nfft)) ** 2).sum())
    )
    return xr.DataArray(
        data=np.log1p(log_scaling * spec).astype(dtype),
        dims=spec.dims,
        coords=spec.coords,
        attrs=spec.attrs,
    )


def get_pcen_smoothing_constant(
    sr: int,
    time_constant: float = 0.4,
    hop_length: int = 512,
) -> float:
    t_frames = time_constant * sr / float(hop_length)
    return (np.sqrt(1 + 4 * t_frames**2) - 1) / (2 * t_frames**2)
