from typing import Optional

import numpy as np
import xarray as xr
from numpy.typing import DTypeLike
from pydantic import Field
from scipy.signal import resample, resample_poly
from soundevent import arrays, audio, data
from soundevent.arrays import operations as ops

from batdetect2.configs import BaseConfig

TARGET_SAMPLERATE_HZ = 256_000
SCALE_RAW_AUDIO = False
DEFAULT_DURATION = None


class ResampleConfig(BaseConfig):
    samplerate: int = Field(default=TARGET_SAMPLERATE_HZ, gt=0)
    mode: str = "poly"


class AudioConfig(BaseConfig):
    resample: Optional[ResampleConfig] = Field(default_factory=ResampleConfig)
    scale: bool = SCALE_RAW_AUDIO
    center: bool = True
    duration: Optional[float] = DEFAULT_DURATION


def load_file_audio(
    path: data.PathLike,
    config: Optional[AudioConfig] = None,
    audio_dir: Optional[data.PathLike] = None,
    dtype: DTypeLike = np.float32,  # type: ignore
) -> xr.DataArray:
    recording = data.Recording.from_file(path)
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
    config = config or AudioConfig()

    wav = (
        audio.load_clip(clip, audio_dir=audio_dir).sel(channel=0).astype(dtype)
    )

    if config.duration is not None:
        wav = adjust_audio_duration(wav, duration=config.duration)

    if config.resample:
        wav = resample_audio(
            wav,
            samplerate=config.resample.samplerate,
            dtype=dtype,
        )

    if config.center:
        wav = ops.center(wav)

    if config.scale:
        wav = ops.scale(wav, 1 / (10e-6 + np.max(np.abs(wav))))

    return wav.astype(dtype)


def adjust_audio_duration(
    wave: xr.DataArray,
    duration: float,
) -> xr.DataArray:
    start_time, end_time = arrays.get_dim_range(wave, dim="time")
    current_duration = end_time - start_time

    if current_duration == duration:
        return wave

    if current_duration > duration:
        return arrays.crop_dim(
            wave,
            dim="time",
            start=start_time,
            stop=start_time + duration,
        )

    return arrays.extend_dim(
        wave,
        dim="time",
        start=start_time,
        stop=start_time + duration,
    )


def resample_audio(
    wav: xr.DataArray,
    samplerate: int = TARGET_SAMPLERATE_HZ,
    mode: str = "poly",
    dtype: DTypeLike = np.float32,  # type: ignore
) -> xr.DataArray:
    if "time" not in wav.dims:
        raise ValueError("Audio must have a time dimension")

    time_axis: int = wav.get_axis_num("time")  # type: ignore
    step = arrays.get_dim_step(wav, dim="time")
    original_samplerate = int(1 / step)

    if original_samplerate == samplerate:
        return wav.astype(dtype)

    if mode == "poly":
        resampled = resample_audio_poly(
            wav,
            sr_orig=original_samplerate,
            sr_new=samplerate,
            axis=time_axis,
        )
    elif mode == "fourier":
        resampled = resample_audio_fourier(
            wav,
            sr_orig=original_samplerate,
            sr_new=samplerate,
            axis=time_axis,
        )
    else:
        raise NotImplementedError(f"Resampling mode '{mode}' not implemented")

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
        attrs=wav.attrs,
    )


def resample_audio_poly(
    array: xr.DataArray,
    sr_orig: int,
    sr_new: int,
    axis: int = -1,
) -> np.ndarray:
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
    ratio = sr_new / sr_orig
    return resample(array, int(array.shape[axis] * ratio), axis=axis)  # type: ignore
