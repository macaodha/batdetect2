"""Module containing functions for preprocessing audio clips."""

import random
from typing import List, Optional, Tuple

import librosa
import librosa.core.spectrum
import numpy as np
import xarray as xr
from numpy.typing import DTypeLike
from scipy.signal import resample_poly
from soundevent import audio, data

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


def preprocess_audio_clip(
    clip: data.Clip,
    target_sampling_rate: int = TARGET_SAMPLERATE_HZ,
    scale_audio: bool = SCALE_RAW_AUDIO,
    fft_win_length: float = FFT_WIN_LENGTH_S,
    fft_overlap: float = FFT_OVERLAP,
    max_freq: int = MAX_FREQ_HZ,
    min_freq: int = MIN_FREQ_HZ,
    spec_scale: str = SPEC_SCALE,
    denoise_spec_avg: bool = True,
    max_scale_spec: bool = False,
    duration: Optional[float] = DEFAULT_DURATION,
    spec_height: int = SPEC_HEIGHT,
    spec_time_period: float = SPEC_TIME_PERIOD,
) -> xr.DataArray:
    """Preprocesses audio clip to generate spectrogram.

    Parameters
    ----------
    clip
        The audio clip to preprocess.
    target_sampling_rate
        Target sampling rate for the audio. If the audio has a different
        sampling rate, it will be resampled to this rate.
    scale_audio
        Whether to scale the audio amplitudes to a range of [-1, 1].
        By default, the audio is not scaled.
    fft_win_length
        Length of the FFT window in seconds.
    fft_overlap
        Amount of overlap between FFT windows as a fraction of the window
        length.
    max_freq
        Maximum frequency for spectrogram. Anything above this frequency will
        be cropped.
    min_freq
        Minimum frequency for spectrogram. Anything below this frequency will
        be cropped.
    spec_scale
        Scaling method for the spectrogram. Can be "pcen", "log" or
        "amplitude".
    denoise_spec_avg
        Whether to denoise the spectrogram. Denoising is done by subtracting
        the average of the spectrogram from the spectrogram and clipping
        negative values to 0.
    max_scale_spec
        Whether to max scale the spectrogram. Max scaling is done by dividing
        the spectrogram by its maximum value thus scaling values to [0, 1].
    duration
        Duration of the spectrogram in seconds. If the clip duration is
        different from this value, the spectrogram will be cropped or extended
        to match this duration. If None, the spectrogram will have the same
        duration as the clip.
    spec_height
        Number of frequency bins for the spectrogram. This is the height of
        the final spectrogram.
    spec_time_period
        Time period for each spectrogram bin in seconds. The spectrogram array
        will be resized (using bilinear interpolation) to have this time
        period.

    Returns
    -------
    xr.DataArray
        Preprocessed spectrogram.

    """
    wav = load_clip_audio(
        clip,
        target_sampling_rate=target_sampling_rate,
        scale=scale_audio,
    )

    wav = wav.assign_attrs(
        recording_id=str(wav.attrs["recording_id"]),
        clip_id=str(wav.attrs["clip_id"]),
        path=str(wav.attrs["path"]),
    )

    spec = compute_spectrogram(
        wav,
        fft_win_length=fft_win_length,
        fft_overlap=fft_overlap,
        max_freq=max_freq,
        min_freq=min_freq,
        spec_scale=spec_scale,
        denoise_spec_avg=denoise_spec_avg,
        max_scale_spec=max_scale_spec,
    )

    if duration is not None:
        spec = adjust_spec_duration(clip, spec, duration)

    duration = get_dim_width(spec, dim="time")
    return resize_spectrogram(
        spec,
        time_bins=int(np.ceil(duration / spec_time_period)),
        freq_bins=spec_height,
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
        return crop_axis(
            spec,
            dim="time",
            start=clip.start_time,
            end=clip.start_time + duration,
        )

    return extend_axis(
        spec,
        dim="time",
        start=clip.start_time,
        end=clip.start_time + duration,
    )


def load_clip_audio(
    clip: data.Clip,
    target_sampling_rate: int = TARGET_SAMPLERATE_HZ,
    scale: bool = SCALE_RAW_AUDIO,
    dtype: DTypeLike = np.float32,
) -> xr.DataArray:
    wav = audio.load_clip(clip).sel(channel=0)

    wav = resample_audio(wav, target_sampling_rate, dtype=dtype)

    if scale:
        wav = scale_audio(wav)

    wav.coords["time"] = wav.time.assign_attrs(
        unit="s",
        long_name="Seconds since start of recording",
        min=clip.start_time,
        max=clip.end_time,
    )

    return wav


def resample_audio(
    wav: xr.DataArray,
    target_samplerate: int = TARGET_SAMPLERATE_HZ,
    dtype: DTypeLike = np.float32,
) -> xr.DataArray:
    if "samplerate" not in wav.attrs:
        raise ValueError("Audio must have a 'samplerate' attribute")

    if "time" not in wav.dims:
        raise ValueError("Audio must have a time dimension")

    time_axis: int = wav.get_axis_num("time")  # type: ignore
    original_samplerate = wav.attrs["samplerate"]

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
        wav.time[0],
        wav.time[-1],
        len(resampled),
        endpoint=False,
        dtype=dtype,
    )

    return xr.DataArray(
        data=resampled.astype(dtype),
        dims=wav.dims,
        coords={
            **wav.coords,
            "time": resampled_times,
        },
        attrs={
            **wav.attrs,
            "samplerate": target_samplerate,
        },
    )


def scale_audio(
    audio: xr.DataArray,
    eps: float = 10e-6,
) -> xr.DataArray:
    audio = audio - audio.mean()
    return audio / np.add(np.abs(audio).max(), eps, dtype=audio.dtype)


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

    spec = crop_axis(
        spec,
        dim="frequency",
        start=min_freq,
        end=max_freq,
    )

    spec = scale_spectrogram(spec, scale=spec_scale)

    if denoise_spec_avg:
        spec = denoise_spectrogram(spec)

    if max_scale_spec:
        spec = max_scale_spectrogram(spec)

    return spec


def crop_axis(
    arr: xr.DataArray,
    dim: str,
    start: float,
    end: float,
    right_closed: bool = False,
    left_closed: bool = True,
    eps: float = 10e-6,
) -> xr.DataArray:
    coord = arr.coords[dim]

    if not all(attr in coord.attrs for attr in ["min", "max"]):
        raise ValueError(
            f"Coordinate '{dim}' must have 'min' and 'max' attributes"
        )

    current_min = coord.attrs["min"]
    current_max = coord.attrs["max"]

    if start < current_min or end > current_max:
        raise ValueError(
            f"Cannot select axis '{dim}' from {start} to {end}. "
            f"Axis range is {current_min} to {current_max}"
        )

    slice_end = end
    if not right_closed:
        slice_end = end - eps

    slice_start = start
    if not left_closed:
        slice_start = start + eps

    arr = arr.sel({dim: slice(slice_start, slice_end)})

    arr.coords[dim].attrs.update(
        min=start,
        max=end,
    )

    return arr


def extend_axis(
    arr: xr.DataArray,
    dim: str,
    start: float,
    end: float,
    fill_value: float = 0,
) -> xr.DataArray:
    coord = arr.coords[dim]

    if not all(attr in coord.attrs for attr in ["min", "max", "period"]):
        raise ValueError(
            f"Coordinate '{dim}' must have 'min', 'max' and 'period' attributes"
            " to extend axis"
        )

    current_min = coord.attrs["min"]
    current_max = coord.attrs["max"]
    period = coord.attrs["period"]

    coords = coord.data

    if start < current_min:
        new_coords = np.arange(
            current_min,
            start,
            -period,
            dtype=coord.dtype,
        )[1:][::-1]
        coords = np.concatenate([new_coords, coords])

    if end > current_max:
        new_coords = np.arange(
            current_max,
            end,
            period,
            dtype=coord.dtype,
        )[1:]
        coords = np.concatenate([coords, new_coords])

    arr = arr.reindex(
        {dim: coords},
        fill_value=fill_value,  # type: ignore
    )

    arr.coords[dim].attrs.update(
        min=start,
        max=end,
    )

    return arr


def gen_mag_spectrogram(
    audio: xr.DataArray,
    window_len: float,
    overlap_perc: float,
    dtype: DTypeLike = np.float32,
) -> xr.DataArray:
    sampling_rate = audio.attrs["samplerate"]
    hop_len = window_len * (1 - overlap_perc)
    nfft = int(window_len * sampling_rate)
    noverlap = int(overlap_perc * nfft)
    start_time = audio.time.attrs["min"]
    end_time = audio.time.attrs["max"]

    # compute spec
    spec, _ = librosa.core.spectrum._spectrogram(
        y=audio.data,
        power=1,
        n_fft=nfft,
        hop_length=nfft - noverlap,
        center=False,
    )

    spec = xr.DataArray(
        data=spec.astype(dtype),
        dims=["frequency", "time"],
        coords={
            "frequency": np.linspace(
                0,
                sampling_rate / 2,
                spec.shape[0],
                endpoint=False,
                dtype=dtype,
            ),
            "time": np.linspace(
                start_time,
                end_time - (window_len - hop_len),
                spec.shape[1],
                endpoint=False,
                dtype=dtype,
            ),
        },
        attrs={
            **audio.attrs,
            "nfft": nfft,
            "noverlap": noverlap,
        },
    )

    # Add metadata to coordinates
    spec.coords["time"].attrs.update(
        unit="s",
        long_name="Time",
        min=start_time,
        max=end_time - (window_len - hop_len),
        period=(nfft - noverlap) / sampling_rate,
    )
    spec.coords["frequency"].attrs.update(
        unit="Hz",
        long_name="Frequency",
        period=(sampling_rate / nfft),
        min=0,
        max=sampling_rate / 2,
    )

    return spec


def denoise_spectrogram(
    spec: xr.DataArray,
) -> xr.DataArray:
    return xr.DataArray(
        data=(spec - spec.mean("time")).clip(0),
        dims=spec.dims,
        coords=spec.coords,
        attrs={
            **spec.attrs,
            "denoised": 1,
        },
    )


def scale_spectrogram(
    spec: xr.DataArray,
    scale: str = SPEC_SCALE,
    dtype: DTypeLike = np.float32,
) -> xr.DataArray:
    if scale == "pcen":
        return pcen(spec, dtype=dtype)

    if scale == "log":
        return log_scale(spec, dtype=dtype)

    return spec


def log_scale(
    spec: xr.DataArray,
    dtype: DTypeLike = np.float32,
) -> xr.DataArray:
    nfft = spec.attrs["nfft"]
    sampling_rate = spec.attrs["samplerate"]
    log_scaling = (
        2.0
        * (1.0 / sampling_rate)
        * (1.0 / (np.abs(np.hanning(nfft)) ** 2).sum())
    )
    return xr.DataArray(
        data=np.log1p(log_scaling * spec).astype(dtype),
        dims=spec.dims,
        coords=spec.coords,
        attrs={
            **spec.attrs,
            "scale": "log",
        },
    )


def pcen(spec: xr.DataArray, dtype: DTypeLike = np.float32) -> xr.DataArray:
    sampling_rate = spec.attrs["samplerate"]
    data = librosa.pcen(
        spec.data * (2**31),
        sr=sampling_rate / 10,
    )
    return xr.DataArray(
        data=data.astype(dtype),
        dims=spec.dims,
        coords=spec.coords,
        attrs={
            **spec.attrs,
            "scale": "pcen",
        },
    )


def max_scale_spectrogram(spec: xr.DataArray, eps=10e-6) -> xr.DataArray:
    return xr.DataArray(
        data=spec / np.add(spec.max(), eps, dtype=spec.dtype),
        dims=spec.dims,
        coords=spec.coords,
        attrs={
            **spec.attrs,
            "max_scaled": 1,
        },
    )


def resize_spectrogram(
    spec: xr.DataArray,
    time_bins: int,
    freq_bins: int,
) -> xr.DataArray:
    new_times = np.linspace(
        spec.time[0],
        spec.time[-1],
        time_bins,
        dtype=spec.time.dtype,
        endpoint=True,
    )
    new_frequencies = np.linspace(
        spec.frequency[0],
        spec.frequency[-1],
        freq_bins,
        dtype=spec.frequency.dtype,
        endpoint=True,
    )

    return spec.interp(
        coords=dict(
            time=new_times,
            frequency=new_frequencies,
        ),
        method="linear",
    )


def get_dim_width(arr: xr.DataArray, dim: str) -> float:
    coord = arr.coords[dim]
    attrs = coord.attrs
    if "min" in attrs and "max" in attrs:
        return attrs["max"] - attrs["min"]

    coord_min = coord.min()
    coord_max = coord.max()
    return float(coord_max - coord_min)


class RandomClipProvider:
    def __init__(
        self,
        clip_annotations: List[data.ClipAnnotation],
        target_sampling_rate: int = TARGET_SAMPLERATE_HZ,
        scale_audio: bool = SCALE_RAW_AUDIO,
    ):
        self.target_sampling_rate = target_sampling_rate
        self.scale_audio = scale_audio
        self.clip_annotations = clip_annotations

    def get_next_clip(self, clip: data.ClipAnnotation) -> data.ClipAnnotation:
        tries = 0
        while True:
            random_clip = random.choice(self.clip_annotations)

            if random_clip.clip != clip.clip:
                return random_clip

            tries += 1
            if tries > 4:
                raise ValueError("Could not find a different clip")

    def __call__(
        self,
        clip: data.ClipAnnotation,
    ) -> Tuple[xr.DataArray, data.ClipAnnotation]:
        random_clip = self.get_next_clip(clip)

        wav = load_clip_audio(
            random_clip.clip,
            target_sampling_rate=self.target_sampling_rate,
            scale=self.scale_audio,
        )

        return wav, random_clip
