from functools import wraps
from typing import Callable, List, Optional, Tuple

import numpy as np
import xarray as xr
from soundevent import data
from soundevent.geometry import compute_bounds


Augmentation = Callable[[xr.Dataset], xr.Dataset]


AUGMENTATION_PROBABILITY = 0.2
MAX_DELAY = 0.005
STRETCH_SQUEEZE_DELTA = 0.04
MASK_MAX_TIME_PERC: float = 0.05
MASK_MAX_FREQ_PERC: float = 0.10


def maybe_apply(
    augmentation: Callable,
    prob: float = AUGMENTATION_PROBABILITY,
) -> Callable:
    """Apply an augmentation with a given probability."""

    @wraps(augmentation)
    def _augmentation(x):
        if np.random.rand() > prob:
            return x
        return augmentation(x)

    return _augmentation


def select_random_subclip(
    train_example: xr.Dataset,
    duration: Optional[float] = None,
    proportion: float = 0.9,
) -> xr.Dataset:
    """Select a random subclip from a clip."""

    time_coords = train_example.coords["time"]

    start_time = time_coords.attrs.get("min", time_coords.min())
    end_time = time_coords.attrs.get("max", time_coords.max())

    if duration is None:
        duration = (end_time - start_time) * proportion

    start_time = np.random.uniform(start_time, end_time - duration)
    return train_example.sel(time=slice(start_time, start_time + duration))


def combine_audio(
    audio1: xr.DataArray,
    audio2: xr.DataArray,
    alpha: Optional[float] = None,
    min_alpha: float = 0.3,
    max_alpha: float = 0.7,
) -> xr.DataArray:
    """Combine two audio clips."""

    if alpha is None:
        alpha = np.random.uniform(min_alpha, max_alpha)

    return alpha * audio1 + (1 - alpha) * audio2.data


# def random_mix(
#     audio: xr.DataArray,
#     clip: data.ClipAnnotation,
#     provider: Optional[ClipProvider] = None,
#     alpha: Optional[float] = None,
#     min_alpha: float = 0.3,
#     max_alpha: float = 0.7,
#     join_annotations: bool = True,
# ) -> Tuple[xr.DataArray, data.ClipAnnotation]:
#     """Mix two audio clips."""
#     if provider is None:
#         raise ValueError("No audio provider given.")
#
#     try:
#         other_audio, other_clip = provider(clip)
#     except (StopIteration, ValueError):
#         raise ValueError("No more audio sources available.")
#
#     new_audio = combine_audio(
#         audio,
#         other_audio,
#         alpha=alpha,
#         min_alpha=min_alpha,
#         max_alpha=max_alpha,
#     )
#
#     if join_annotations:
#         clip = clip.model_copy(
#             update=dict(
#                 sound_events=clip.sound_events + other_clip.sound_events,
#             )
#         )
#
#     return new_audio, clip


def add_echo(
    train_example: xr.Dataset,
    delay: Optional[float] = None,
    alpha: Optional[float] = None,
    min_alpha: float = 0.0,
    max_alpha: float = 1.0,
    max_delay: float = MAX_DELAY,
) -> xr.Dataset:
    """Add a delay to the audio."""
    if delay is None:
        delay = np.random.uniform(0, max_delay)

    if alpha is None:
        alpha = np.random.uniform(min_alpha, max_alpha)

    spec = train_example["spectrogram"]

    time_coords = spec.coords["time"]
    start_time = time_coords.attrs["min"]
    end_time = time_coords.attrs["max"]
    step = (end_time - start_time) / time_coords.size

    spec_delay = spec.shift(time=int(delay / step), fill_value=0)

    return train_example.assign(spectrogram=spec + alpha * spec_delay)


def scale_volume(
    train_example: xr.Dataset,
    factor: Optional[float] = None,
    max_scaling: float = 2,
    min_scaling: float = 0,
) -> xr.Dataset:
    """Scale the volume of a spectrogram."""
    if factor is None:
        factor = np.random.uniform(min_scaling, max_scaling)

    return train_example.assign(
        spectrogram=train_example["spectrogram"] * factor
    )


def warp_spectrogram(
    train_example: xr.Dataset,
    factor: Optional[float] = None,
    delta: float = STRETCH_SQUEEZE_DELTA,
) -> xr.Dataset:
    """Warp a spectrogram."""
    if factor is None:
        factor = np.random.uniform(1 - delta, 1 + delta)

    time_coords = train_example.coords["time"]
    start_time = time_coords.attrs["min"]
    end_time = time_coords.attrs["max"]
    duration = end_time - start_time

    new_time = np.linspace(
        start_time,
        start_time + duration * factor,
        train_example.time.size,
    )

    return train_example.interp(time=new_time)


def mask_axis(
    train_example: xr.Dataset,
    dim: str,
    start: float,
    end: float,
    mask_all: bool = False,
    mask_value: float = 0,
) -> xr.Dataset:
    if dim not in train_example.dims:
        raise ValueError(f"Axis {dim} not found in array")

    coord = train_example.coords[dim]
    condition = (coord < start) | (coord > end)

    if mask_all:
        return train_example.where(condition, other=mask_value)

    return train_example.assign(
        spectrogram=train_example.spectrogram.where(
            condition, other=mask_value
        )
    )


def mask_time(
    train_example: xr.Dataset,
    max_time_mask: float = MASK_MAX_TIME_PERC,
    max_num_masks: int = 3,
) -> xr.Dataset:
    """Mask a random section of the time axis."""

    num_masks = np.random.randint(1, max_num_masks + 1)

    time_coord = train_example.coords["time"]
    start_time = time_coord.attrs.get("min", time_coord.min())
    end_time = time_coord.attrs.get("max", time_coord.max())

    for _ in range(num_masks):
        mask_size = np.random.uniform(0, max_time_mask)
        start = np.random.uniform(start_time, end_time - mask_size)
        end = start + mask_size
        train_example = mask_axis(train_example, "time", start, end)

    return train_example


def mask_frequency(
    train_example: xr.Dataset,
    max_freq_mask: float = MASK_MAX_FREQ_PERC,
    max_num_masks: int = 3,
) -> xr.Dataset:
    """Mask a random section of the frequency axis."""

    num_masks = np.random.randint(1, max_num_masks + 1)

    freq_coord = train_example.coords["frequency"]
    min_freq = freq_coord.min()
    max_freq = freq_coord.max()

    for _ in range(num_masks):
        mask_size = np.random.uniform(0, max_freq_mask)
        start = np.random.uniform(min_freq, max_freq - mask_size)
        end = start + mask_size
        train_example = mask_axis(train_example, "frequency", start, end)

    return train_example


AUGMENTATIONS: List[Augmentation] = [
    select_random_subclip,
    add_echo,
    scale_volume,
    mask_time,
    mask_frequency,
]
