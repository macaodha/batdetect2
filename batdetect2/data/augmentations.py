from functools import wraps
from typing import Callable, List, Optional, Tuple

import numpy as np
import xarray as xr
from soundevent import data
from soundevent.geometry import compute_bounds

ClipAugmentation = Callable[[data.ClipAnnotation], data.ClipAnnotation]
AudioAugmentation = Callable[
    [xr.DataArray, data.ClipAnnotation],
    Tuple[xr.DataArray, data.ClipAnnotation],
]
SpecAugmentation = Callable[
    [xr.DataArray, data.ClipAnnotation],
    Tuple[xr.DataArray, data.ClipAnnotation],
]

ClipProvider = Callable[
    [data.ClipAnnotation], Tuple[xr.DataArray, data.ClipAnnotation]
]
"""A function that provides some clip and its annotation.

Usually this function loads a random clip from a dataset. Takes
as input a clip annotation that can be used to filter the clips
to load (in case you want to avoid loading the same clip multiple times).
"""


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
    clip_annotation: data.ClipAnnotation,
    duration: Optional[float] = None,
    proportion: float = 0.9,
) -> data.ClipAnnotation:
    """Select a random subclip from a clip."""
    clip = clip_annotation.clip

    if duration is None:
        clip_duration = clip.end_time - clip.start_time
        duration = clip_duration * proportion

    start_time = np.random.uniform(clip.start_time, clip.end_time - duration)
    return clip_annotation.model_copy(
        update=dict(
            clip=clip.model_copy(
                update=dict(
                    start_time=start_time,
                    end_time=start_time + duration,
                )
            )
        )
    )


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


def random_mix(
    audio: xr.DataArray,
    clip: data.ClipAnnotation,
    provider: Optional[ClipProvider] = None,
    alpha: Optional[float] = None,
    min_alpha: float = 0.3,
    max_alpha: float = 0.7,
    join_annotations: bool = True,
) -> Tuple[xr.DataArray, data.ClipAnnotation]:
    """Mix two audio clips."""
    if provider is None:
        raise ValueError("No audio provider given.")

    try:
        other_audio, other_clip = provider(clip)
    except (StopIteration, ValueError):
        raise ValueError("No more audio sources available.")

    new_audio = combine_audio(
        audio,
        other_audio,
        alpha=alpha,
        min_alpha=min_alpha,
        max_alpha=max_alpha,
    )

    if join_annotations:
        clip = clip.model_copy(
            update=dict(
                sound_events=clip.sound_events + other_clip.sound_events,
            )
        )

    return new_audio, clip


def add_echo(
    audio: xr.DataArray,
    clip: data.ClipAnnotation,
    delay: Optional[float] = None,
    alpha: Optional[float] = None,
    min_alpha: float = 0.0,
    max_alpha: float = 1.0,
    max_delay: float = MAX_DELAY,
) -> Tuple[xr.DataArray, data.ClipAnnotation]:
    """Add a delay to the audio."""
    if delay is None:
        delay = np.random.uniform(0, max_delay)

    if alpha is None:
        alpha = np.random.uniform(min_alpha, max_alpha)

    samplerate = audio.attrs["samplerate"]
    offset = int(delay * samplerate)

    # NOTE: We use the copy method to avoid modifying the original audio
    # data.
    new_audio = audio.copy()
    new_audio[offset:] += alpha * audio.data[:-offset]
    return new_audio, clip


def scale_volume(
    spec: xr.DataArray,
    clip: data.ClipAnnotation,
    factor: Optional[float] = None,
    max_scaling: float = 2,
    min_scaling: float = 0,
) -> Tuple[xr.DataArray, data.ClipAnnotation]:
    """Scale the volume of a spectrogram."""
    if factor is None:
        factor = np.random.uniform(min_scaling, max_scaling)

    return spec * factor, clip


def scale_sound_event_annotation(
    sound_event_annotation: data.SoundEventAnnotation,
    time_factor: float = 1,
    frequency_factor: float = 1,
) -> data.SoundEventAnnotation:
    sound_event = sound_event_annotation.sound_event
    geometry = sound_event.geometry

    if geometry is None:
        return sound_event_annotation

    start_time, low_freq, end_time, high_freq = compute_bounds(geometry)
    new_geometry = data.BoundingBox(
        coordinates=[
            start_time * time_factor,
            low_freq * frequency_factor,
            end_time * time_factor,
            high_freq * frequency_factor,
        ]
    )

    return sound_event_annotation.model_copy(
        update=dict(
            sound_event=sound_event.model_copy(
                update=dict(
                    geometry=new_geometry,
                )
            )
        )
    )


def warp_spectrogram(
    spec: xr.DataArray,
    clip: data.ClipAnnotation,
    factor: Optional[float] = None,
    delta: float = STRETCH_SQUEEZE_DELTA,
) -> Tuple[xr.DataArray, data.ClipAnnotation]:
    """Warp a spectrogram."""
    if factor is None:
        factor = np.random.uniform(1 - delta, 1 + delta)

    start_time = clip.clip.start_time
    end_time = clip.clip.end_time
    duration = end_time - start_time
    new_time = np.linspace(
        start_time,
        start_time + duration * factor,
        spec.time.size,
    )

    scaled_spec = spec.interp(
        time=new_time,
        method="linear",
        kwargs={"fill_value": 0},
    )
    scaled_spec.coords["time"] = spec.time

    scaled_clip = clip.model_copy(
        update=dict(
            sound_events=[
                scale_sound_event_annotation(
                    sound_event_annotation,
                    time_factor=1 / factor,
                )
                for sound_event_annotation in clip.sound_events
            ]
        )
    )
    return scaled_spec, scaled_clip


def mask_axis(
    array: xr.DataArray,
    axis: str,
    start: float,
    end: float,
    mask_value: float = 0,
) -> xr.DataArray:
    if axis not in array.dims:
        raise ValueError(f"Axis {axis} not found in array")

    coord = array[axis]
    return array.where((coord < start) | (coord > end), mask_value)


def mask_time(
    spec: xr.DataArray,
    clip: data.ClipAnnotation,
    max_time_mask: float = MASK_MAX_TIME_PERC,
    max_num_masks: int = 3,
) -> Tuple[xr.DataArray, data.ClipAnnotation]:
    """Mask a random section of the time axis."""

    num_masks = np.random.randint(1, max_num_masks + 1)
    for _ in range(num_masks):
        mask_size = np.random.uniform(0, max_time_mask)
        start = np.random.uniform(0, spec.time[-1] - mask_size)
        end = start + mask_size
        spec = mask_axis(spec, "time", start, end)

    return spec, clip


def mask_frequency(
    spec: xr.DataArray,
    clip: data.ClipAnnotation,
    max_freq_mask: float = MASK_MAX_FREQ_PERC,
    max_num_masks: int = 3,
) -> Tuple[xr.DataArray, data.ClipAnnotation]:
    """Mask a random section of the frequency axis."""

    num_masks = np.random.randint(1, max_num_masks + 1)
    for _ in range(num_masks):
        mask_size = np.random.uniform(0, max_freq_mask)
        start = np.random.uniform(0, spec.frequency[-1] - mask_size)
        end = start + mask_size
        spec = mask_axis(spec, "frequency", start, end)

    return spec, clip


CLIP_AUGMENTATIONS: List[ClipAugmentation] = [
    select_random_subclip,
]

AUDIO_AUGMENTATIONS: List[AudioAugmentation] = [
    add_echo,
    random_mix,
]

SPEC_AUGMENTATIONS: List[SpecAugmentation] = [
    scale_volume,
    warp_spectrogram,
    mask_time,
    mask_frequency,
]
