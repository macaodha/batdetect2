from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter
from soundevent import data, geometry

__all__ = [
    "generate_heatmaps",
]

PositionFn = Callable[[data.SoundEvent], Tuple[float, float]]
"""Convert a sound event to a single position in time-frequency space."""

SizeFn = Callable[[data.SoundEvent, float, float], np.ndarray]
"""Compute the size of a sound event in time-frequency space.

The time and frequency scales are provided as arguments to allow
modifying the size of the sound event based on the spectrogram
parameters.
"""

LabelFn = Callable[[data.SoundEventAnnotation], Optional[str]]
"""Convert a sound event annotation to a label.

When the label is None, this indicates that the sound event does not
belong to any of the classes of interest.
"""

TARGET_SIGMA = 3.0


GENERIC_LABEL = "__UNKNOWN__"


def get_lower_left_position(
    sound_event: data.SoundEvent,
) -> Tuple[float, float]:
    if sound_event.geometry is None:
        raise ValueError("Sound event has no geometry.")

    start_time, low_freq, _, _ = geometry.compute_bounds(sound_event.geometry)
    return start_time, low_freq


def get_bbox_size(
    sound_event: data.SoundEvent,
    time_scale: float = 1.0,
    frequency_scale: float = 1.0,
) -> np.ndarray:
    if sound_event.geometry is None:
        raise ValueError("Sound event has no geometry.")

    start_time, low_freq, end_time, high_freq = geometry.compute_bounds(
        sound_event.geometry
    )

    return np.array(
        [
            time_scale * (end_time - start_time),
            frequency_scale * (high_freq - low_freq),
        ]
    )


def _tag_key(tag: data.Tag) -> Tuple[str, str]:
    return (tag.key, tag.value)


def set_value_at_position(
    array: xr.DataArray,
    value: Any,
    **query,
) -> xr.DataArray:
    dims = {dim: n for n, dim in enumerate(array.dims)}
    indexer: List[Union[slice, int]] = [slice(None) for _ in range(array.ndim)]

    for key, coord in query.items():
        if key not in dims:
            raise ValueError(f"Dimension {key} not found in array.")

        coordinates = array.indexes[key]
        indexer[dims[key]] = coordinates.get_loc(coordinates.asof(coord))

    if isinstance(value, (tuple, list)):
        value = np.array(value)

    array.data[tuple(indexer)] = value
    return array


def generate_heatmaps(
    clip_annotation: data.ClipAnnotation,
    spec: xr.DataArray,
    num_classes: int = 1,
    label_fn: LabelFn = lambda _: None,
    target_sigma: float = TARGET_SIGMA,
    size_fn: SizeFn = get_bbox_size,
    position_fn: PositionFn = get_lower_left_position,
    class_labels: Optional[List[str]] = None,
    dtype=np.float32,
) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    if class_labels is None:
        class_labels = [str(i) for i in range(num_classes)]

    if len(class_labels) != num_classes:
        raise ValueError(
            "Number of class labels must match the number of classes."
        )

    shape = dict(zip(spec.dims, spec.shape))

    if "time" not in shape or "frequency" not in shape:
        raise ValueError(
            "Spectrogram must have time and frequency dimensions."
        )

    time_duration = spec.time.attrs["max"] - spec.time.attrs["min"]
    freq_bandwidth = spec.frequency.attrs["max"] - spec.frequency.attrs["min"]

    # Compute the size factors
    time_scale = 1 / time_duration
    frequency_scale = 1 / freq_bandwidth

    # Initialize heatmaps
    detection_heatmap = xr.zeros_like(spec, dtype=dtype)
    class_heatmap = xr.DataArray(
        data=np.zeros((num_classes, *spec.shape), dtype=dtype),
        dims=["category", *spec.dims],
        coords={
            "category": class_labels,
            **spec.coords,
        },
    )
    size_heatmap = xr.DataArray(
        data=np.zeros((2, *spec.shape), dtype=dtype),
        dims=["dimension", *spec.dims],
        coords={
            "dimension": ["width", "height"],
            **spec.coords,
        },
    )

    for sound_event_annotation in clip_annotation.sound_events:
        # Get the position of the sound event
        time, frequency = position_fn(sound_event_annotation.sound_event)

        # Set 1.0 at the position of the sound event in the detection heatmap
        detection_heatmap = set_value_at_position(
            detection_heatmap,
            1.0,
            time=time,
            frequency=frequency,
        )

        # Set the size of the sound event at the position in the size heatmap
        size = size_fn(
            sound_event_annotation.sound_event,
            time_scale,
            frequency_scale,

        )
        size_heatmap = set_value_at_position(
            size_heatmap,
            size,
            time=time,
            frequency=frequency,
        )

        # Get the label id for the sound event
        label = label_fn(sound_event_annotation)

        if label is None or label not in class_labels:
            # If the label is None or not in the class labels, we skip the
            # sound event
            continue

        # Set 1.0 at the position and category of the sound event in the class
        # heatmap
        class_heatmap = set_value_at_position(
            class_heatmap,
            1.0,
            time=time,
            frequency=frequency,
            category=label,
        )

    # Apply gaussian filters
    detection_heatmap = xr.apply_ufunc(
        gaussian_filter,
        detection_heatmap,
        target_sigma,
    )

    class_heatmap = class_heatmap.groupby("category").map(
        gaussian_filter,  # type: ignore
        args=(target_sigma,),
    )

    # Normalize heatmaps
    detection_heatmap = (
        detection_heatmap / detection_heatmap.max(dim=["time", "frequency"])
    ).fillna(0.0)

    class_heatmap = (
        class_heatmap / class_heatmap.max(dim=["time", "frequency"])
    ).fillna(0.0)

    return detection_heatmap, class_heatmap, size_heatmap


class Labeler:
    def __init__(self, tags: List[data.Tag]):
        """Create a labeler from a list of tags.

        Each tag is assigned a unique label. The labeler can then be used
        to convert sound event annotations to labels.
        """
        self.tags = tags
        self._label_map = {_tag_key(tag): i for i, tag in enumerate(tags)}
        self._inverse_label_map = {v: k for k, v in self._label_map.items()}

    def __call__(
        self, sound_event_annotation: data.SoundEventAnnotation
    ) -> Optional[int]:
        for tag in sound_event_annotation.tags:
            key = _tag_key(tag)
            if key in self._label_map:
                return self._label_map[key]

        return None
