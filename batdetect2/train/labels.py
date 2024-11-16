from typing import Sequence, Tuple

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter
from soundevent import arrays, data, geometry
from soundevent.geometry.operations import Positions
from soundevent.types import ClassMapper

__all__ = [
    "ClassMapper",
    "generate_heatmaps",
]


TARGET_SIGMA = 3.0


def generate_heatmaps(
    sound_events: Sequence[data.SoundEventAnnotation],
    spec: xr.DataArray,
    class_mapper: ClassMapper,
    target_sigma: float = TARGET_SIGMA,
    position: Positions = "bottom-left",
    time_scale: float = 1.0,
    frequency_scale: float = 1.0,
    dtype=np.float32,
) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    shape = dict(zip(spec.dims, spec.shape))

    if "time" not in shape or "frequency" not in shape:
        raise ValueError(
            "Spectrogram must have time and frequency dimensions."
        )

    # Initialize heatmaps
    detection_heatmap = xr.zeros_like(spec, dtype=dtype)
    class_heatmap = xr.DataArray(
        data=np.zeros((class_mapper.num_classes, *spec.shape), dtype=dtype),
        dims=["category", *spec.dims],
        coords={
            "category": class_mapper.class_labels,
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

    for sound_event_annotation in sound_events:
        geom = sound_event_annotation.sound_event.geometry
        if geom is None:
            continue

        # Get the position of the sound event
        time, frequency = geometry.get_geometry_point(geom, position=position)

        # Set 1.0 at the position of the sound event in the detection heatmap
        detection_heatmap = arrays.set_value_at_pos(
            detection_heatmap,
            1.0,
            time=time,
            frequency=frequency,
        )

        # Set the size of the sound event at the position in the size heatmap
        start_time, low_freq, end_time, high_freq = geometry.compute_bounds(
            geom
        )
        size = np.array(
            [
                (end_time - start_time) * time_scale,
                (high_freq - low_freq) * frequency_scale,
            ]
        )
        size_heatmap = arrays.set_value_at_pos(
            size_heatmap,
            size,
            time=time,
            frequency=frequency,
        )

        # Get the class name of the sound event
        class_name = class_mapper.encode(sound_event_annotation)

        if class_name is None:
            # If the label is None skip the sound event
            continue

        # Set 1.0 at the position and category of the sound event in the class
        # heatmap
        class_heatmap = arrays.set_value_at_pos(
            class_heatmap,
            1.0,
            time=time,
            frequency=frequency,
            category=class_name,
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
