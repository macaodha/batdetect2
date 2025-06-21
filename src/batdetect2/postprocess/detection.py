"""Extracts candidate detection points from a model output heatmap.

This module implements a specific step within the BatDetect2 postprocessing
pipeline. Its primary function is to identify potential sound event locations
by finding peaks (local maxima or high-scoring points) in the detection heatmap
produced by the neural network (usually after Non-Maximum Suppression and
coordinate remapping have been applied).

It provides functionality to:
- Identify the locations (time, frequency) of the highest-scoring points.
- Filter these points based on a minimum confidence score threshold.
- Limit the maximum number of detection points returned (top-k).

The main output is an `xarray.DataArray` containing the scores and
corresponding time/frequency coordinates for the extracted detection points.
This output serves as the input for subsequent postprocessing steps, such as
extracting predicted class probabilities and bounding box sizes at these
specific locations.
"""

from typing import Optional

import numpy as np
import xarray as xr
from soundevent.arrays import Dimensions, get_dim_width

__all__ = [
    "extract_detections_from_array",
    "get_max_detections",
    "DEFAULT_DETECTION_THRESHOLD",
    "TOP_K_PER_SEC",
]

DEFAULT_DETECTION_THRESHOLD = 0.01
"""Default confidence score threshold used for filtering detections."""

TOP_K_PER_SEC = 200
"""Default desired maximum number of detections per second of audio."""


def extract_detections_from_array(
    detection_array: xr.DataArray,
    max_detections: Optional[int] = None,
    threshold: Optional[float] = DEFAULT_DETECTION_THRESHOLD,
) -> xr.DataArray:
    """Extract detection locations (time, freq) and scores from a heatmap.

    Identifies the pixels with the highest scores in the input detection
    heatmap, filters them based on an optional score `threshold`, limits the
    number to an optional `max_detections`, and returns their scores along with
    their corresponding time and frequency coordinates.

    Parameters
    ----------
    detection_array : xr.DataArray
        A 2D xarray DataArray representing the detection heatmap. Must have
        dimensions and coordinates named 'time' and 'frequency'. Higher values
        are assumed to indicate higher detection confidence.
    max_detections : int, optional
        The absolute maximum number of detections to return. If specified, only
        the top `max_detections` highest-scoring detections (passing the
        threshold) are returned. If None (default), all detections passing
        the threshold are returned, sorted by score.
    threshold : float, optional
        The minimum confidence score required for a detection peak to be
        kept. Detections with scores below this value are discarded.
        Defaults to `DEFAULT_DETECTION_THRESHOLD`. If set to None, no
        thresholding is applied.

    Returns
    -------
    xr.DataArray
        A 1D xarray DataArray named 'score' with a 'detection' dimension.
        - The data values are the scores of the extracted detections, sorted
          in descending order.
        - It has coordinates 'time' and 'frequency' (also indexed by the
          'detection' dimension) indicating the location of each detection
          peak in the original coordinate system.
        - Returns an empty DataArray if no detections pass the criteria.

    Raises
    ------
    ValueError
        If `max_detections` is not None and not a positive integer, or if
        `detection_array` lacks required dimensions/coordinates.
    """
    if max_detections is not None:
        if max_detections <= 0:
            raise ValueError("Max detections must be positive")

    values = detection_array.values.flatten()

    if max_detections is not None:
        top_indices = np.argpartition(-values, max_detections)[:max_detections]
        top_sorted_indices = top_indices[np.argsort(-values[top_indices])]
    else:
        top_sorted_indices = np.argsort(-values)

    top_values = values[top_sorted_indices]

    if threshold is not None:
        mask = top_values > threshold
        top_values = top_values[mask]
        top_sorted_indices = top_sorted_indices[mask]

    freq_indices, time_indices = np.unravel_index(
        top_sorted_indices,
        detection_array.shape,
    )

    times = detection_array.coords[Dimensions.time.value].values[time_indices]
    freqs = detection_array.coords[Dimensions.frequency.value].values[
        freq_indices
    ]

    return xr.DataArray(
        data=top_values,
        coords={
            Dimensions.frequency.value: ("detection", freqs),
            Dimensions.time.value: ("detection", times),
        },
        dims="detection",
        name="score",
    )


def get_max_detections(
    detection_array: xr.DataArray,
    top_k_per_sec: int = TOP_K_PER_SEC,
) -> int:
    """Calculate max detections allowed based on duration and rate.

    Determines the total maximum number of detections to extract from a
    heatmap based on its time duration and a desired rate of detections
    per second.

    Parameters
    ----------
    detection_array : xr.DataArray
        The detection heatmap, requiring 'time' coordinates from which the
        total duration can be calculated using
        `soundevent.arrays.get_dim_width`.
    top_k_per_sec : int, default=TOP_K_PER_SEC
        The desired maximum number of detections to allow per second of audio.

    Returns
    -------
    int
        The calculated total maximum number of detections allowed for the
        entire duration of the `detection_array`.

    Raises
    ------
    ValueError
        If the duration cannot be calculated from the `detection_array` (e.g.,
        missing or invalid 'time' coordinates/dimension).
    """
    if top_k_per_sec < 0:
        raise ValueError("top_k_per_sec cannot be negative.")

    duration = get_dim_width(detection_array, Dimensions.time.value)
    return int(duration * top_k_per_sec)
