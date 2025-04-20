"""Extracts associated data for detected points from model output arrays.

This module implements a key step (Step 4) in the BatDetect2 postprocessing
pipeline. After candidate detection points (time, frequency, score) have been
identified, this module extracts the corresponding values from other raw model
output arrays, such as:

- Predicted bounding box sizes (width, height).
- Class probability scores for each defined target class.
- Intermediate feature vectors.

It uses coordinate-based indexing provided by `xarray` to ensure that the
correct values are retrieved from the original heatmaps/feature maps at the
precise time-frequency location of each detection. The final output aggregates
all extracted information into a structured `xarray.Dataset`.
"""

import xarray as xr
from soundevent.arrays import Dimensions

__all__ = [
    "extract_values_at_positions",
    "extract_detection_xr_dataset",
]


def extract_values_at_positions(
    array: xr.DataArray,
    positions: xr.DataArray,
) -> xr.DataArray:
    """Extract values from an array at specified time-frequency positions.

    Uses coordinate-based indexing to retrieve values from a source `array`
    (e.g., class probabilities, size predictions, features) at the time and
    frequency coordinates defined in the `positions` array.

    Parameters
    ----------
    array : xr.DataArray
        The source DataArray from which to extract values. Must have 'time'
        and 'frequency' dimensions and coordinates matching the space of
        `positions`.
    positions : xr.DataArray
        A 1D DataArray whose 'time' and 'frequency' coordinates specify the
        locations from which to extract values.

    Returns
    -------
    xr.DataArray
        A DataArray containing the values extracted from `array` at the given
        positions.

    Raises
    ------
    ValueError, IndexError, KeyError
        If dimensions or coordinates are missing or incompatible between
        `array` and `positions`, or if selection fails.
    """
    return array.sel(
        **{
            Dimensions.frequency.value: positions.coords[
                Dimensions.frequency.value
            ],
            Dimensions.time.value: positions.coords[Dimensions.time.value],
        }
    )


def extract_detection_xr_dataset(
    positions: xr.DataArray,
    sizes: xr.DataArray,
    classes: xr.DataArray,
    features: xr.DataArray,
) -> xr.Dataset:
    """Combine extracted detection information into a structured xr.Dataset.

    Takes the detection positions/scores and the full model output heatmaps
    (sizes, classes, optional features), extracts the relevant data at the
    detection positions, and packages everything into a single `xarray.Dataset`
    where all variables are indexed by a common 'detection' dimension.

    Parameters
    ----------
    positions : xr.DataArray
        Output from `extract_detections_from_array`, containing detection
        scores as data and 'time', 'frequency' coordinates along the
        'detection' dimension.
    sizes : xr.DataArray
        The full size prediction heatmap from the model, with dimensions like
        ('dimension', 'time', 'frequency').
    classes : xr.DataArray
        The full class probability heatmap from the model, with dimensions like
        ('category', 'time', 'frequency').
    features : xr.DataArray
        The full feature map from the model, with
        dimensions like ('feature', 'time', 'frequency').

    Returns
    -------
    xr.Dataset
        An xarray Dataset containing aligned information for each detection:
        - 'scores': DataArray from `positions` (score data, time/freq coords).
        - 'dimensions': DataArray with extracted size values
          (dims: 'detection', 'dimension').
        - 'classes': DataArray with extracted class probabilities
          (dims: 'detection', 'category').
        - 'features': DataArray with extracted feature vectors
          (dims: 'detection', 'feature'), if `features` was provided. All
          DataArrays share the 'detection' dimension and associated
          time/frequency coordinates.
    """
    sizes = extract_values_at_positions(sizes, positions).T
    classes = extract_values_at_positions(classes, positions).T
    features = extract_values_at_positions(features, positions).T
    return xr.Dataset(
        {
            "scores": positions,
            "dimensions": sizes,
            "classes": classes,
            "features": features,
        }
    )
