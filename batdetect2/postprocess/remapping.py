"""Remaps raw model output tensors to coordinate-aware xarray DataArrays.

This module provides utility functions to convert the raw numerical outputs
(typically PyTorch tensors) from the BatDetect2 DNN model into
`xarray.DataArray` objects. This step adds coordinate information
(time in seconds, frequency in Hz) back to the model's predictions, making them
interpretable in the context of the original audio signal and facilitating
subsequent processing steps.

Functions are provided for common BatDetect2 output types: detection heatmaps,
classification probability maps, size prediction maps, and potentially
intermediate features.
"""

from typing import List

import numpy as np
import torch
import xarray as xr
from soundevent.arrays import Dimensions

from batdetect2.preprocess import MAX_FREQ, MIN_FREQ

__all__ = [
    "features_to_xarray",
    "detection_to_xarray",
    "classification_to_xarray",
    "sizes_to_xarray",
]


def features_to_xarray(
    features: torch.Tensor,
    start_time: float,
    end_time: float,
    min_freq: float = MIN_FREQ,
    max_freq: float = MAX_FREQ,
    features_prefix: str = "batdetect2_feature_",
):
    """Convert a multi-channel feature tensor to a coordinate-aware DataArray.

    Assigns time, frequency, and feature coordinates to a raw feature tensor
    output by the model.

    Parameters
    ----------
    features : torch.Tensor
        The raw feature tensor from the model. Expected shape is
        (num_features, num_freq_bins, num_time_bins).
    start_time : float
        The start time (in seconds) corresponding to the first time bin of
        the tensor.
    end_time : float
        The end time (in seconds) corresponding to the *end* of the last time
        bin.
    min_freq : float, default=MIN_FREQ
        The minimum frequency (in Hz) corresponding to the first frequency bin.
    max_freq : float, default=MAX_FREQ
        The maximum frequency (in Hz) corresponding to the *end* of the last
        frequency bin.
    features_prefix : str, default="batdetect2_feature_"
        Prefix used to generate names for the feature coordinate dimension
        (e.g., "batdetect2_feature_0", "batdetect2_feature_1", ...).

    Returns
    -------
    xr.DataArray
        An xarray DataArray containing the feature data with named dimensions
        ('feature', 'frequency', 'time') and calculated coordinates.

    Raises
    ------
    ValueError
        If the input tensor does not have 3 dimensions.
    """
    if features.ndim != 3:
        raise ValueError(
            "Input features tensor must have 3 dimensions (C, T, F), "
            f"got shape {features.shape}"
        )

    num_features, height, width = features.shape
    times = np.linspace(start_time, end_time, width, endpoint=False)
    freqs = np.linspace(min_freq, max_freq, height, endpoint=False)

    return xr.DataArray(
        data=features.detach().numpy(),
        dims=[
            Dimensions.feature.value,
            Dimensions.frequency.value,
            Dimensions.time.value,
        ],
        coords={
            Dimensions.feature.value: [
                f"{features_prefix}{i}" for i in range(num_features)
            ],
            Dimensions.frequency.value: freqs,
            Dimensions.time.value: times,
        },
        name="features",
    )


def detection_to_xarray(
    detection: torch.Tensor,
    start_time: float,
    end_time: float,
    min_freq: float = MIN_FREQ,
    max_freq: float = MAX_FREQ,
) -> xr.DataArray:
    """Convert a single-channel detection heatmap tensor to a DataArray.

    Assigns time and frequency coordinates to a raw detection heatmap tensor.

    Parameters
    ----------
    detection : torch.Tensor
        Raw detection heatmap tensor from the model. Expected shape is
        (1, num_freq_bins, num_time_bins).
    start_time : float
        Start time (seconds) corresponding to the first time bin.
    end_time : float
        End time (seconds) corresponding to the end of the last time bin.
    min_freq : float, default=MIN_FREQ
        Minimum frequency (Hz) corresponding to the first frequency bin.
    max_freq : float, default=MAX_FREQ
        Maximum frequency (Hz) corresponding to the end of the last frequency
        bin.

    Returns
    -------
    xr.DataArray
        An xarray DataArray containing the detection scores with named
        dimensions ('frequency', 'time') and calculated coordinates.

    Raises
    ------
    ValueError
        If the input tensor does not have 3 dimensions or if the first
        dimension size is not 1.
    """
    if detection.ndim != 3:
        raise ValueError(
            "Input detection tensor must have 3 dimensions (1, T, F), "
            f"got shape {detection.shape}"
        )

    num_channels, height, width = detection.shape

    if num_channels != 1:
        raise ValueError(
            "Expected a single channel output, instead got "
            f"{num_channels} channels"
        )

    times = np.linspace(start_time, end_time, width, endpoint=False)
    freqs = np.linspace(min_freq, max_freq, height, endpoint=False)

    return xr.DataArray(
        data=detection.squeeze(dim=0).detach().numpy(),
        dims=[
            Dimensions.frequency.value,
            Dimensions.time.value,
        ],
        coords={
            Dimensions.frequency.value: freqs,
            Dimensions.time.value: times,
        },
        name="detection_score",
    )


def classification_to_xarray(
    classes: torch.Tensor,
    start_time: float,
    end_time: float,
    class_names: List[str],
    min_freq: float = MIN_FREQ,
    max_freq: float = MAX_FREQ,
) -> xr.DataArray:
    """Convert multi-channel class probability tensor to a DataArray.

    Assigns category (class name), frequency, and time coordinates to a raw
    class probability tensor output by the model.

    Parameters
    ----------
    classes : torch.Tensor
        Raw class probability tensor. Expected shape is
        (num_classes, num_freq_bins, num_time_bins).
    start_time : float
        Start time (seconds) corresponding to the first time bin.
    end_time : float
        End time (seconds) corresponding to the end of the last time bin.
    class_names : List[str]
        Ordered list of class names corresponding to the first dimension
        of the `classes` tensor. The length must match `classes.shape[0]`.
    min_freq : float, default=MIN_FREQ
        Minimum frequency (Hz) corresponding to the first frequency bin.
    max_freq : float, default=MAX_FREQ
        Maximum frequency (Hz) corresponding to the end of the last frequency
        bin.

    Returns
    -------
    xr.DataArray
        An xarray DataArray containing class probabilities with named
        dimensions ('category', 'frequency', 'time') and calculated
        coordinates.

    Raises
    ------
    ValueError
        If the input tensor does not have 3 dimensions, or if the size of the
        first dimension does not match the length of `class_names`.
    """
    if classes.ndim != 3:
        raise ValueError(
            "Input classes tensor must have 3 dimensions (C, F, T), "
            f"got shape {classes.shape}"
        )

    num_classes, height, width = classes.shape

    if num_classes != len(class_names):
        raise ValueError(
            "The number of classes does not coincide with the number of "
            "class names provided: "
            f"({num_classes = }) != ({len(class_names) = })"
        )

    times = np.linspace(start_time, end_time, width, endpoint=False)
    freqs = np.linspace(min_freq, max_freq, height, endpoint=False)

    return xr.DataArray(
        data=classes.detach().numpy(),
        dims=[
            "category",
            Dimensions.frequency.value,
            Dimensions.time.value,
        ],
        coords={
            "category": class_names,
            Dimensions.frequency.value: freqs,
            Dimensions.time.value: times,
        },
        name="class_scores",
    )


def sizes_to_xarray(
    sizes: torch.Tensor,
    start_time: float,
    end_time: float,
    min_freq: float = MIN_FREQ,
    max_freq: float = MAX_FREQ,
) -> xr.DataArray:
    """Convert the 2-channel size prediction tensor to a DataArray.

    Assigns dimension ('width', 'height'), frequency, and time coordinates
    to the raw size prediction tensor output by the model.

    Parameters
    ----------
    sizes : torch.Tensor
        Raw size prediction tensor. Expected shape is
        (2, num_freq_bins, num_time_bins), where the first dimension
        corresponds to predicted width and height respectively.
    start_time : float
        Start time (seconds) corresponding to the first time bin.
    end_time : float
        End time (seconds) corresponding to the end of the last time bin.
    min_freq : float, default=MIN_FREQ
        Minimum frequency (Hz) corresponding to the first frequency bin.
    max_freq : float, default=MAX_FREQ
        Maximum frequency (Hz) corresponding to the end of the last frequency
        bin.

    Returns
    -------
    xr.DataArray
        An xarray DataArray containing predicted sizes with named dimensions
        ('dimension', 'frequency', 'time') and calculated time/frequency
        coordinates. The 'dimension' coordinate will have values
        ['width', 'height'].

    Raises
    ------
    ValueError
        If the input tensor does not have 3 dimensions or if the first
        dimension size is not exactly 2.
    """
    num_channels, height, width = sizes.shape

    if num_channels != 2:
        raise ValueError(
            "Expected a two-channel output, instead got "
            f"{num_channels} channels"
        )

    times = np.linspace(start_time, end_time, width, endpoint=False)
    freqs = np.linspace(min_freq, max_freq, height, endpoint=False)

    return xr.DataArray(
        data=sizes.detach().numpy(),
        dims=[
            "dimension",
            Dimensions.frequency.value,
            Dimensions.time.value,
        ],
        coords={
            "dimension": ["width", "height"],
            Dimensions.frequency.value: freqs,
            Dimensions.time.value: times,
        },
    )
