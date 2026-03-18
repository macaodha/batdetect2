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

from typing import Dict, List

import numpy as np
import torch
import xarray as xr
from soundevent.arrays import Dimensions

from batdetect2.preprocess import MAX_FREQ, MIN_FREQ

__all__ = [
    "to_xarray",
]


def to_xarray(
    array: torch.Tensor | np.ndarray,
    start_time: float,
    end_time: float,
    min_freq: float = MIN_FREQ,
    max_freq: float = MAX_FREQ,
    name: str = "xarray",
    extra_dims: List[str] | None = None,
    extra_coords: Dict[str, np.ndarray] | None = None,
) -> xr.DataArray:
    if isinstance(array, torch.Tensor):
        array = array.detach().cpu().numpy()

    extra_ndims = array.ndim - 2

    if extra_ndims < 0:
        raise ValueError(
            "Input array must have at least 2 dimensions, "
            f"got shape {array.shape}"
        )

    width = array.shape[-1]
    height = array.shape[-2]

    times = np.linspace(start_time, end_time, width, endpoint=False)
    freqs = np.linspace(min_freq, max_freq, height, endpoint=False)

    if extra_dims is None:
        extra_dims = [f"dim_{i}" for i in range(extra_ndims)]

    if extra_coords is None:
        extra_coords = {}

    return xr.DataArray(
        data=array,
        dims=[
            *extra_dims,
            Dimensions.frequency.value,
            Dimensions.time.value,
        ],
        coords={
            **extra_coords,
            Dimensions.frequency.value: freqs,
            Dimensions.time.value: times,
        },
        name=name,
    )
