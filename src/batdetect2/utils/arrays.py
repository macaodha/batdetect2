import numpy as np
import xarray as xr


def extend_width(
    array: np.ndarray,
    extra: int,
    axis: int = -1,
    value: float = 0,
) -> np.ndarray:
    dims = len(array.shape)
    axis = axis % dims
    pad = [[0, 0] if index != axis else [0, extra] for index in range(dims)]
    return np.pad(
        array,
        pad,
        mode="constant",
        constant_values=value,
    )


def make_width_divisible(
    array: np.ndarray,
    factor: int,
    axis: int = -1,
    value: float = 0,
) -> np.ndarray:
    width = array.shape[axis]

    if width % factor == 0:
        return array

    extra = (-width) % factor
    return extend_width(array, extra, axis=axis, value=value)


def adjust_width(
    array: np.ndarray,
    width: int,
    axis: int = -1,
    value: float = 0,
) -> np.ndarray:
    dims = len(array.shape)
    axis = axis % dims
    current_width = array.shape[axis]

    if current_width == width:
        return array

    if current_width < width:
        return extend_width(
            array,
            extra=width - current_width,
            axis=axis,
            value=value,
        )

    slices = [
        slice(None, None) if index != axis else slice(None, width)
        for index in range(dims)
    ]
    return array[tuple(slices)]


def iterate_over_array(array: xr.DataArray):
    dim_name = array.dims[0]
    coords = array.coords[dim_name]
    for value, coord in zip(array.values, coords.values):
        yield coord, float(value)
