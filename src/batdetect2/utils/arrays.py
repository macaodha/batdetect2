import numpy as np
import xarray as xr


def spec_to_xarray(
    spec: np.ndarray,
    start_time: float,
    end_time: float,
    min_freq: float,
    max_freq: float,
) -> xr.DataArray:
    if spec.ndim != 2:
        raise ValueError(
            "Input numpy spectrogram array should be 2-dimensional"
        )

    height, width = spec.shape
    return xr.DataArray(
        data=spec,
        dims=["frequency", "time"],
        coords={
            "frequency": np.linspace(
                min_freq,
                max_freq,
                height,
                endpoint=False,
            ),
            "time": np.linspace(
                start_time,
                end_time,
                width,
                endpoint=False,
            ),
        },
    )


def audio_to_xarray(
    wav: np.ndarray,
    start_time: float,
    end_time: float,
    time_axis: str = "time",
) -> xr.DataArray:
    if wav.ndim != 1:
        raise ValueError("Input numpy audio array should be 1-dimensional")

    return xr.DataArray(
        data=wav,
        dims=[time_axis],
        coords={
            time_axis: np.linspace(
                start_time,
                end_time,
                len(wav),
                endpoint=False,
            ),
        },
    )


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
