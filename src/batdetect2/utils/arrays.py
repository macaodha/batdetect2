import numpy as np
import torch
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


def extend_width(
    tensor: torch.Tensor,
    extra: int,
    axis: int = -1,
    value: float = 0,
) -> torch.Tensor:
    dims = len(tensor.shape)
    axis = dims - axis % dims - 1
    pad = [0 for _ in range(2 * dims)]
    pad[2 * axis + 1] = extra
    return torch.nn.functional.pad(
        tensor,
        pad,
        mode="constant",
        value=value,
    )


def adjust_width(
    tensor: torch.Tensor,
    width: int,
    axis: int = -1,
    value: float = 0,
) -> torch.Tensor:
    dims = len(tensor.shape)
    axis = axis % dims
    current_width = tensor.shape[axis]

    if current_width == width:
        return tensor

    if current_width < width:
        return extend_width(
            tensor,
            extra=width - current_width,
            axis=axis,
            value=value,
        )

    slices = [
        slice(None, None) if index != axis else slice(None, width)
        for index in range(dims)
    ]
    return tensor[tuple(slices)]
