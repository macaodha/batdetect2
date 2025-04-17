from typing import Union

import numpy as np
import torch
from torch.nn import functional as F


def extend_width(
    array: Union[np.ndarray, torch.Tensor],
    extra: int,
    axis: int = -1,
    value: float = 0,
) -> torch.Tensor:
    if not isinstance(array, torch.Tensor):
        array = torch.Tensor(array)

    dims = len(array.shape)
    axis = axis % dims
    pad = [
        [0, 0] if index != axis else [0, extra]
        for index in range(axis, dims)[::-1]
    ]
    return F.pad(
        array,
        [x for y in pad for x in y],
        value=value,
    )


def make_width_divisible(
    array: Union[np.ndarray, torch.Tensor],
    factor: int,
    axis: int = -1,
    value: float = 0,
) -> torch.Tensor:
    if not isinstance(array, torch.Tensor):
        array = torch.Tensor(array)

    width = array.shape[axis]

    if width % factor == 0:
        return array

    extra = (-width) % factor
    return extend_width(array, extra, axis=axis, value=value)


def adjust_width(
    array: Union[np.ndarray, torch.Tensor],
    width: int,
    axis: int = -1,
    value: float = 0,
) -> torch.Tensor:
    if not isinstance(array, torch.Tensor):
        array = torch.Tensor(array)

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
