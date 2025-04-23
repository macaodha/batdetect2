from typing import Optional, Tuple, Union

import numpy as np
import xarray as xr
from soundevent import arrays

from batdetect2.configs import BaseConfig
from batdetect2.train.types import ClipperProtocol

DEFAULT_TRAIN_CLIP_DURATION = 0.513
DEFAULT_MAX_EMPTY_CLIP = 0.1


class ClipingConfig(BaseConfig):
    duration: float = DEFAULT_TRAIN_CLIP_DURATION
    random: bool = True
    max_empty: float = DEFAULT_MAX_EMPTY_CLIP


class Clipper(ClipperProtocol):
    def __init__(
        self,
        duration: float = 0.5,
        max_empty: float = 0.2,
        random: bool = True,
    ):
        self.duration = duration
        self.random = random
        self.max_empty = max_empty

    def extract_clip(
        self, example: xr.Dataset
    ) -> Tuple[xr.Dataset, float, float]:
        step = arrays.get_dim_step(
            example.spectrogram,
            dim=arrays.Dimensions.time.value,
        )
        duration = (
            arrays.get_dim_width(
                example.spectrogram,
                dim=arrays.Dimensions.time.value,
            )
            + step
        )

        start_time = 0
        if self.random:
            start_time = np.random.uniform(
                -self.max_empty,
                duration - self.duration + self.max_empty,
            )

        subclip = select_subclip(
            example,
            start=start_time,
            span=self.duration,
            dim="time",
        )

        return (
            select_subclip(
                subclip,
                start=start_time,
                span=self.duration,
                dim="audio_time",
            ),
            start_time,
            start_time + self.duration,
        )


def build_clipper(config: Optional[ClipingConfig] = None) -> ClipperProtocol:
    config = config or ClipingConfig()
    return Clipper(
        duration=config.duration,
        max_empty=config.max_empty,
        random=config.random,
    )


def select_subclip(
    dataset: xr.Dataset,
    span: float,
    start: float,
    fill_value: float = 0,
    dim: str = "time",
) -> xr.Dataset:
    width = _compute_expected_width(
        dataset,  # type: ignore
        span,
        dim=dim,
    )

    coord = dataset.coords[dim]

    if len(coord) == width:
        return dataset

    new_coords, start_pad, end_pad, dim_slice = _extract_coordinate(
        coord, start, span
    )

    data_vars = {}
    for name, data_array in dataset.data_vars.items():
        if dim not in data_array.dims:
            data_vars[name] = data_array
            continue

        if width == data_array.sizes[dim]:
            data_vars[name] = data_array
            continue

        sliced = data_array.isel({dim: dim_slice}).data

        if start_pad > 0 or end_pad > 0:
            padding = [
                [0, 0] if other_dim != dim else [start_pad, end_pad]
                for other_dim in data_array.dims
            ]
            sliced = np.pad(sliced, padding, constant_values=fill_value)

        data_vars[name] = xr.DataArray(
            data=sliced,
            dims=data_array.dims,
            coords={**data_array.coords, dim: new_coords},
            attrs=data_array.attrs,
        )

    return xr.Dataset(data_vars=data_vars, attrs=dataset.attrs)


def _extract_coordinate(
    coord: xr.DataArray,
    start: float,
    span: float,
) -> Tuple[xr.Variable, int, int, slice]:
    step = arrays.get_dim_step(coord, str(coord.name))

    current_width = len(coord)
    expected_width = int(np.floor(span / step))

    coord_start = float(coord[0])
    offset = start - coord_start

    start_index = int(np.floor(offset / step))
    end_index = start_index + expected_width

    if start_index > current_width:
        raise ValueError("Requested span does not overlap with current range")

    if end_index < 0:
        raise ValueError("Requested span does not overlap with current range")

    corrected_start = float(start_index * step)
    corrected_end = float(end_index * step)

    start_index_offset = max(0, -start_index)
    end_index_offset = max(0, end_index - current_width)

    sl = slice(
        start_index if start_index >= 0 else None,
        end_index if end_index < current_width else None,
    )

    return (
        arrays.create_range_dim(
            str(coord.name),
            start=corrected_start,
            stop=corrected_end,
            step=step,
        ),
        start_index_offset,
        end_index_offset,
        sl,
    )


def _compute_expected_width(
    array: Union[xr.DataArray, xr.Dataset],
    duration: float,
    dim: str,
) -> int:
    step = arrays.get_dim_step(array, dim)  # type: ignore
    return int(np.floor(duration / step))
