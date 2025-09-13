"""General plotting utilities."""

from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import axes

__all__ = [
    "create_ax",
]


def create_ax(
    ax: Optional[axes.Axes] = None,
    figsize: Optional[Tuple[int, int]] = None,
    **kwargs,
) -> axes.Axes:
    """Create a new axis if none is provided"""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize, **kwargs)  # type: ignore

    return ax  # type: ignore


def plot_spectrogram(
    spec: Union[torch.Tensor, np.ndarray],
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    min_freq: Optional[float] = None,
    max_freq: Optional[float] = None,
    ax: Optional[axes.Axes] = None,
    figsize: Optional[Tuple[int, int]] = None,
    add_colorbar: bool = False,
    colorbar_kwargs: Optional[dict] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap="gray",
) -> axes.Axes:
    if isinstance(spec, torch.Tensor):
        spec = spec.numpy()

    spec = spec.squeeze()

    ax = create_ax(ax=ax, figsize=figsize)

    if start_time is None:
        start_time = 0

    if end_time is None:
        end_time = spec.shape[-1]

    if min_freq is None:
        min_freq = 0

    if max_freq is None:
        max_freq = spec.shape[-2]

    mappable = ax.pcolormesh(
        np.linspace(start_time, end_time, spec.shape[-1] + 1, endpoint=True),
        np.linspace(min_freq, max_freq, spec.shape[-2] + 1, endpoint=True),
        spec,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    if add_colorbar:
        plt.colorbar(mappable, ax=ax, **(colorbar_kwargs or {}))

    return ax
