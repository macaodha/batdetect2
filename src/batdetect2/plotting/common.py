"""General plotting utilities."""

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import axes

__all__ = [
    "create_ax",
]


def create_ax(
    ax: axes.Axes | None = None,
    figsize: Tuple[int, int] | None = None,
    **kwargs,
) -> axes.Axes:
    """Create a new axis if none is provided"""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize, nrows=1, ncols=1, **kwargs)

    return ax


def plot_spectrogram(
    spec: torch.Tensor | np.ndarray,
    start_time: float | None = None,
    end_time: float | None = None,
    min_freq: float | None = None,
    max_freq: float | None = None,
    ax: axes.Axes | None = None,
    figsize: Tuple[int, int] | None = None,
    add_colorbar: bool = False,
    colorbar_kwargs: dict | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
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

    ax.set_xlim(start_time, end_time)
    ax.set_ylim(min_freq, max_freq)

    if add_colorbar:
        plt.colorbar(mappable, ax=ax, **(colorbar_kwargs or {}))

    return ax
