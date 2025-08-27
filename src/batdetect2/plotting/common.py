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
    start_time: float,
    end_time: float,
    min_freq: float,
    max_freq: float,
    ax: Optional[axes.Axes] = None,
    figsize: Optional[Tuple[int, int]] = None,
    cmap="gray",
) -> axes.Axes:
    if isinstance(spec, torch.Tensor):
        spec = spec.numpy()

    ax = create_ax(ax=ax, figsize=figsize)

    ax.pcolormesh(
        np.linspace(start_time, end_time, spec.shape[-1] + 1, endpoint=True),
        np.linspace(min_freq, max_freq, spec.shape[-2] + 1, endpoint=True),
        spec,
        cmap=cmap,
    )
    return ax
