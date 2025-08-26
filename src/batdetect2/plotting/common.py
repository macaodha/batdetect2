"""General plotting utilities."""

from typing import Optional, Tuple

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
    spec: torch.Tensor,
    start_time: float,
    end_time: float,
    min_freq: float,
    max_freq: float,
    ax: Optional[axes.Axes] = None,
    figsize: Optional[Tuple[int, int]] = None,
    cmap="gray",
) -> axes.Axes:
    ax = create_ax(ax=ax, figsize=figsize)

    ax.pcolormesh(
        np.linspace(start_time, end_time, spec.shape[-1], endpoint=False),
        np.linspace(min_freq, max_freq, spec.shape[-2], endpoint=False),
        spec.numpy(),
        cmap=cmap,
    )
    return ax
