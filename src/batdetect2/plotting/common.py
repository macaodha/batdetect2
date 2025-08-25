"""General plotting utilities."""

from typing import Optional, Tuple

import matplotlib.pyplot as plt
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
    ax: Optional[axes.Axes] = None,
    figsize: Optional[Tuple[int, int]] = None,
    cmap="gray",
) -> axes.Axes:
    ax = create_ax(ax=ax, figsize=figsize)
    ax.pcolormesh(spec.numpy(), cmap=cmap)
    return ax
