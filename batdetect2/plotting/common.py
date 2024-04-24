"""General plotting utilities."""

from typing import Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib import axes

__all__ = [
    "create_ax",
]


def create_ax(
    ax: Optional[axes.Axes] = None,
    figsize: Tuple[int, int] = (10, 10),
    **kwargs,
) -> axes.Axes:
    """Create a new axis if none is provided"""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize, **kwargs) # type: ignore

    return ax # type: ignore
