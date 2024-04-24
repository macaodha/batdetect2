"""Plot heatmaps"""

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import xarray as xr
from matplotlib import axes

from batdetect2.plotting.common import create_ax


def plot_heatmap(
    heatmap: xr.DataArray,
    ax: Optional[axes.Axes] = None,
    figsize: Tuple[int, int] = (10, 10),
) -> axes.Axes:
    ax = create_ax(ax, figsize=figsize)

    ax.pcolormesh(
        heatmap.time,
        heatmap.frequency,
        heatmap,
        vmax=1,
        vmin=0,
    )

    return ax
