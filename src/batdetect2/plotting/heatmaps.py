"""Plot heatmaps"""

import numpy as np
import torch
from matplotlib import axes, patches
from matplotlib.cm import get_cmap
from matplotlib.colors import Colormap, LinearSegmentedColormap, to_rgba

from batdetect2.plotting.common import create_ax


def plot_detection_heatmap(
    heatmap: torch.Tensor | np.ndarray,
    ax: axes.Axes | None = None,
    figsize: tuple[int, int] = (10, 10),
    threshold: float | None = None,
    alpha: float = 1,
    cmap: str | Colormap = "jet",
    color: str | None = None,
) -> axes.Axes:
    ax = create_ax(ax, figsize=figsize)

    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.numpy()

    heatmap = heatmap.squeeze()

    if threshold is not None:
        heatmap = np.ma.masked_where(
            heatmap < threshold,
            heatmap,
        )

    if color is not None:
        cmap = create_colormap(color)

    ax.pcolormesh(
        heatmap,
        vmax=1,
        vmin=0,
        cmap=cmap,
        alpha=alpha,
    )

    return ax


def plot_classification_heatmap(
    heatmap: torch.Tensor | np.ndarray,
    ax: axes.Axes | None = None,
    figsize: tuple[int, int] = (10, 10),
    class_names: list[str] | None = None,
    threshold: float | None = 0.1,
    alpha: float = 1,
    cmap: str | Colormap = "tab20",
):
    ax = create_ax(ax, figsize=figsize)

    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.numpy()

    if heatmap.ndim == 4:
        heatmap = heatmap[0]

    if heatmap.ndim != 3:
        raise ValueError("Expecting a 3-dimensional array")

    num_classes = heatmap.shape[0]

    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]

    if len(class_names) != num_classes:
        raise ValueError("Inconsistent number of class names")

    if not isinstance(cmap, Colormap):
        cmap = get_cmap(cmap)

    handles = []

    for index, class_heatmap in enumerate(heatmap):
        class_name = class_names[index]

        color = cmap(index / num_classes)

        max = class_heatmap.max()

        if max == 0:
            continue

        if threshold is not None:
            class_heatmap = np.ma.masked_where(
                class_heatmap < threshold,
                class_heatmap,
            )

        ax.pcolormesh(
            class_heatmap,
            vmax=1,
            vmin=0,
            cmap=create_colormap(color),
            alpha=alpha,
        )

        handles.append(patches.Patch(color=color, label=class_name))

    ax.legend(handles=handles)
    return ax


def create_colormap(color: str) -> Colormap:
    (r, g, b, a) = to_rgba(color)
    return LinearSegmentedColormap.from_list(
        "cmap", colors=[(0, 0, 0, 0), (r, g, b, a)]
    )
