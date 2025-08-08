from typing import Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from soundevent import data

from batdetect2.preprocess import (
    PreprocessorProtocol,
    get_default_preprocessor,
)

__all__ = [
    "plot_clip",
]


def plot_clip(
    clip: data.Clip,
    preprocessor: Optional[PreprocessorProtocol] = None,
    figsize: Optional[Tuple[int, int]] = None,
    ax: Optional[Axes] = None,
    audio_dir: Optional[data.PathLike] = None,
    add_colorbar: bool = False,
    add_labels: bool = False,
    spec_cmap: str = "gray",
) -> Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if preprocessor is None:
        preprocessor = get_default_preprocessor()

    spec = preprocessor.preprocess_clip(clip, audio_dir=audio_dir)

    spec.plot(  # type: ignore
        ax=ax,
        add_colorbar=add_colorbar,
        cmap=spec_cmap,
        add_labels=add_labels,
        vmin=spec.min().item(),
        vmax=spec.max().item(),
    )

    return ax
