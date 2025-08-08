from typing import Optional, Tuple

from matplotlib.axes import Axes
from soundevent import data, plot

from batdetect2.plotting.clips import plot_clip
from batdetect2.preprocess import PreprocessorProtocol

__all__ = [
    "plot_clip_annotation",
]


def plot_clip_annotation(
    clip_annotation: data.ClipAnnotation,
    preprocessor: Optional[PreprocessorProtocol] = None,
    figsize: Optional[Tuple[int, int]] = None,
    ax: Optional[Axes] = None,
    audio_dir: Optional[data.PathLike] = None,
    add_colorbar: bool = False,
    add_labels: bool = False,
    add_points: bool = False,
    cmap: str = "gray",
    alpha: float = 1,
    linewidth: float = 1,
    fill: bool = False,
) -> Axes:
    ax = plot_clip(
        clip_annotation.clip,
        preprocessor=preprocessor,
        figsize=figsize,
        ax=ax,
        audio_dir=audio_dir,
        add_colorbar=add_colorbar,
        add_labels=add_labels,
        spec_cmap=cmap,
    )

    plot.plot_annotations(
        clip_annotation.sound_events,
        ax=ax,
        time_offset=0.004,
        freq_offset=2_000,
        add_points=add_points,
        alpha=alpha,
        linewidth=linewidth,
        facecolor="none" if not fill else None,
    )
    return ax
