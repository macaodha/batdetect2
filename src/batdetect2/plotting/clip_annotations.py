from typing import Optional, Tuple

from matplotlib.axes import Axes
from soundevent import data, plot

from batdetect2.plotting.clips import plot_clip
from batdetect2.plotting.common import create_ax
from batdetect2.typing.preprocess import PreprocessorProtocol
from batdetect2.typing.targets import TargetProtocol

__all__ = [
    "plot_clip_annotation",
]


def plot_clip_annotation(
    clip_annotation: data.ClipAnnotation,
    preprocessor: PreprocessorProtocol | None = None,
    figsize: Tuple[int, int] | None = None,
    ax: Axes | None = None,
    audio_dir: data.PathLike | None = None,
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


def plot_anchor_points(
    clip_annotation: data.ClipAnnotation,
    targets: TargetProtocol,
    figsize: Tuple[int, int] | None = None,
    ax: Axes | None = None,
    size: int = 1,
    color: str = "red",
    marker: str = "x",
    alpha: float = 1,
) -> Axes:
    ax = create_ax(ax=ax, figsize=figsize)

    positions = []

    for sound_event in clip_annotation.sound_events:
        if not targets.filter(sound_event):
            continue

        position, _ = targets.encode_roi(sound_event)
        positions.append(position)

    X, Y = zip(*positions)
    ax.scatter(X, Y, s=size, c=color, marker=marker, alpha=alpha)
    return ax
