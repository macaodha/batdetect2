from typing import Iterable, Optional, Tuple

from matplotlib.axes import Axes
from soundevent import data
from soundevent.geometry.operations import Positions, get_geometry_point
from soundevent.plot.common import create_axes
from soundevent.plot.geometries import plot_geometry
from soundevent.plot.tags import TagColorMapper, add_tags_legend, plot_tag

from batdetect2.plotting.clips import plot_clip
from batdetect2.preprocess import PreprocessorProtocol

__all__ = [
    "plot_clip_prediction",
]


def plot_clip_prediction(
    clip_prediction: data.ClipPrediction,
    preprocessor: Optional[PreprocessorProtocol] = None,
    figsize: Optional[Tuple[int, int]] = None,
    ax: Optional[Axes] = None,
    audio_dir: Optional[data.PathLike] = None,
    add_colorbar: bool = False,
    add_labels: bool = False,
    add_legend: bool = False,
    spec_cmap: str = "gray",
    linewidth: float = 1,
    fill: bool = False,
) -> Axes:
    ax = plot_clip(
        clip_prediction.clip,
        preprocessor=preprocessor,
        figsize=figsize,
        ax=ax,
        audio_dir=audio_dir,
        add_colorbar=add_colorbar,
        add_labels=add_labels,
        spec_cmap=spec_cmap,
    )

    plot_predictions(
        clip_prediction.sound_events,
        ax=ax,
        time_offset=0.004,
        freq_offset=2_000,
        add_points=False,
        linewidth=linewidth,
        facecolor="none" if not fill else None,
        legend=add_legend,
    )
    return ax


def plot_predictions(
    predictions: Iterable[data.SoundEventPrediction],
    ax: Optional[Axes] = None,
    position: Positions = "top-right",
    color_mapper: Optional[TagColorMapper] = None,
    time_offset: float = 0.001,
    freq_offset: float = 1000,
    legend: bool = True,
    max_alpha: float = 0.5,
    color: Optional[str] = None,
    **kwargs,
):
    """Plot an prediction."""
    if ax is None:
        ax = create_axes(**kwargs)

    if color_mapper is None:
        color_mapper = TagColorMapper()

    for prediction in predictions:
        ax = plot_prediction(
            prediction,
            ax=ax,
            position=position,
            color_mapper=color_mapper,
            time_offset=time_offset,
            freq_offset=freq_offset,
            max_alpha=max_alpha,
            color=color,
            **kwargs,
        )

    if legend:
        ax = add_tags_legend(ax, color_mapper)

    return ax


def plot_prediction(
    prediction: data.SoundEventPrediction,
    ax: Optional[Axes] = None,
    position: Positions = "top-right",
    color_mapper: Optional[TagColorMapper] = None,
    time_offset: float = 0.001,
    freq_offset: float = 1000,
    max_alpha: float = 0.5,
    alpha: Optional[float] = None,
    color: Optional[str] = None,
    **kwargs,
) -> Axes:
    """Plot an annotation."""
    geometry = prediction.sound_event.geometry

    if geometry is None:
        raise ValueError("Annotation does not have a geometry.")

    if ax is None:
        ax = create_axes(**kwargs)

    if color_mapper is None:
        color_mapper = TagColorMapper()

    if alpha is None:
        alpha = min(prediction.score * max_alpha, 1)

    ax = plot_geometry(
        geometry,
        ax=ax,
        color=color,
        alpha=alpha,
        **kwargs,
    )

    x, y = get_geometry_point(geometry, position=position)

    for index, tag in enumerate(prediction.tags):
        color = color_mapper.get_color(tag.tag)
        ax = plot_tag(
            time=x + time_offset,
            frequency=y - index * freq_offset,
            color=color,
            ax=ax,
            alpha=min(tag.score, prediction.score),
            **kwargs,
        )

    return ax
