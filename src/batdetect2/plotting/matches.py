from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from soundevent import data, plot
from soundevent.geometry import compute_bounds
from soundevent.plot.tags import TagColorMapper

from batdetect2.evaluate.types import MatchEvaluation
from batdetect2.plotting.clip_predictions import plot_prediction
from batdetect2.plotting.clips import plot_clip
from batdetect2.preprocess import (
    PreprocessorProtocol,
    get_default_preprocessor,
)

__all__ = [
    "plot_matches",
    "plot_false_positive_match",
    "plot_true_positive_match",
    "plot_false_negative_match",
    "plot_cross_trigger_match",
]

DEFAULT_FALSE_POSITIVE_COLOR = "orange"
DEFAULT_FALSE_NEGATIVE_COLOR = "red"
DEFAULT_TRUE_POSITIVE_COLOR = "green"
DEFAULT_CROSS_TRIGGER_COLOR = "orange"
DEFAULT_ANNOTATION_LINE_STYLE = "-"
DEFAULT_PREDICTION_LINE_STYLE = "--"


def plot_matches(
    matches: List[data.Match],
    clip: data.Clip,
    preprocessor: Optional[PreprocessorProtocol] = None,
    figsize: Optional[Tuple[int, int]] = None,
    ax: Optional[Axes] = None,
    audio_dir: Optional[data.PathLike] = None,
    color_mapper: Optional[TagColorMapper] = None,
    add_colorbar: bool = False,
    add_labels: bool = False,
    add_points: bool = False,
    fill: bool = False,
    spec_cmap: str = "gray",
    false_positive_color: str = DEFAULT_FALSE_POSITIVE_COLOR,
    false_negative_color: str = DEFAULT_FALSE_NEGATIVE_COLOR,
    true_positive_color: str = DEFAULT_TRUE_POSITIVE_COLOR,
    annotation_linestyle: str = DEFAULT_ANNOTATION_LINE_STYLE,
    prediction_linestyle: str = DEFAULT_PREDICTION_LINE_STYLE,
) -> Axes:
    if preprocessor is None:
        preprocessor = get_default_preprocessor()

    ax = plot_clip(
        clip,
        ax=ax,
        figsize=figsize,
        audio_dir=audio_dir,
        add_colorbar=add_colorbar,
        add_labels=add_labels,
        spec_cmap=spec_cmap,
    )

    if color_mapper is None:
        color_mapper = TagColorMapper()

    for match in matches:
        if match.source is None and match.target is not None:
            plot.plot_annotation(
                annotation=match.target,
                ax=ax,
                time_offset=0.004,
                freq_offset=2_000,
                add_points=add_points,
                facecolor="none" if not fill else None,
                color=false_negative_color,
                color_mapper=color_mapper,
                linestyle=annotation_linestyle,
            )
        elif match.target is None and match.source is not None:
            plot_prediction(
                prediction=match.source,
                ax=ax,
                time_offset=0.004,
                freq_offset=2_000,
                add_points=add_points,
                facecolor="none" if not fill else None,
                color=false_positive_color,
                color_mapper=color_mapper,
                linestyle=prediction_linestyle,
            )
        elif match.target is not None and match.source is not None:
            plot.plot_annotation(
                annotation=match.target,
                ax=ax,
                time_offset=0.004,
                freq_offset=2_000,
                add_points=add_points,
                facecolor="none" if not fill else None,
                color=true_positive_color,
                color_mapper=color_mapper,
                linestyle=annotation_linestyle,
            )
            plot_prediction(
                prediction=match.source,
                ax=ax,
                time_offset=0.004,
                freq_offset=2_000,
                add_points=add_points,
                facecolor="none" if not fill else None,
                color=true_positive_color,
                color_mapper=color_mapper,
                linestyle=prediction_linestyle,
            )
        else:
            continue

    return ax


DEFAULT_DURATION = 0.05


def plot_false_positive_match(
    match: MatchEvaluation,
    preprocessor: Optional[PreprocessorProtocol] = None,
    figsize: Optional[Tuple[int, int]] = None,
    ax: Optional[Axes] = None,
    audio_dir: Optional[data.PathLike] = None,
    duration: float = DEFAULT_DURATION,
    add_colorbar: bool = False,
    add_labels: bool = False,
    add_points: bool = False,
    fill: bool = False,
    spec_cmap: str = "gray",
    time_offset: float = 0,
    color: str = DEFAULT_FALSE_POSITIVE_COLOR,
    fontsize: Union[float, str] = "small",
) -> Axes:
    assert match.match.source is not None
    assert match.match.target is None
    sound_event = match.match.source.sound_event
    geometry = sound_event.geometry
    assert geometry is not None

    start_time, _, _, high_freq = compute_bounds(geometry)

    clip = data.Clip(
        start_time=max(start_time - duration / 2, 0),
        end_time=min(
            start_time + duration / 2, sound_event.recording.duration
        ),
        recording=sound_event.recording,
    )

    ax = plot_clip(
        clip,
        preprocessor=preprocessor,
        figsize=figsize,
        ax=ax,
        audio_dir=audio_dir,
        add_colorbar=add_colorbar,
        add_labels=add_labels,
        spec_cmap=spec_cmap,
    )

    plot_prediction(
        match.match.source,
        ax=ax,
        time_offset=time_offset,
        freq_offset=2_000,
        add_points=add_points,
        facecolor="none" if not fill else None,
        alpha=1,
        color=color,
    )

    plt.text(
        start_time,
        high_freq,
        f"False Positive \nScore: {match.pred_score} \nTop Class: {match.pred_class} \nTop Class Score: {match.pred_class_score} ",
        va="top",
        ha="right",
        color=color,
        fontsize=fontsize,
    )

    return ax


def plot_false_negative_match(
    match: MatchEvaluation,
    preprocessor: Optional[PreprocessorProtocol] = None,
    figsize: Optional[Tuple[int, int]] = None,
    ax: Optional[Axes] = None,
    audio_dir: Optional[data.PathLike] = None,
    duration: float = DEFAULT_DURATION,
    add_colorbar: bool = False,
    add_labels: bool = False,
    add_points: bool = False,
    fill: bool = False,
    spec_cmap: str = "gray",
    color: str = DEFAULT_FALSE_NEGATIVE_COLOR,
    fontsize: Union[float, str] = "small",
) -> Axes:
    assert match.match.source is None
    assert match.match.target is not None
    sound_event = match.match.target.sound_event
    geometry = sound_event.geometry
    assert geometry is not None

    start_time, _, _, high_freq = compute_bounds(geometry)

    clip = data.Clip(
        start_time=max(start_time - duration / 2, 0),
        end_time=min(
            start_time + duration / 2, sound_event.recording.duration
        ),
        recording=sound_event.recording,
    )

    ax = plot_clip(
        clip,
        preprocessor=preprocessor,
        figsize=figsize,
        ax=ax,
        audio_dir=audio_dir,
        add_colorbar=add_colorbar,
        add_labels=add_labels,
        spec_cmap=spec_cmap,
    )

    plot.plot_annotation(
        match.match.target,
        ax=ax,
        time_offset=0.001,
        freq_offset=2_000,
        add_points=add_points,
        facecolor="none" if not fill else None,
        alpha=1,
        color=color,
    )

    plt.text(
        start_time,
        high_freq,
        f"False Negative \nClass: {match.gt_class} ",
        va="top",
        ha="right",
        color=color,
        fontsize=fontsize,
    )

    return ax


def plot_true_positive_match(
    match: MatchEvaluation,
    preprocessor: Optional[PreprocessorProtocol] = None,
    figsize: Optional[Tuple[int, int]] = None,
    ax: Optional[Axes] = None,
    audio_dir: Optional[data.PathLike] = None,
    duration: float = DEFAULT_DURATION,
    add_colorbar: bool = False,
    add_labels: bool = False,
    add_points: bool = False,
    fill: bool = False,
    spec_cmap: str = "gray",
    color: str = DEFAULT_TRUE_POSITIVE_COLOR,
    fontsize: Union[float, str] = "small",
    annotation_linestyle: str = DEFAULT_ANNOTATION_LINE_STYLE,
    prediction_linestyle: str = DEFAULT_PREDICTION_LINE_STYLE,
) -> Axes:
    assert match.match.source is not None
    assert match.match.target is not None
    sound_event = match.match.target.sound_event
    geometry = sound_event.geometry
    assert geometry is not None

    start_time, _, _, high_freq = compute_bounds(geometry)

    clip = data.Clip(
        start_time=max(start_time - duration / 2, 0),
        end_time=min(
            start_time + duration / 2, sound_event.recording.duration
        ),
        recording=sound_event.recording,
    )

    ax = plot_clip(
        clip,
        preprocessor=preprocessor,
        figsize=figsize,
        ax=ax,
        audio_dir=audio_dir,
        add_colorbar=add_colorbar,
        add_labels=add_labels,
        spec_cmap=spec_cmap,
    )

    plot.plot_annotation(
        match.match.target,
        ax=ax,
        time_offset=0.001,
        freq_offset=2_000,
        add_points=add_points,
        facecolor="none" if not fill else None,
        alpha=1,
        color=color,
        linestyle=annotation_linestyle,
    )

    plot_prediction(
        match.match.source,
        ax=ax,
        time_offset=0.001,
        freq_offset=2_000,
        add_points=add_points,
        facecolor="none" if not fill else None,
        alpha=1,
        color=color,
        linestyle=prediction_linestyle,
    )

    plt.text(
        start_time,
        high_freq,
        f"True Positive \nClass: {match.gt_class} \nDet Score: {match.pred_score} \nTop Class Score: {match.pred_class_score} ",
        va="top",
        ha="right",
        color=color,
        fontsize=fontsize,
    )

    return ax


def plot_cross_trigger_match(
    match: MatchEvaluation,
    preprocessor: Optional[PreprocessorProtocol] = None,
    figsize: Optional[Tuple[int, int]] = None,
    ax: Optional[Axes] = None,
    audio_dir: Optional[data.PathLike] = None,
    duration: float = DEFAULT_DURATION,
    add_colorbar: bool = False,
    add_labels: bool = False,
    add_points: bool = False,
    fill: bool = False,
    spec_cmap: str = "gray",
    color: str = DEFAULT_CROSS_TRIGGER_COLOR,
    fontsize: Union[float, str] = "small",
    annotation_linestyle: str = DEFAULT_ANNOTATION_LINE_STYLE,
    prediction_linestyle: str = DEFAULT_PREDICTION_LINE_STYLE,
) -> Axes:
    assert match.match.source is not None
    assert match.match.target is not None
    sound_event = match.match.source.sound_event
    geometry = sound_event.geometry
    assert geometry is not None

    start_time, _, _, high_freq = compute_bounds(geometry)

    clip = data.Clip(
        start_time=max(start_time - duration / 2, 0),
        end_time=min(
            start_time + duration / 2, sound_event.recording.duration
        ),
        recording=sound_event.recording,
    )

    ax = plot_clip(
        clip,
        preprocessor=preprocessor,
        figsize=figsize,
        ax=ax,
        audio_dir=audio_dir,
        add_colorbar=add_colorbar,
        add_labels=add_labels,
        spec_cmap=spec_cmap,
    )

    plot.plot_annotation(
        match.match.target,
        ax=ax,
        time_offset=0.001,
        freq_offset=2_000,
        add_points=add_points,
        facecolor="none" if not fill else None,
        alpha=1,
        color=color,
        linestyle=annotation_linestyle,
    )

    plot_prediction(
        match.match.source,
        ax=ax,
        time_offset=0.001,
        freq_offset=2_000,
        add_points=add_points,
        facecolor="none" if not fill else None,
        alpha=1,
        color=color,
        linestyle=prediction_linestyle,
    )

    plt.text(
        start_time,
        high_freq,
        f"Cross Trigger \nTrue Class: {match.gt_class} \nPred Class: {match.pred_class} \nDet Score: {match.pred_score} \nTop Class Score: {match.pred_class_score} ",
        va="top",
        ha="right",
        color=color,
        fontsize=fontsize,
    )

    return ax
