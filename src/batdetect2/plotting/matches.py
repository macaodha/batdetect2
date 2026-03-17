from typing import Protocol

from matplotlib.axes import Axes
from soundevent import data, plot
from soundevent.geometry import compute_bounds

from batdetect2.plotting.clips import plot_clip
from batdetect2.typing import (
    AudioLoader,
    Detection,
    PreprocessorProtocol,
)

__all__ = [
    "plot_false_positive_match",
    "plot_true_positive_match",
    "plot_false_negative_match",
    "plot_cross_trigger_match",
]


class MatchProtocol(Protocol):
    clip: data.Clip
    gt: data.SoundEventAnnotation | None
    pred: Detection | None
    score: float
    true_class: str | None


DEFAULT_DURATION = 0.05
DEFAULT_FALSE_POSITIVE_COLOR = "orange"
DEFAULT_FALSE_NEGATIVE_COLOR = "red"
DEFAULT_TRUE_POSITIVE_COLOR = "green"
DEFAULT_CROSS_TRIGGER_COLOR = "orange"
DEFAULT_ANNOTATION_LINE_STYLE = "-"
DEFAULT_PREDICTION_LINE_STYLE = "--"


def plot_false_positive_match(
    match: MatchProtocol,
    audio_loader: AudioLoader | None = None,
    preprocessor: PreprocessorProtocol | None = None,
    figsize: tuple[int, int] | None = None,
    ax: Axes | None = None,
    audio_dir: data.PathLike | None = None,
    duration: float = DEFAULT_DURATION,
    use_score: bool = True,
    add_spectrogram: bool = True,
    add_text: bool = True,
    add_points: bool = False,
    add_title: bool = True,
    fill: bool = False,
    spec_cmap: str = "gray",
    color: str = DEFAULT_FALSE_POSITIVE_COLOR,
    fontsize: float | str = "small",
) -> Axes:
    assert match.pred is not None

    start_time, _, _, high_freq = compute_bounds(match.pred.geometry)

    clip = data.Clip(
        start_time=max(
            start_time - duration / 2,
            0,
        ),
        end_time=min(
            start_time + duration / 2,
            match.clip.recording.duration,
        ),
        recording=match.clip.recording,
    )

    if add_spectrogram:
        ax = plot_clip(
            clip,
            audio_loader=audio_loader,
            preprocessor=preprocessor,
            figsize=figsize,
            ax=ax,
            audio_dir=audio_dir,
            spec_cmap=spec_cmap,
        )

    ax = plot.plot_geometry(
        match.pred.geometry,
        ax=ax,
        add_points=add_points,
        facecolor="none" if not fill else None,
        alpha=match.score if use_score else 1,
        color=color,
    )

    if add_text:
        ax.text(
            start_time,
            high_freq,
            f"score={match.score:.2f}",
            va="top",
            ha="right",
            color=color,
            fontsize=fontsize,
        )

    if add_title:
        ax.set_title("False Positive")

    return ax


def plot_false_negative_match(
    match: MatchProtocol,
    audio_loader: AudioLoader | None = None,
    preprocessor: PreprocessorProtocol | None = None,
    figsize: tuple[int, int] | None = None,
    ax: Axes | None = None,
    audio_dir: data.PathLike | None = None,
    duration: float = DEFAULT_DURATION,
    add_spectrogram: bool = True,
    add_points: bool = False,
    add_title: bool = True,
    fill: bool = False,
    spec_cmap: str = "gray",
    color: str = DEFAULT_FALSE_NEGATIVE_COLOR,
) -> Axes:
    assert match.gt is not None

    geometry = match.gt.sound_event.geometry
    assert geometry is not None

    start_time = compute_bounds(geometry)[0]

    clip = data.Clip(
        start_time=max(
            start_time - duration / 2,
            0,
        ),
        end_time=min(
            start_time + duration / 2,
            match.clip.recording.duration,
        ),
        recording=match.clip.recording,
    )

    if add_spectrogram:
        ax = plot_clip(
            clip,
            audio_loader=audio_loader,
            preprocessor=preprocessor,
            figsize=figsize,
            ax=ax,
            audio_dir=audio_dir,
            spec_cmap=spec_cmap,
        )

    ax = plot.plot_geometry(
        geometry,
        ax=ax,
        add_points=add_points,
        facecolor="none" if not fill else None,
        alpha=1,
        color=color,
    )

    if add_title:
        ax.set_title("False Negative")

    return ax


def plot_true_positive_match(
    match: MatchProtocol,
    preprocessor: PreprocessorProtocol | None = None,
    audio_loader: AudioLoader | None = None,
    figsize: tuple[int, int] | None = None,
    ax: Axes | None = None,
    audio_dir: data.PathLike | None = None,
    duration: float = DEFAULT_DURATION,
    use_score: bool = True,
    add_spectrogram: bool = True,
    add_points: bool = False,
    add_text: bool = True,
    fill: bool = False,
    spec_cmap: str = "gray",
    color: str = DEFAULT_TRUE_POSITIVE_COLOR,
    fontsize: float | str = "small",
    annotation_linestyle: str = DEFAULT_ANNOTATION_LINE_STYLE,
    prediction_linestyle: str = DEFAULT_PREDICTION_LINE_STYLE,
    add_title: bool = True,
) -> Axes:
    assert match.gt is not None
    assert match.pred is not None

    geometry = match.gt.sound_event.geometry
    assert geometry is not None

    start_time, _, _, high_freq = compute_bounds(geometry)

    clip = data.Clip(
        start_time=max(
            start_time - duration / 2,
            0,
        ),
        end_time=min(
            start_time + duration / 2,
            match.clip.recording.duration,
        ),
        recording=match.clip.recording,
    )

    if add_spectrogram:
        ax = plot_clip(
            clip,
            ax=ax,
            audio_loader=audio_loader,
            preprocessor=preprocessor,
            figsize=figsize,
            audio_dir=audio_dir,
            spec_cmap=spec_cmap,
        )

    ax = plot.plot_geometry(
        geometry,
        ax=ax,
        add_points=add_points,
        facecolor="none" if not fill else None,
        alpha=1,
        color=color,
        linestyle=annotation_linestyle,
    )

    plot.plot_geometry(
        match.pred.geometry,
        ax=ax,
        add_points=add_points,
        facecolor="none" if not fill else None,
        alpha=match.score if use_score else 1,
        color=color,
        linestyle=prediction_linestyle,
    )

    if add_text:
        ax.text(
            start_time,
            high_freq,
            f"score={match.score:.2f}",
            va="top",
            ha="right",
            color=color,
            fontsize=fontsize,
        )

    if add_title:
        ax.set_title("True Positive")

    return ax


def plot_cross_trigger_match(
    match: MatchProtocol,
    preprocessor: PreprocessorProtocol | None = None,
    audio_loader: AudioLoader | None = None,
    figsize: tuple[int, int] | None = None,
    ax: Axes | None = None,
    audio_dir: data.PathLike | None = None,
    duration: float = DEFAULT_DURATION,
    use_score: bool = True,
    add_spectrogram: bool = True,
    add_points: bool = False,
    add_text: bool = True,
    add_title: bool = True,
    fill: bool = False,
    spec_cmap: str = "gray",
    color: str = DEFAULT_CROSS_TRIGGER_COLOR,
    fontsize: float | str = "small",
    annotation_linestyle: str = DEFAULT_ANNOTATION_LINE_STYLE,
    prediction_linestyle: str = DEFAULT_PREDICTION_LINE_STYLE,
) -> Axes:
    assert match.gt is not None
    assert match.pred is not None

    geometry = match.gt.sound_event.geometry
    assert geometry is not None

    start_time, _, _, high_freq = compute_bounds(geometry)

    clip = data.Clip(
        start_time=max(
            start_time - duration / 2,
            0,
        ),
        end_time=min(
            start_time + duration / 2,
            match.clip.recording.duration,
        ),
        recording=match.clip.recording,
    )

    if add_spectrogram:
        ax = plot_clip(
            clip,
            audio_loader=audio_loader,
            preprocessor=preprocessor,
            figsize=figsize,
            ax=ax,
            audio_dir=audio_dir,
            spec_cmap=spec_cmap,
        )

    ax = plot.plot_geometry(
        geometry,
        ax=ax,
        add_points=add_points,
        facecolor="none" if not fill else None,
        alpha=1,
        color=color,
        linestyle=annotation_linestyle,
    )

    ax = plot.plot_geometry(
        match.pred.geometry,
        ax=ax,
        add_points=add_points,
        facecolor="none" if not fill else None,
        alpha=match.score if use_score else 1,
        color=color,
        linestyle=prediction_linestyle,
    )

    if add_text:
        ax.text(
            start_time,
            high_freq,
            f"score={match.score:.2f}\nclass={match.true_class}",
            va="top",
            ha="right",
            color=color,
            fontsize=fontsize,
        )

    if add_title:
        ax.set_title("Cross Trigger")

    return ax
