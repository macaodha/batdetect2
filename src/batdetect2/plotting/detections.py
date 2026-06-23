import numpy as np
from matplotlib import axes, patches
from soundevent.geometry import compute_bounds
from soundevent.plot import plot_geometry

from batdetect2.evaluate.metrics.detection import ClipEval
from batdetect2.plotting.clips import (
    AudioLoader,
    PreprocessorProtocol,
    plot_clip,
)
from batdetect2.plotting.common import create_ax
from batdetect2.postprocess import ClipDetections, Detection

__all__ = [
    "plot_clip_evaluation",
    "plot_detection",
]


def plot_detection(
    detection: Detection,
    figsize: tuple[int, int] = (10, 10),
    ax: axes.Axes | None = None,
    fill: bool = False,
    linewidth: float = 1.0,
    linestyle: str = "--",
    color: str = "red",
    show_class: bool = True,
    class_names: list[str] | None = None,
    fontsize: float | str = "small",
):
    ax = create_ax(figsize=figsize, ax=ax)

    plot_geometry(
        detection.geometry,
        ax=ax,
        add_points=False,
        facecolor="none" if not fill else color,
        alpha=detection.detection_score,
        linewidth=linewidth,
        linestyle=linestyle,
        color=color,
    )

    if not show_class:
        return ax

    start_time, low_freq, _, _ = compute_bounds(detection.geometry)

    top_class = np.argmax(detection.class_scores)
    score = detection.class_scores[top_class]

    if class_names is not None:
        class_name = class_names[top_class]
    else:
        class_name = f"class {top_class}"

    ax.text(
        start_time,
        low_freq,
        f"{class_name}={score:.2f}",
        va="top",
        ha="left",
        color=color,
        fontsize=fontsize,
        alpha=detection.detection_score,
    )
    return ax


def plot_clip_detection(
    clip_detections: ClipDetections,
    figsize: tuple[int, int] = (10, 10),
    ax: axes.Axes | None = None,
    audio_loader: AudioLoader | None = None,
    preprocessor: PreprocessorProtocol | None = None,
    threshold: float | None = None,
    spec_cmap: str = "gray",
    fill: bool = False,
    linewidth: float = 1.0,
    linestyle: str = "--",
    color: str = "red",
    show_class: bool = True,
    class_names: list[str] | None = None,
    fontsize: float | str = "small",
):
    ax = create_ax(figsize=figsize, ax=ax)

    plot_clip(
        clip_detections.clip,
        audio_loader=audio_loader,
        preprocessor=preprocessor,
        ax=ax,
        spec_cmap=spec_cmap,
    )

    for detection in clip_detections.detections:
        if threshold and detection.detection_score < threshold:
            continue

        ax = plot_detection(
            detection,
            ax=ax,
            class_names=class_names,
            fontsize=fontsize,
            fill=fill,
            linewidth=linewidth,
            linestyle=linestyle,
            color=color,
            show_class=show_class,
        )

    return ax


def plot_clip_evaluation(
    clip_eval: ClipEval,
    figsize: tuple[int, int] = (10, 10),
    ax: axes.Axes | None = None,
    audio_loader: AudioLoader | None = None,
    preprocessor: PreprocessorProtocol | None = None,
    threshold: float = 0.2,
    add_legend: bool = True,
    add_title: bool = True,
    fill: bool = False,
    linewidth: float = 1.0,
    gt_color: str = "green",
    gt_linestyle: str = "-",
    true_pred_color: str = "yellow",
    true_pred_linestyle: str = "--",
    false_pred_color: str = "blue",
    false_pred_linestyle: str = "-",
    missed_gt_color: str = "red",
    missed_gt_linestyle: str = "-",
) -> axes.Axes:
    ax = create_ax(figsize=figsize, ax=ax)

    plot_clip(
        clip_eval.clip,
        audio_loader=audio_loader,
        preprocessor=preprocessor,
        ax=ax,
    )

    for m in clip_eval.matches:
        is_match = (
            m.pred is not None and m.gt is not None and m.score >= threshold
        )

        if m.pred is not None:
            color = true_pred_color if is_match else false_pred_color
            plot_geometry(
                m.pred.geometry,
                ax=ax,
                add_points=False,
                facecolor="none" if not fill else color,
                alpha=m.pred.detection_score,
                linewidth=linewidth,
                linestyle=true_pred_linestyle
                if is_match
                else missed_gt_linestyle,
                color=color,
            )

        if m.gt is not None:
            color = gt_color if is_match else missed_gt_color
            plot_geometry(
                m.gt.sound_event.geometry,
                ax=ax,
                add_points=False,
                linewidth=linewidth,
                facecolor="none" if not fill else color,
                linestyle=gt_linestyle if is_match else false_pred_linestyle,
                color=color,
            )

    if add_title:
        ax.set_title(clip_eval.clip.recording.path.name)

    if add_legend:
        ax.legend(
            handles=[
                patches.Patch(
                    label="found GT",
                    edgecolor=gt_color,
                    facecolor="none" if not fill else gt_color,
                    linestyle=gt_linestyle,
                ),
                patches.Patch(
                    label="missed GT",
                    edgecolor=missed_gt_color,
                    facecolor="none" if not fill else missed_gt_color,
                    linestyle=missed_gt_linestyle,
                ),
                patches.Patch(
                    label="true Det",
                    edgecolor=true_pred_color,
                    facecolor="none" if not fill else true_pred_color,
                    linestyle=true_pred_linestyle,
                ),
                patches.Patch(
                    label="false Det",
                    edgecolor=false_pred_color,
                    facecolor="none" if not fill else false_pred_color,
                    linestyle=false_pred_linestyle,
                ),
            ]
        )

    return ax
