from typing import Optional

from matplotlib import axes, patches
from soundevent.plot import plot_geometry

from batdetect2.evaluate.metrics.detection import ClipEval
from batdetect2.plotting.clips import (
    AudioLoader,
    PreprocessorProtocol,
    plot_clip,
)
from batdetect2.plotting.common import create_ax

__all__ = [
    "plot_clip_detections",
]


def plot_clip_detections(
    clip_eval: ClipEval,
    figsize: tuple[int, int] = (10, 10),
    ax: Optional[axes.Axes] = None,
    audio_loader: Optional[AudioLoader] = None,
    preprocessor: Optional[PreprocessorProtocol] = None,
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
                m.gt.sound_event.geometry,  # type: ignore
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
