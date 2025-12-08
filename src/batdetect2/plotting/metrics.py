from typing import Dict, Optional, Tuple, Union

import numpy as np
import seaborn as sns
from cycler import cycler
from matplotlib import axes

from batdetect2.evaluate.metrics.common import _average_precision
from batdetect2.plotting.common import create_ax


def set_default_styler(ax: axes.Axes) -> axes.Axes:
    color_cycler = cycler(color=sns.color_palette("muted"))
    style_cycler = cycler(linestyle=["-", "--", ":"]) * cycler(
        marker=["o", "s", "^"]
    )
    custom_cycler = color_cycler * len(style_cycler) + style_cycler * len(
        color_cycler
    )

    ax.set_prop_cycle(custom_cycler)
    return ax


def set_default_style(ax: axes.Axes) -> axes.Axes:
    ax = set_default_styler(ax)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    return ax


def plot_pr_curve(
    precision: np.ndarray,
    recall: np.ndarray,
    thresholds: np.ndarray,
    ax: axes.Axes | None = None,
    figsize: Tuple[int, int] | None = None,
    color: str | Tuple[float, float, float] | None = None,
    add_labels: bool = True,
    add_legend: bool = False,
    marker: str | Tuple[int, int, float] | None = "o",
    markeredgecolor: str | Tuple[float, float, float] | None = None,
    markersize: float | None = None,
    linestyle: str | Tuple[int, ...] | None = None,
    linewidth: float | None = None,
    label: str = "PR Curve",
) -> axes.Axes:
    ax = create_ax(ax=ax, figsize=figsize)

    ax = set_default_style(ax)

    ax.plot(
        recall,
        precision,
        color=color,
        label=label,
        marker=marker,
        markeredgecolor=markeredgecolor,
        markevery=_get_marker_positions(thresholds),
        markersize=markersize,
        linestyle=linestyle,
        linewidth=linewidth,
    )

    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)

    if add_legend:
        ax.legend()

    if add_labels:
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")

    return ax


def plot_pr_curves(
    data: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    ax: axes.Axes | None = None,
    figsize: Tuple[int, int] | None = None,
    add_legend: bool = True,
    add_labels: bool = True,
    include_ap: bool = False,
) -> axes.Axes:
    ax = create_ax(ax=ax, figsize=figsize)
    ax = set_default_style(ax)

    for name, (precision, recall, thresholds) in data.items():
        label = name

        if include_ap:
            label += f" (AP={_average_precision(recall, precision):.2f})"

        ax.plot(
            recall,
            precision,
            label=label,
            markevery=_get_marker_positions(thresholds),
        )

    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)

    if add_labels:
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")

    if add_legend:
        ax.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0.0,
        )
    return ax


def plot_threshold_precision_curve(
    threshold: np.ndarray,
    precision: np.ndarray,
    ax: axes.Axes | None = None,
    figsize: Tuple[int, int] | None = None,
    add_labels: bool = True,
):
    ax = create_ax(ax=ax, figsize=figsize)

    ax = set_default_style(ax)

    ax.plot(threshold, precision, markevery=_get_marker_positions(threshold))

    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)

    if add_labels:
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Precision")

    return ax


def plot_threshold_precision_curves(
    data: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    ax: axes.Axes | None = None,
    figsize: Tuple[int, int] | None = None,
    add_legend: bool = True,
    add_labels: bool = True,
):
    ax = create_ax(ax=ax, figsize=figsize)
    ax = set_default_style(ax)

    for name, (precision, _, thresholds) in data.items():
        ax.plot(
            thresholds,
            precision,
            label=name,
            markevery=_get_marker_positions(thresholds),
        )

    if add_legend:
        ax.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0.0,
        )

    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)

    if add_labels:
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Precision")

    return ax


def plot_threshold_recall_curve(
    threshold: np.ndarray,
    recall: np.ndarray,
    ax: axes.Axes | None = None,
    figsize: Tuple[int, int] | None = None,
    add_labels: bool = True,
):
    ax = create_ax(ax=ax, figsize=figsize)

    ax = set_default_style(ax)

    ax.plot(threshold, recall, markevery=_get_marker_positions(threshold))

    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)

    if add_labels:
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Recall")

    return ax


def plot_threshold_recall_curves(
    data: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    ax: axes.Axes | None = None,
    figsize: Tuple[int, int] | None = None,
    add_legend: bool = True,
    add_labels: bool = True,
):
    ax = create_ax(ax=ax, figsize=figsize)
    ax = set_default_style(ax)

    for name, (_, recall, thresholds) in data.items():
        ax.plot(
            thresholds,
            recall,
            label=name,
            markevery=_get_marker_positions(thresholds),
        )

    if add_legend:
        ax.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0.0,
        )

    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)

    if add_labels:
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Recall")

    return ax


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    thresholds: np.ndarray,
    ax: axes.Axes | None = None,
    figsize: Tuple[int, int] | None = None,
    add_labels: bool = True,
) -> axes.Axes:
    ax = create_ax(ax=ax, figsize=figsize)

    ax = set_default_style(ax)

    ax.plot(
        fpr,
        tpr,
        markevery=_get_marker_positions(thresholds),
    )

    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)

    if add_labels:
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")

    return ax


def plot_roc_curves(
    data: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    ax: axes.Axes | None = None,
    figsize: Tuple[int, int] | None = None,
    add_legend: bool = True,
    add_labels: bool = True,
) -> axes.Axes:
    ax = create_ax(ax=ax, figsize=figsize)
    ax = set_default_style(ax)

    for name, (fpr, tpr, thresholds) in data.items():
        ax.plot(
            fpr,
            tpr,
            label=name,
            markevery=_get_marker_positions(thresholds),
        )

    if add_legend:
        ax.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0.0,
        )

    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)

    if add_labels:
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")

    return ax


def _get_marker_positions(
    thresholds: np.ndarray,
    n_points: int = 11,
) -> np.ndarray:
    size = len(thresholds)
    cut_points = np.linspace(0, 1, n_points)
    indices = np.searchsorted(thresholds[::-1], cut_points)
    return np.clip(size - indices, 0, size - 1)  # type: ignore
