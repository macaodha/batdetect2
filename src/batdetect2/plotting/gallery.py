from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from batdetect2.plotting.matches import (
    MatchProtocol,
    plot_cross_trigger_match,
    plot_false_negative_match,
    plot_false_positive_match,
    plot_true_positive_match,
)
from batdetect2.typing.preprocess import AudioLoader, PreprocessorProtocol

__all__ = ["plot_match_gallery"]


def plot_match_gallery(
    true_positives: Sequence[MatchProtocol],
    false_positives: Sequence[MatchProtocol],
    false_negatives: Sequence[MatchProtocol],
    cross_triggers: Sequence[MatchProtocol],
    audio_loader: AudioLoader | None = None,
    preprocessor: PreprocessorProtocol | None = None,
    n_examples: int = 5,
    duration: float = 0.1,
    fig: Figure | None = None,
):
    if fig is None:
        fig = plt.figure(figsize=(20, 20))

    axes = fig.subplots(
        nrows=4,
        ncols=n_examples,
        sharex="none",
        sharey="row",
    )

    for tp_ax, tp_match in zip(axes[0], true_positives[:n_examples], strict=False):
        try:
            plot_true_positive_match(
                tp_match,
                ax=tp_ax,
                audio_loader=audio_loader,
                preprocessor=preprocessor,
                duration=duration,
            )
        except (
            ValueError,
            AssertionError,
            RuntimeError,
            FileNotFoundError,
        ):
            continue

    for fp_ax, fp_match in zip(axes[1], false_positives[:n_examples], strict=False):
        try:
            plot_false_positive_match(
                fp_match,
                ax=fp_ax,
                audio_loader=audio_loader,
                preprocessor=preprocessor,
                duration=duration,
            )
        except (
            ValueError,
            AssertionError,
            RuntimeError,
            FileNotFoundError,
        ):
            continue

    for fn_ax, fn_match in zip(axes[2], false_negatives[:n_examples], strict=False):
        try:
            plot_false_negative_match(
                fn_match,
                ax=fn_ax,
                audio_loader=audio_loader,
                preprocessor=preprocessor,
                duration=duration,
            )
        except (
            ValueError,
            AssertionError,
            RuntimeError,
            FileNotFoundError,
        ):
            continue

    for ct_ax, ct_match in zip(axes[3], cross_triggers[:n_examples], strict=False):
        try:
            plot_cross_trigger_match(
                ct_match,
                ax=ct_ax,
                audio_loader=audio_loader,
                preprocessor=preprocessor,
                duration=duration,
            )
        except (
            ValueError,
            AssertionError,
            RuntimeError,
            FileNotFoundError,
        ):
            continue

    fig.tight_layout()

    return fig
