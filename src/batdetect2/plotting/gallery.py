from typing import List, Optional

import matplotlib.pyplot as plt

from batdetect2.plotting.matches import (
    plot_cross_trigger_match,
    plot_false_negative_match,
    plot_false_positive_match,
    plot_true_positive_match,
)
from batdetect2.typing.evaluate import MatchEvaluation
from batdetect2.typing.preprocess import AudioLoader, PreprocessorProtocol

__all__ = ["plot_match_gallery"]


def plot_match_gallery(
    true_positives: List[MatchEvaluation],
    false_positives: List[MatchEvaluation],
    false_negatives: List[MatchEvaluation],
    cross_triggers: List[MatchEvaluation],
    audio_loader: Optional[AudioLoader] = None,
    preprocessor: Optional[PreprocessorProtocol] = None,
    n_examples: int = 5,
    duration: float = 0.1,
):
    fig = plt.figure(figsize=(20, 20))

    for index, match in enumerate(true_positives[:n_examples]):
        ax = plt.subplot(4, n_examples, index + 1)
        try:
            plot_true_positive_match(
                match,
                ax=ax,
                audio_loader=audio_loader,
                preprocessor=preprocessor,
                duration=duration,
            )
        except (ValueError, AssertionError, RuntimeError, FileNotFoundError):
            continue

    for index, match in enumerate(false_positives[:n_examples]):
        ax = plt.subplot(4, n_examples, n_examples + index + 1)
        try:
            plot_false_positive_match(
                match,
                ax=ax,
                audio_loader=audio_loader,
                preprocessor=preprocessor,
                duration=duration,
            )
        except (ValueError, AssertionError, RuntimeError, FileNotFoundError):
            continue

    for index, match in enumerate(false_negatives[:n_examples]):
        ax = plt.subplot(4, n_examples, 2 * n_examples + index + 1)
        try:
            plot_false_negative_match(
                match,
                ax=ax,
                audio_loader=audio_loader,
                preprocessor=preprocessor,
                duration=duration,
            )
        except (ValueError, AssertionError, RuntimeError, FileNotFoundError):
            continue

    for index, match in enumerate(cross_triggers[:n_examples]):
        ax = plt.subplot(4, n_examples, 3 * n_examples + index + 1)
        try:
            plot_cross_trigger_match(
                match,
                ax=ax,
                audio_loader=audio_loader,
                preprocessor=preprocessor,
                duration=duration,
            )
        except (ValueError, AssertionError, RuntimeError, FileNotFoundError):
            continue

    return fig
