import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List

import matplotlib.pyplot as plt
import pandas as pd

from batdetect2 import plotting
from batdetect2.typing.evaluate import MatchEvaluation
from batdetect2.typing.preprocess import PreprocessorProtocol


@dataclass
class ClassExamples:
    false_positives: List[MatchEvaluation] = field(default_factory=list)
    false_negatives: List[MatchEvaluation] = field(default_factory=list)
    true_positives: List[MatchEvaluation] = field(default_factory=list)
    cross_triggers: List[MatchEvaluation] = field(default_factory=list)


def plot_example_gallery(
    matches: List[MatchEvaluation],
    preprocessor: PreprocessorProtocol,
    n_examples: int = 5,
):
    class_examples = defaultdict(ClassExamples)

    for match in matches:
        gt_class = match.gt_class
        pred_class = match.pred_class

        if pred_class is None:
            class_examples[gt_class].false_negatives.append(match)
            continue

        if gt_class is None:
            class_examples[pred_class].false_positives.append(match)
            continue

        if gt_class != pred_class:
            class_examples[gt_class].cross_triggers.append(match)
            class_examples[pred_class].cross_triggers.append(match)
            continue

        class_examples[gt_class].true_positives.append(match)

    for class_name, examples in class_examples.items():
        true_positives = get_binned_sample(
            examples.true_positives,
            n_examples=n_examples,
        )

        false_positives = get_binned_sample(
            examples.false_positives,
            n_examples=n_examples,
        )

        false_negatives = random.sample(
            examples.false_negatives,
            k=min(n_examples, len(examples.false_negatives)),
        )

        cross_triggers = get_binned_sample(
            examples.cross_triggers,
            n_examples=n_examples,
        )

        fig = plot_class_examples(
            true_positives,
            false_positives,
            false_negatives,
            cross_triggers,
            preprocessor=preprocessor,
            n_examples=n_examples,
        )

        yield class_name, fig

        plt.close(fig)


def plot_class_examples(
    true_positives: List[MatchEvaluation],
    false_positives: List[MatchEvaluation],
    false_negatives: List[MatchEvaluation],
    cross_triggers: List[MatchEvaluation],
    preprocessor: PreprocessorProtocol,
    n_examples: int = 5,
    duration: float = 0.1,
):
    fig = plt.figure(figsize=(20, 20))

    for index, match in enumerate(true_positives[:n_examples]):
        ax = plt.subplot(4, n_examples, index + 1)
        try:
            plotting.plot_true_positive_match(
                match,
                ax=ax,
                preprocessor=preprocessor,
                duration=duration,
            )
        except (ValueError, AssertionError):
            continue

    for index, match in enumerate(false_positives[:n_examples]):
        ax = plt.subplot(4, n_examples, n_examples + index + 1)
        try:
            plotting.plot_false_positive_match(
                match,
                ax=ax,
                preprocessor=preprocessor,
                duration=duration,
            )
        except (ValueError, AssertionError):
            continue

    for index, match in enumerate(false_negatives[:n_examples]):
        ax = plt.subplot(4, n_examples, 2 * n_examples + index + 1)
        try:
            plotting.plot_false_negative_match(
                match,
                ax=ax,
                preprocessor=preprocessor,
                duration=duration,
            )
        except (ValueError, AssertionError):
            continue

    for index, match in enumerate(cross_triggers[:n_examples]):
        ax = plt.subplot(4, n_examples, 3 * n_examples + index + 1)
        try:
            plotting.plot_cross_trigger_match(
                match,
                ax=ax,
                preprocessor=preprocessor,
                duration=duration,
            )
        except (ValueError, AssertionError):
            continue

    return fig


def get_binned_sample(matches: List[MatchEvaluation], n_examples: int = 5):
    if len(matches) < n_examples:
        return matches

    indices, pred_scores = zip(
        *[
            (index, match.pred_class_scores[pred_class])
            for index, match in enumerate(matches)
            if (pred_class := match.pred_class) is not None
        ]
    )

    bins = pd.qcut(pred_scores, q=n_examples, labels=False, duplicates="drop")
    df = pd.DataFrame({"indices": indices, "bins": bins})
    sample = df.groupby("bins").sample(1)
    return [matches[ind] for ind in sample["indices"]]
