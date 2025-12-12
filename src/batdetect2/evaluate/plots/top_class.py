import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Annotated,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Sequence,
    Tuple,
)

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from pydantic import Field
from sklearn import metrics

from batdetect2.audio import AudioConfig, build_audio_loader
from batdetect2.core import Registry
from batdetect2.evaluate.metrics.common import compute_precision_recall
from batdetect2.evaluate.metrics.top_class import (
    ClipEval,
    MatchEval,
    compute_confusion_matrix,
)
from batdetect2.evaluate.plots.base import BasePlot, BasePlotConfig
from batdetect2.plotting.gallery import plot_match_gallery
from batdetect2.plotting.metrics import plot_pr_curve, plot_roc_curve
from batdetect2.preprocess import PreprocessingConfig, build_preprocessor
from batdetect2.typing import AudioLoader, PreprocessorProtocol, TargetProtocol

TopClassPlotter = Callable[[Sequence[ClipEval]], Iterable[Tuple[str, Figure]]]

top_class_plots: Registry[TopClassPlotter, [TargetProtocol]] = Registry(
    name="top_class_plot"
)


class PRCurveConfig(BasePlotConfig):
    name: Literal["pr_curve"] = "pr_curve"
    label: str = "pr_curve"
    title: str | None = "Top Class Precision-Recall Curve"
    ignore_non_predictions: bool = True
    ignore_generic: bool = True


class PRCurve(BasePlot):
    def __init__(
        self,
        *args,
        ignore_non_predictions: bool = True,
        ignore_generic: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.ignore_non_predictions = ignore_non_predictions
        self.ignore_generic = ignore_generic

    def __call__(
        self,
        clip_evaluations: Sequence[ClipEval],
    ) -> Iterable[Tuple[str, Figure]]:
        y_true = []
        y_score = []
        num_positives = 0

        for clip_eval in clip_evaluations:
            for m in clip_eval.matches:
                if m.is_generic and self.ignore_generic:
                    # Ignore gt sounds with unknown class
                    continue

                num_positives += int(m.is_ground_truth)

                if not m.is_prediction and self.ignore_non_predictions:
                    # Ignore non predictions
                    continue

                y_true.append(m.pred_class == m.true_class)
                y_score.append(m.score)

        precision, recall, thresholds = compute_precision_recall(
            y_true,
            y_score,
            num_positives=num_positives,
        )

        fig = self.create_figure()
        ax = fig.subplots()

        plot_pr_curve(precision, recall, thresholds, ax=ax)

        yield self.label, fig

    @top_class_plots.register(PRCurveConfig)
    @staticmethod
    def from_config(config: PRCurveConfig, targets: TargetProtocol):
        return PRCurve.build(
            config=config,
            targets=targets,
            ignore_non_predictions=config.ignore_non_predictions,
            ignore_generic=config.ignore_generic,
        )


class ROCCurveConfig(BasePlotConfig):
    name: Literal["roc_curve"] = "roc_curve"
    label: str = "roc_curve"
    title: str | None = "Top Class ROC Curve"
    ignore_non_predictions: bool = True
    ignore_generic: bool = True


class ROCCurve(BasePlot):
    def __init__(
        self,
        *args,
        ignore_non_predictions: bool = True,
        ignore_generic: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.ignore_non_predictions = ignore_non_predictions
        self.ignore_generic = ignore_generic

    def __call__(
        self,
        clip_evaluations: Sequence[ClipEval],
    ) -> Iterable[Tuple[str, Figure]]:
        y_true = []
        y_score = []

        for clip_eval in clip_evaluations:
            for m in clip_eval.matches:
                if m.is_generic and self.ignore_generic:
                    # Ignore gt sounds with unknown class
                    continue

                if not m.is_prediction and self.ignore_non_predictions:
                    # Ignore non predictions
                    continue

                y_true.append(m.pred_class == m.true_class)
                y_score.append(m.score)

        fpr, tpr, thresholds = metrics.roc_curve(
            y_true,
            y_score,
        )

        fig = self.create_figure()
        ax = fig.subplots()

        plot_roc_curve(fpr, tpr, thresholds, ax=ax)

        yield self.label, fig

    @top_class_plots.register(ROCCurveConfig)
    @staticmethod
    def from_config(config: ROCCurveConfig, targets: TargetProtocol):
        return ROCCurve.build(
            config=config,
            targets=targets,
            ignore_non_predictions=config.ignore_non_predictions,
            ignore_generic=config.ignore_generic,
        )


class ConfusionMatrixConfig(BasePlotConfig):
    name: Literal["confusion_matrix"] = "confusion_matrix"
    title: str | None = "Top Class Confusion Matrix"
    figsize: tuple[int, int] = (10, 10)
    label: str = "confusion_matrix"
    exclude_generic: bool = True
    exclude_noise: bool = False
    noise_class: str = "noise"
    normalize: Literal["true", "pred", "all", "none"] = "true"
    threshold: float = 0.2
    add_colorbar: bool = True
    cmap: str = "Blues"


class ConfusionMatrix(BasePlot):
    def __init__(
        self,
        *args,
        exclude_generic: bool = True,
        exclude_false_positives: bool = True,
        exclude_false_negatives: bool = True,
        exclude_noise: bool = False,
        noise_class: str = "noise",
        add_colorbar: bool = True,
        normalize: Literal["true", "pred", "all", "none"] = "true",
        cmap: str = "Blues",
        threshold: float = 0.2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.exclude_generic = exclude_generic
        self.exclude_false_positives = exclude_false_positives
        self.exclude_false_negatives = exclude_false_negatives
        self.exclude_noise = exclude_noise
        self.noise_class = noise_class
        self.normalize: Literal["true", "pred", "all", "none"] = normalize
        self.add_colorbar = add_colorbar
        self.threshold = threshold
        self.cmap = cmap

    def __call__(
        self,
        clip_evaluations: Sequence[ClipEval],
    ) -> Iterable[Tuple[str, Figure]]:
        cm, labels = compute_confusion_matrix(
            clip_evaluations,
            self.targets,
            threshold=self.threshold,
            normalize=self.normalize,
            exclude_generic=self.exclude_generic,
            exclude_false_positives=self.exclude_false_positives,
            exclude_false_negatives=self.exclude_false_negatives,
            noise_class=self.noise_class,
        )

        fig = self.create_figure()
        ax = fig.subplots()

        metrics.ConfusionMatrixDisplay(cm, display_labels=labels).plot(
            ax=ax,
            xticks_rotation="vertical",
            cmap=self.cmap,
            colorbar=self.add_colorbar,
            values_format=".2f",
        )

        yield self.label, fig

    @top_class_plots.register(ConfusionMatrixConfig)
    @staticmethod
    def from_config(config: ConfusionMatrixConfig, targets: TargetProtocol):
        return ConfusionMatrix.build(
            config=config,
            targets=targets,
            exclude_generic=config.exclude_generic,
            exclude_noise=config.exclude_noise,
            noise_class=config.noise_class,
            add_colorbar=config.add_colorbar,
            normalize=config.normalize,
            cmap=config.cmap,
        )


class ExampleClassificationPlotConfig(BasePlotConfig):
    name: Literal["example_classification"] = "example_classification"
    label: str = "example_classification"
    title: str | None = "Example Classification"
    num_examples: int = 4
    threshold: float = 0.2
    audio: AudioConfig = Field(default_factory=AudioConfig)
    preprocessing: PreprocessingConfig = Field(
        default_factory=PreprocessingConfig
    )


class ExampleClassificationPlot(BasePlot):
    def __init__(
        self,
        *args,
        num_examples: int = 4,
        threshold: float = 0.2,
        audio_loader: AudioLoader,
        preprocessor: PreprocessorProtocol,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_examples = num_examples
        self.audio_loader = audio_loader
        self.threshold = threshold
        self.preprocessor = preprocessor
        self.num_examples = num_examples

    def __call__(
        self,
        clip_evaluations: Sequence[ClipEval],
    ) -> Iterable[Tuple[str, Figure]]:
        grouped = group_matches(clip_evaluations, threshold=self.threshold)

        for class_name, matches in grouped.items():
            true_positives: List[MatchEval] = get_binned_sample(
                matches.true_positives,
                n_examples=self.num_examples,
            )

            false_positives: List[MatchEval] = get_binned_sample(
                matches.false_positives,
                n_examples=self.num_examples,
            )

            false_negatives: List[MatchEval] = random.sample(
                matches.false_negatives,
                k=min(self.num_examples, len(matches.false_negatives)),
            )

            cross_triggers: List[MatchEval] = get_binned_sample(
                matches.cross_triggers, n_examples=self.num_examples
            )

            fig = self.create_figure()

            fig = plot_match_gallery(
                true_positives,
                false_positives,
                false_negatives,
                cross_triggers,
                preprocessor=self.preprocessor,
                audio_loader=self.audio_loader,
                n_examples=self.num_examples,
                fig=fig,
            )

            if self.title is not None:
                fig.suptitle(f"{self.title}: {class_name}")
            else:
                fig.suptitle(class_name)

            yield f"{self.label}/{class_name}", fig

            plt.close(fig)

    @top_class_plots.register(ExampleClassificationPlotConfig)
    @staticmethod
    def from_config(
        config: ExampleClassificationPlotConfig,
        targets: TargetProtocol,
    ):
        return ExampleClassificationPlot.build(
            config=config,
            targets=targets,
            num_examples=config.num_examples,
            threshold=config.threshold,
            audio_loader=build_audio_loader(config.audio),
            preprocessor=build_preprocessor(config.preprocessing),
        )


TopClassPlotConfig = Annotated[
    PRCurveConfig
    | ROCCurveConfig
    | ConfusionMatrixConfig
    | ExampleClassificationPlotConfig,
    Field(discriminator="name"),
]


def build_top_class_plotter(
    config: TopClassPlotConfig,
    targets: TargetProtocol,
) -> TopClassPlotter:
    return top_class_plots.build(config, targets)


@dataclass
class ClassMatches:
    false_positives: List[MatchEval] = field(default_factory=list)
    false_negatives: List[MatchEval] = field(default_factory=list)
    true_positives: List[MatchEval] = field(default_factory=list)
    cross_triggers: List[MatchEval] = field(default_factory=list)


def group_matches(
    clip_evals: Sequence[ClipEval],
    threshold: float = 0.2,
) -> Dict[str, ClassMatches]:
    class_examples = defaultdict(ClassMatches)

    for clip_eval in clip_evals:
        for match in clip_eval.matches:
            gt_class = match.true_class
            pred_class = match.pred_class
            is_pred = match.score >= threshold

            if not is_pred and gt_class is not None:
                class_examples[gt_class].false_negatives.append(match)
                continue

            if not is_pred:
                continue

            if gt_class is None:
                class_examples[pred_class].false_positives.append(match)
                continue

            if gt_class != pred_class:
                class_examples[pred_class].cross_triggers.append(match)
                continue

            class_examples[gt_class].true_positives.append(match)

    return class_examples


def get_binned_sample(matches: List[MatchEval], n_examples: int = 5):
    if len(matches) < n_examples:
        return matches

    indices, pred_scores = zip(
        *[(index, match.score) for index, match in enumerate(matches)],
        strict=False,
    )

    bins = pd.qcut(pred_scores, q=n_examples, labels=False, duplicates="drop")
    df = pd.DataFrame({"indices": indices, "bins": bins})
    sample = df.groupby("bins").sample(1)
    return [matches[ind] for ind in sample["indices"]]
