import random
from typing import (
    Annotated,
    Callable,
    Iterable,
    Literal,
    Sequence,
    Tuple,
)

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from pydantic import Field
from sklearn import metrics

from batdetect2.audio import AudioConfig, build_audio_loader
from batdetect2.audio.types import AudioLoader
from batdetect2.core import ImportConfig, Registry, add_import_config
from batdetect2.evaluate.metrics.common import compute_precision_recall
from batdetect2.evaluate.metrics.detection import ClipEval
from batdetect2.evaluate.plots.base import BasePlot, BasePlotConfig
from batdetect2.plotting.detections import plot_clip_detections
from batdetect2.plotting.metrics import plot_pr_curve, plot_roc_curve
from batdetect2.preprocess import PreprocessingConfig, build_preprocessor
from batdetect2.preprocess.types import PreprocessorProtocol
from batdetect2.targets.types import TargetProtocol

DetectionPlotter = Callable[[Sequence[ClipEval]], Iterable[Tuple[str, Figure]]]

detection_plots: Registry[DetectionPlotter, [TargetProtocol]] = Registry(
    name="detection_plot"
)


@add_import_config(detection_plots)
class DetectionPlotImportConfig(ImportConfig):
    """Use any callable as a detection plot.

    Set ``name="import"`` and provide a ``target`` pointing to any
    callable to use it instead of a built-in option.
    """

    name: Literal["import"] = "import"


class PRCurveConfig(BasePlotConfig):
    name: Literal["pr_curve"] = "pr_curve"
    label: str = "pr_curve"
    title: str | None = "Detection Precision-Recall Curve"
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
        clip_evals: Sequence[ClipEval],
    ) -> Iterable[Tuple[str, Figure]]:
        y_true = []
        y_score = []
        num_positives = 0

        for clip_eval in clip_evals:
            for m in clip_eval.matches:
                num_positives += int(m.is_ground_truth)

                # Ignore matches that don't correspond to a prediction
                if not m.is_prediction and self.ignore_non_predictions:
                    continue

                y_true.append(m.is_ground_truth)
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

    @detection_plots.register(PRCurveConfig)
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
    title: str | None = "Detection ROC Curve"
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
                if not m.is_prediction and self.ignore_non_predictions:
                    # Ignore matches that don't correspond to a prediction
                    continue

                y_true.append(m.is_ground_truth)
                y_score.append(m.score)

        fpr, tpr, thresholds = metrics.roc_curve(
            y_true,
            y_score,
        )

        fig = self.create_figure()
        ax = fig.subplots()

        plot_roc_curve(fpr, tpr, thresholds, ax=ax)

        yield self.label, fig

    @detection_plots.register(ROCCurveConfig)
    @staticmethod
    def from_config(config: ROCCurveConfig, targets: TargetProtocol):
        return ROCCurve.build(
            config=config,
            targets=targets,
            ignore_non_predictions=config.ignore_non_predictions,
            ignore_generic=config.ignore_generic,
        )


class ScoreDistributionPlotConfig(BasePlotConfig):
    name: Literal["score_distribution"] = "score_distribution"
    label: str = "score_distribution"
    title: str | None = "Detection Score Distribution"
    ignore_non_predictions: bool = True
    ignore_generic: bool = True


class ScoreDistributionPlot(BasePlot):
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
                if not m.is_prediction and self.ignore_non_predictions:
                    # Ignore matches that don't correspond to a prediction
                    continue

                y_true.append(m.is_ground_truth)
                y_score.append(m.score)

        df = pd.DataFrame({"is_true": y_true, "score": y_score})

        fig = self.create_figure()
        ax = fig.subplots()

        sns.histplot(
            data=df,
            x="score",
            binwidth=0.025,
            binrange=(0, 1),
            hue="is_true",
            ax=ax,
            stat="probability",
            common_norm=False,
        )

        yield self.label, fig

    @detection_plots.register(ScoreDistributionPlotConfig)
    @staticmethod
    def from_config(
        config: ScoreDistributionPlotConfig, targets: TargetProtocol
    ):
        return ScoreDistributionPlot.build(
            config=config,
            targets=targets,
            ignore_non_predictions=config.ignore_non_predictions,
            ignore_generic=config.ignore_generic,
        )


class ExampleDetectionPlotConfig(BasePlotConfig):
    name: Literal["example_detection"] = "example_detection"
    label: str = "example_detection"
    title: str | None = "Example Detection"
    figsize: tuple[int, int] = (10, 4)
    num_examples: int = 5
    threshold: float = 0.2
    audio: AudioConfig = Field(default_factory=AudioConfig)
    preprocessing: PreprocessingConfig = Field(
        default_factory=PreprocessingConfig
    )


class ExampleDetectionPlot(BasePlot):
    def __init__(
        self,
        *args,
        num_examples: int = 5,
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

    def __call__(
        self,
        clip_evaluations: Sequence[ClipEval],
    ) -> Iterable[Tuple[str, Figure]]:
        sample = clip_evaluations

        if self.num_examples < len(sample):
            sample = random.sample(sample, self.num_examples)

        for num_example, clip_eval in enumerate(sample):
            fig = self.create_figure()
            ax = fig.subplots()

            plot_clip_detections(
                clip_eval,
                ax=ax,
                audio_loader=self.audio_loader,
                preprocessor=self.preprocessor,
            )

            yield f"{self.label}/example_{num_example}", fig

            plt.close(fig)

    @detection_plots.register(ExampleDetectionPlotConfig)
    @staticmethod
    def from_config(
        config: ExampleDetectionPlotConfig,
        targets: TargetProtocol,
    ):
        return ExampleDetectionPlot.build(
            config=config,
            targets=targets,
            num_examples=config.num_examples,
            audio_loader=build_audio_loader(config.audio),
            preprocessor=build_preprocessor(config.preprocessing),
        )


DetectionPlotConfig = Annotated[
    PRCurveConfig
    | ROCCurveConfig
    | ScoreDistributionPlotConfig
    | ExampleDetectionPlotConfig,
    Field(discriminator="name"),
]


def build_detection_plotter(
    config: DetectionPlotConfig,
    targets: TargetProtocol,
) -> DetectionPlotter:
    return detection_plots.build(config, targets)
