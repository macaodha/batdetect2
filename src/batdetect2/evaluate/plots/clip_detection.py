from typing import (
    Annotated,
    Callable,
    Iterable,
    Literal,
    Sequence,
    Tuple,
)

import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from pydantic import Field
from sklearn import metrics

from batdetect2.core import Registry
from batdetect2.evaluate.metrics.clip_detection import ClipEval
from batdetect2.evaluate.metrics.common import compute_precision_recall
from batdetect2.evaluate.plots.base import BasePlot, BasePlotConfig
from batdetect2.plotting.metrics import plot_pr_curve, plot_roc_curve
from batdetect2.typing import TargetProtocol

__all__ = [
    "ClipDetectionPlotConfig",
    "ClipDetectionPlotter",
    "build_clip_detection_plotter",
]

ClipDetectionPlotter = Callable[
    [Sequence[ClipEval]], Iterable[Tuple[str, Figure]]
]


clip_detection_plots: Registry[ClipDetectionPlotter, [TargetProtocol]] = (
    Registry("clip_detection_plot")
)


class PRCurveConfig(BasePlotConfig):
    name: Literal["pr_curve"] = "pr_curve"
    label: str = "pr_curve"
    title: str | None = "Clip Detection Precision-Recall Curve"


class PRCurve(BasePlot):
    def __call__(
        self,
        clip_evaluations: Sequence[ClipEval],
    ) -> Iterable[Tuple[str, Figure]]:
        y_true = [c.gt_det for c in clip_evaluations]
        y_score = [c.score for c in clip_evaluations]

        precision, recall, thresholds = compute_precision_recall(
            y_true,
            y_score,
        )

        fig = self.create_figure()
        ax = fig.subplots()
        plot_pr_curve(precision, recall, thresholds, ax=ax)
        yield self.label, fig

    @clip_detection_plots.register(PRCurveConfig)
    @staticmethod
    def from_config(config: PRCurveConfig, targets: TargetProtocol):
        return PRCurve.build(
            config=config,
            targets=targets,
        )


class ROCCurveConfig(BasePlotConfig):
    name: Literal["roc_curve"] = "roc_curve"
    label: str = "roc_curve"
    title: str | None = "Clip Detection ROC Curve"


class ROCCurve(BasePlot):
    def __call__(
        self,
        clip_evaluations: Sequence[ClipEval],
    ) -> Iterable[Tuple[str, Figure]]:
        y_true = [c.gt_det for c in clip_evaluations]
        y_score = [c.score for c in clip_evaluations]

        fpr, tpr, thresholds = metrics.roc_curve(
            y_true,
            y_score,
        )

        fig = self.create_figure()
        ax = fig.subplots()
        plot_roc_curve(fpr, tpr, thresholds, ax=ax)
        yield self.label, fig

    @clip_detection_plots.register(ROCCurveConfig)
    @staticmethod
    def from_config(config: ROCCurveConfig, targets: TargetProtocol):
        return ROCCurve.build(
            config=config,
            targets=targets,
        )


class ScoreDistributionPlotConfig(BasePlotConfig):
    name: Literal["score_distribution"] = "score_distribution"
    label: str = "score_distribution"
    title: str | None = "Clip Detection Score Distribution"


class ScoreDistributionPlot(BasePlot):
    def __call__(
        self,
        clip_evaluations: Sequence[ClipEval],
    ) -> Iterable[Tuple[str, Figure]]:
        y_true = [c.gt_det for c in clip_evaluations]
        y_score = [c.score for c in clip_evaluations]

        fig = self.create_figure()
        ax = fig.subplots()

        df = pd.DataFrame({"is_true": y_true, "score": y_score})
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

    @clip_detection_plots.register(ScoreDistributionPlotConfig)
    @staticmethod
    def from_config(
        config: ScoreDistributionPlotConfig, targets: TargetProtocol
    ):
        return ScoreDistributionPlot.build(
            config=config,
            targets=targets,
        )


ClipDetectionPlotConfig = Annotated[
    PRCurveConfig | ROCCurveConfig | ScoreDistributionPlotConfig,
    Field(discriminator="name"),
]


def build_clip_detection_plotter(
    config: ClipDetectionPlotConfig,
    targets: TargetProtocol,
) -> ClipDetectionPlotter:
    return clip_detection_plots.build(config, targets)
