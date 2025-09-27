from typing import (
    Annotated,
    Callable,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from matplotlib.figure import Figure
from pydantic import Field
from sklearn import metrics

from batdetect2.core import Registry
from batdetect2.evaluate.metrics.clip_classification import ClipEval
from batdetect2.evaluate.metrics.common import compute_precision_recall
from batdetect2.evaluate.plots.base import BasePlot, BasePlotConfig
from batdetect2.plotting.metrics import (
    plot_pr_curves,
    plot_roc_curves,
)
from batdetect2.typing import TargetProtocol

__all__ = [
    "ClipClassificationPlotConfig",
    "ClipClassificationPlotter",
    "build_clip_classification_plotter",
]

ClipClassificationPlotter = Callable[[Sequence[ClipEval]], Tuple[str, Figure]]

clip_classification_plots: Registry[
    ClipClassificationPlotter, [TargetProtocol]
] = Registry("clip_classification_plot")


class PRCurveConfig(BasePlotConfig):
    name: Literal["pr_curve"] = "pr_curve"
    label: str = "pr_curve"
    title: Optional[str] = "Precision-Recall Curve"


class PRCurve(BasePlot):
    def __call__(
        self,
        clip_evaluations: Sequence[ClipEval],
    ) -> Tuple[str, Figure]:
        data = {}

        for class_name in self.targets.class_names:
            y_true = [class_name in c.true_classes for c in clip_evaluations]
            y_score = [
                c.class_scores.get(class_name, 0) for c in clip_evaluations
            ]

            precision, recall, thresholds = compute_precision_recall(
                y_true,
                y_score,
            )

            data[class_name] = (precision, recall, thresholds)

        fig = self.get_figure()
        ax = fig.subplots()
        plot_pr_curves(data, ax=ax)
        return self.label, fig

    @clip_classification_plots.register(PRCurveConfig)
    @staticmethod
    def from_config(config: PRCurveConfig, targets: TargetProtocol):
        return PRCurve.build(
            config=config,
            targets=targets,
        )


class ROCCurveConfig(BasePlotConfig):
    name: Literal["roc_curve"] = "roc_curve"
    label: str = "roc_curve"
    title: Optional[str] = "ROC Curve"


class ROCCurve(BasePlot):
    def __call__(
        self,
        clip_evaluations: Sequence[ClipEval],
    ) -> Tuple[str, Figure]:
        data = {}

        for class_name in self.targets.class_names:
            y_true = [class_name in c.true_classes for c in clip_evaluations]
            y_score = [
                c.class_scores.get(class_name, 0) for c in clip_evaluations
            ]

            fpr, tpr, thresholds = metrics.roc_curve(
                y_true,
                y_score,
            )

            data[class_name] = (fpr, tpr, thresholds)

        fig = self.get_figure()
        ax = fig.subplots()
        plot_roc_curves(data, ax=ax)
        return self.label, fig

    @clip_classification_plots.register(ROCCurveConfig)
    @staticmethod
    def from_config(config: ROCCurveConfig, targets: TargetProtocol):
        return ROCCurve.build(
            config=config,
            targets=targets,
        )


ClipClassificationPlotConfig = Annotated[
    Union[
        PRCurveConfig,
        ROCCurveConfig,
    ],
    Field(discriminator="name"),
]


def build_clip_classification_plotter(
    config: ClipClassificationPlotConfig,
    targets: TargetProtocol,
) -> ClipClassificationPlotter:
    return clip_classification_plots.build(config, targets)
