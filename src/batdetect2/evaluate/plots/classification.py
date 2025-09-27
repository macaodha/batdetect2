from typing import Annotated, Callable, Literal, Sequence, Tuple, Union

from matplotlib.figure import Figure
from pydantic import Field
from sklearn import metrics

from batdetect2.core import Registry
from batdetect2.evaluate.metrics.classification import (
    ClipEval,
    _extract_per_class_metric_data,
)
from batdetect2.evaluate.metrics.common import compute_precision_recall
from batdetect2.evaluate.plots.base import BasePlot, BasePlotConfig
from batdetect2.plotting.metrics import (
    plot_pr_curves,
    plot_roc_curves,
    plot_threshold_precision_curves,
    plot_threshold_recall_curves,
)
from batdetect2.typing import TargetProtocol

ClassificationPlotter = Callable[[Sequence[ClipEval]], Tuple[str, Figure]]

classification_plots: Registry[ClassificationPlotter, [TargetProtocol]] = (
    Registry("classification_plot")
)


class PRCurveConfig(BasePlotConfig):
    name: Literal["pr_curve"] = "pr_curve"
    label: str = "pr_curve"
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
    ) -> Tuple[str, Figure]:
        y_true, y_score, num_positives = _extract_per_class_metric_data(
            clip_evaluations,
            ignore_non_predictions=self.ignore_non_predictions,
            ignore_generic=self.ignore_generic,
        )

        fig = self.get_figure()
        ax = fig.subplots()

        data = {
            class_name: compute_precision_recall(
                y_true[class_name],
                y_score[class_name],
                num_positives=num_positives[class_name],
            )
            for class_name in self.targets.class_names
        }

        plot_pr_curves(data, ax=ax)

        return self.label, fig

    @classification_plots.register(PRCurveConfig)
    @staticmethod
    def from_config(config: PRCurveConfig, targets: TargetProtocol):
        return PRCurve.build(
            config=config,
            targets=targets,
            ignore_non_predictions=config.ignore_non_predictions,
            ignore_generic=config.ignore_generic,
        )


class ThresholdPRCurveConfig(BasePlotConfig):
    name: Literal["threshold_pr_curve"] = "threshold_pr_curve"
    label: str = "threshold_pr_curve"
    figsize: tuple[int, int] = (10, 5)
    ignore_non_predictions: bool = True
    ignore_generic: bool = True


class ThresholdPRCurve(BasePlot):
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
    ) -> Tuple[str, Figure]:
        y_true, y_score, num_positives = _extract_per_class_metric_data(
            clip_evaluations,
            ignore_non_predictions=self.ignore_non_predictions,
            ignore_generic=self.ignore_generic,
        )

        data = {
            class_name: compute_precision_recall(
                y_true[class_name],
                y_score[class_name],
                num_positives[class_name],
            )
            for class_name in self.targets.class_names
        }

        fig = self.get_figure()
        ax1, ax2 = fig.subplots(nrows=1, ncols=2, sharey=True)

        plot_threshold_precision_curves(data, ax=ax1, add_legend=False)
        plot_threshold_recall_curves(data, ax=ax2, add_legend=True)

        return self.label, fig

    @classification_plots.register(ThresholdPRCurveConfig)
    @staticmethod
    def from_config(config: ThresholdPRCurveConfig, targets: TargetProtocol):
        return ThresholdPRCurve.build(
            config=config,
            targets=targets,
            ignore_non_predictions=config.ignore_non_predictions,
            ignore_generic=config.ignore_generic,
        )


class ROCCurveConfig(BasePlotConfig):
    name: Literal["roc_curve"] = "roc_curve"
    label: str = "roc_curve"
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
    ) -> Tuple[str, Figure]:
        y_true, y_score, _ = _extract_per_class_metric_data(
            clip_evaluations,
            ignore_non_predictions=self.ignore_non_predictions,
            ignore_generic=self.ignore_generic,
        )

        data = {
            class_name: metrics.roc_curve(
                y_true[class_name],
                y_score[class_name],
            )
            for class_name in self.targets.class_names
        }

        fig = self.get_figure()
        ax = fig.subplots()

        plot_roc_curves(data, ax=ax)

        return self.label, fig

    @classification_plots.register(ROCCurveConfig)
    @staticmethod
    def from_config(config: ROCCurveConfig, targets: TargetProtocol):
        return ROCCurve.build(
            config=config,
            targets=targets,
            ignore_non_predictions=config.ignore_non_predictions,
            ignore_generic=config.ignore_generic,
        )


ClassificationPlotConfig = Annotated[
    Union[
        PRCurveConfig,
        ROCCurveConfig,
        ThresholdPRCurveConfig,
    ],
    Field(discriminator="name"),
]


def build_classification_plotter(
    config: ClassificationPlotConfig,
    targets: TargetProtocol,
) -> ClassificationPlotter:
    return classification_plots.build(config, targets)
