from typing import (
    Annotated,
    Callable,
    Iterable,
    Literal,
    Sequence,
    Tuple,
)

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pydantic import Field
from sklearn import metrics

from batdetect2.core import ImportConfig, Registry, add_import_config
from batdetect2.evaluate.metrics.classification import (
    ClipEval,
    _extract_per_class_metric_data,
    compute_precision_recall_curves,
)
from batdetect2.evaluate.plots.base import BasePlot, BasePlotConfig
from batdetect2.plotting.metrics import (
    plot_pr_curve,
    plot_pr_curves,
    plot_roc_curve,
    plot_roc_curves,
    plot_threshold_precision_curve,
    plot_threshold_precision_curves,
    plot_threshold_recall_curve,
    plot_threshold_recall_curves,
)
from batdetect2.targets.types import TargetProtocol

ClassificationPlotter = Callable[
    [Sequence[ClipEval]], Iterable[Tuple[str, Figure]]
]

classification_plots: Registry[ClassificationPlotter, [TargetProtocol]] = (
    Registry("classification_plot")
)


@add_import_config(classification_plots)
class ClassificationPlotImportConfig(ImportConfig):
    """Use any callable as a classification plot.

    Set ``name="import"`` and provide a ``target`` pointing to any
    callable to use it instead of a built-in option.
    """

    name: Literal["import"] = "import"


class PRCurveConfig(BasePlotConfig):
    name: Literal["pr_curve"] = "pr_curve"
    label: str = "pr_curve"
    title: str | None = "Classification Precision-Recall Curve"
    ignore_non_predictions: bool = True
    ignore_generic: bool = True
    separate_figures: bool = False


class PRCurve(BasePlot):
    def __init__(
        self,
        *args,
        ignore_non_predictions: bool = True,
        ignore_generic: bool = True,
        separate_figures: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.ignore_non_predictions = ignore_non_predictions
        self.ignore_generic = ignore_generic
        self.separate_figures = separate_figures

    def __call__(
        self,
        clip_evaluations: Sequence[ClipEval],
    ) -> Iterable[Tuple[str, Figure]]:
        data = compute_precision_recall_curves(
            clip_evaluations,
            ignore_non_predictions=self.ignore_non_predictions,
            ignore_generic=self.ignore_generic,
        )

        if not self.separate_figures:
            fig = self.create_figure()
            ax = fig.subplots()
            plot_pr_curves(data, ax=ax)
            yield self.label, fig
            return

        for class_name, (precision, recall, thresholds) in data.items():
            fig = self.create_figure()
            ax = fig.subplots()

            ax = plot_pr_curve(precision, recall, thresholds, ax=ax)
            ax.set_title(class_name)
            yield f"{self.label}/{class_name}", fig
            plt.close(fig)

    @classification_plots.register(PRCurveConfig)
    @staticmethod
    def from_config(config: PRCurveConfig, targets: TargetProtocol):
        return PRCurve.build(
            config=config,
            targets=targets,
            ignore_non_predictions=config.ignore_non_predictions,
            ignore_generic=config.ignore_generic,
            separate_figures=config.separate_figures,
        )


class ThresholdPrecisionCurveConfig(BasePlotConfig):
    name: Literal["threshold_precision_curve"] = "threshold_precision_curve"
    label: str = "threshold_precision_curve"
    title: str | None = "Classification Threshold-Precision Curve"
    ignore_non_predictions: bool = True
    ignore_generic: bool = True
    separate_figures: bool = False


class ThresholdPrecisionCurve(BasePlot):
    def __init__(
        self,
        *args,
        ignore_non_predictions: bool = True,
        ignore_generic: bool = True,
        separate_figures: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.ignore_non_predictions = ignore_non_predictions
        self.ignore_generic = ignore_generic
        self.separate_figures = separate_figures

    def __call__(
        self,
        clip_evaluations: Sequence[ClipEval],
    ) -> Iterable[Tuple[str, Figure]]:
        data = compute_precision_recall_curves(
            clip_evaluations,
            ignore_non_predictions=self.ignore_non_predictions,
            ignore_generic=self.ignore_generic,
        )

        if not self.separate_figures:
            fig = self.create_figure()
            ax = fig.subplots()

            plot_threshold_precision_curves(data, ax=ax)

            yield self.label, fig

            return

        for class_name, (precision, _, thresholds) in data.items():
            fig = self.create_figure()
            ax = fig.subplots()

            ax = plot_threshold_precision_curve(
                thresholds,
                precision,
                ax=ax,
            )

            ax.set_title(class_name)

            yield f"{self.label}/{class_name}", fig

            plt.close(fig)

    @classification_plots.register(ThresholdPrecisionCurveConfig)
    @staticmethod
    def from_config(
        config: ThresholdPrecisionCurveConfig, targets: TargetProtocol
    ):
        return ThresholdPrecisionCurve.build(
            config=config,
            targets=targets,
            ignore_non_predictions=config.ignore_non_predictions,
            ignore_generic=config.ignore_generic,
            separate_figures=config.separate_figures,
        )


class ThresholdRecallCurveConfig(BasePlotConfig):
    name: Literal["threshold_recall_curve"] = "threshold_recall_curve"
    label: str = "threshold_recall_curve"
    title: str | None = "Classification Threshold-Recall Curve"
    ignore_non_predictions: bool = True
    ignore_generic: bool = True
    separate_figures: bool = False


class ThresholdRecallCurve(BasePlot):
    def __init__(
        self,
        *args,
        ignore_non_predictions: bool = True,
        ignore_generic: bool = True,
        separate_figures: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.ignore_non_predictions = ignore_non_predictions
        self.ignore_generic = ignore_generic
        self.separate_figures = separate_figures

    def __call__(
        self,
        clip_evaluations: Sequence[ClipEval],
    ) -> Iterable[Tuple[str, Figure]]:
        data = compute_precision_recall_curves(
            clip_evaluations,
            ignore_non_predictions=self.ignore_non_predictions,
            ignore_generic=self.ignore_generic,
        )

        if not self.separate_figures:
            fig = self.create_figure()
            ax = fig.subplots()

            plot_threshold_recall_curves(data, ax=ax, add_legend=True)

            yield self.label, fig

            return

        for class_name, (_, recall, thresholds) in data.items():
            fig = self.create_figure()
            ax = fig.subplots()

            ax = plot_threshold_recall_curve(
                thresholds,
                recall,
                ax=ax,
            )

            ax.set_title(class_name)

            yield f"{self.label}/{class_name}", fig

            plt.close(fig)

    @classification_plots.register(ThresholdRecallCurveConfig)
    @staticmethod
    def from_config(
        config: ThresholdRecallCurveConfig, targets: TargetProtocol
    ):
        return ThresholdRecallCurve.build(
            config=config,
            targets=targets,
            ignore_non_predictions=config.ignore_non_predictions,
            ignore_generic=config.ignore_generic,
            separate_figures=config.separate_figures,
        )


class ROCCurveConfig(BasePlotConfig):
    name: Literal["roc_curve"] = "roc_curve"
    label: str = "roc_curve"
    title: str | None = "Classification ROC Curve"
    ignore_non_predictions: bool = True
    ignore_generic: bool = True
    separate_figures: bool = False


class ROCCurve(BasePlot):
    def __init__(
        self,
        *args,
        ignore_non_predictions: bool = True,
        ignore_generic: bool = True,
        separate_figures: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.ignore_non_predictions = ignore_non_predictions
        self.ignore_generic = ignore_generic
        self.separate_figures = separate_figures

    def __call__(
        self,
        clip_evaluations: Sequence[ClipEval],
    ) -> Iterable[Tuple[str, Figure]]:
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

        if not self.separate_figures:
            fig = self.create_figure()
            ax = fig.subplots()

            plot_roc_curves(data, ax=ax)

            yield self.label, fig

            return

        for class_name, (fpr, tpr, thresholds) in data.items():
            fig = self.create_figure()
            ax = fig.subplots()

            ax = plot_roc_curve(fpr, tpr, thresholds, ax=ax)
            ax.set_title(class_name)

            yield f"{self.label}/{class_name}", fig

            plt.close(fig)

    @classification_plots.register(ROCCurveConfig)
    @staticmethod
    def from_config(config: ROCCurveConfig, targets: TargetProtocol):
        return ROCCurve.build(
            config=config,
            targets=targets,
            ignore_non_predictions=config.ignore_non_predictions,
            ignore_generic=config.ignore_generic,
            separate_figures=config.separate_figures,
        )


ClassificationPlotConfig = Annotated[
    PRCurveConfig
    | ROCCurveConfig
    | ThresholdPrecisionCurveConfig
    | ThresholdRecallCurveConfig,
    Field(discriminator="name"),
]


def build_classification_plotter(
    config: ClassificationPlotConfig,
    targets: TargetProtocol,
) -> ClassificationPlotter:
    return classification_plots.build(config, targets)
