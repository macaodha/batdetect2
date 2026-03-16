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
from batdetect2.evaluate.metrics.clip_classification import ClipEval
from batdetect2.evaluate.metrics.common import compute_precision_recall
from batdetect2.evaluate.plots.base import BasePlot, BasePlotConfig
from batdetect2.plotting.metrics import (
    plot_pr_curve,
    plot_pr_curves,
    plot_roc_curve,
    plot_roc_curves,
)
from batdetect2.typing import TargetProtocol

__all__ = [
    "ClipClassificationPlotConfig",
    "ClipClassificationPlotImportConfig",
    "ClipClassificationPlotter",
    "build_clip_classification_plotter",
]

ClipClassificationPlotter = Callable[
    [Sequence[ClipEval]], Iterable[Tuple[str, Figure]]
]

clip_classification_plots: Registry[
    ClipClassificationPlotter, [TargetProtocol]
] = Registry("clip_classification_plot")


@add_import_config(clip_classification_plots)
class ClipClassificationPlotImportConfig(ImportConfig):
    """Use any callable as a clip classification plot.

    Set ``name="import"`` and provide a ``target`` pointing to any
    callable to use it instead of a built-in option.
    """

    name: Literal["import"] = "import"


class PRCurveConfig(BasePlotConfig):
    name: Literal["pr_curve"] = "pr_curve"
    label: str = "pr_curve"
    title: str | None = "Clip Classification Precision-Recall Curve"
    separate_figures: bool = False


class PRCurve(BasePlot):
    def __init__(
        self,
        *args,
        separate_figures: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.separate_figures = separate_figures

    def __call__(
        self,
        clip_evaluations: Sequence[ClipEval],
    ) -> Iterable[Tuple[str, Figure]]:
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

    @clip_classification_plots.register(PRCurveConfig)
    @staticmethod
    def from_config(config: PRCurveConfig, targets: TargetProtocol):
        return PRCurve.build(
            config=config,
            targets=targets,
            separate_figures=config.separate_figures,
        )


class ROCCurveConfig(BasePlotConfig):
    name: Literal["roc_curve"] = "roc_curve"
    label: str = "roc_curve"
    title: str | None = "Clip Classification ROC Curve"
    separate_figures: bool = False


class ROCCurve(BasePlot):
    def __init__(
        self,
        *args,
        separate_figures: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.separate_figures = separate_figures

    def __call__(
        self,
        clip_evaluations: Sequence[ClipEval],
    ) -> Iterable[Tuple[str, Figure]]:
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

    @clip_classification_plots.register(ROCCurveConfig)
    @staticmethod
    def from_config(config: ROCCurveConfig, targets: TargetProtocol):
        return ROCCurve.build(
            config=config,
            targets=targets,
            separate_figures=config.separate_figures,
        )


ClipClassificationPlotConfig = Annotated[
    PRCurveConfig | ROCCurveConfig,
    Field(discriminator="name"),
]


def build_clip_classification_plotter(
    config: ClipClassificationPlotConfig,
    targets: TargetProtocol,
) -> ClipClassificationPlotter:
    return clip_classification_plots.build(config, targets)
