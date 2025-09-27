from typing import Annotated, Callable, List, Literal, Sequence, Tuple, Union

from matplotlib.figure import Figure
from pydantic import Field
from sklearn import metrics

from batdetect2.core import Registry
from batdetect2.evaluate.metrics.common import compute_precision_recall
from batdetect2.evaluate.metrics.top_class import ClipEval
from batdetect2.evaluate.plots.base import BasePlot, BasePlotConfig
from batdetect2.plotting.metrics import plot_pr_curve, plot_roc_curve
from batdetect2.typing import TargetProtocol

TopClassPlotter = Callable[[Sequence[ClipEval]], Tuple[str, Figure]]

top_class_plots: Registry[TopClassPlotter, [TargetProtocol]] = Registry(
    name="top_class_plot"
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

        fig = self.get_figure()
        ax = fig.subplots()
        plot_pr_curve(precision, recall, thresholds, ax=ax)
        return self.label, fig

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

        fig = self.get_figure()
        ax = fig.subplots()
        plot_roc_curve(fpr, tpr, thresholds, ax=ax)
        return self.label, fig

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
        self.exclude_noise = exclude_noise
        self.noise_class = noise_class
        self.normalize = normalize
        self.add_colorbar = add_colorbar
        self.threshold = threshold
        self.cmap = cmap

    def __call__(
        self,
        clip_evaluations: Sequence[ClipEval],
    ) -> Tuple[str, Figure]:
        y_true: List[str] = []
        y_pred: List[str] = []

        for clip_eval in clip_evaluations:
            for m in clip_eval.matches:
                true_class = m.true_class
                pred_class = m.pred_class

                if not m.is_prediction and self.exclude_noise:
                    # Ignore matches that don't correspond to a prediction
                    continue

                if not m.is_ground_truth and self.exclude_noise:
                    # Ignore matches that don't correspond to a ground truth
                    continue

                if m.score < self.threshold:
                    if self.exclude_noise:
                        continue

                    pred_class = self.noise_class

                if m.is_generic:
                    if self.exclude_generic:
                        # Ignore gt sounds with unknown class
                        continue

                    true_class = self.targets.detection_class_name

                y_true.append(true_class or self.noise_class)
                y_pred.append(pred_class or self.noise_class)

        fig = self.get_figure()
        ax = fig.subplots()

        class_names = [*self.targets.class_names]

        if not self.exclude_generic:
            class_names.append(self.targets.detection_class_name)

        if not self.exclude_noise:
            class_names.append(self.noise_class)

        metrics.ConfusionMatrixDisplay.from_predictions(
            y_true,
            y_pred,
            labels=class_names,
            ax=ax,
            xticks_rotation="vertical",
            cmap=self.cmap,
            colorbar=self.add_colorbar,
            normalize=self.normalize if self.normalize != "none" else None,
            values_format=".2f",
        )

        return self.label, fig

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


TopClassPlotConfig = Annotated[
    Union[
        PRCurveConfig,
        ROCCurveConfig,
        ConfusionMatrixConfig,
    ],
    Field(discriminator="name"),
]


def build_top_class_plotter(
    config: TopClassPlotConfig,
    targets: TargetProtocol,
) -> TopClassPlotter:
    return top_class_plots.build(config, targets)
