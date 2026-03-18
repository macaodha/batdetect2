from collections import defaultdict
from dataclasses import dataclass
from typing import (
    Annotated,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    Sequence,
    Tuple,
)

import numpy as np
from pydantic import Field
from sklearn import metrics
from soundevent import data

from batdetect2.core import (
    BaseConfig,
    ImportConfig,
    Registry,
    add_import_config,
)
from batdetect2.evaluate.metrics.common import (
    average_precision,
    compute_precision_recall,
)
from batdetect2.postprocess.types import Detection
from batdetect2.targets.types import TargetProtocol

__all__ = [
    "ClassificationMetric",
    "ClassificationMetricConfig",
    "ClassificationMetricImportConfig",
    "build_classification_metric",
    "compute_precision_recall_curves",
]


@dataclass
class MatchEval:
    clip: data.Clip
    gt: data.SoundEventAnnotation | None
    pred: Detection | None

    is_prediction: bool
    is_ground_truth: bool
    is_generic: bool
    true_class: str | None
    score: float


@dataclass
class ClipEval:
    clip: data.Clip
    matches: Mapping[str, List[MatchEval]]


ClassificationMetric = Callable[[Sequence[ClipEval]], Dict[str, float]]


classification_metrics: Registry[ClassificationMetric, [TargetProtocol]] = (
    Registry("classification_metric")
)


@add_import_config(classification_metrics)
class ClassificationMetricImportConfig(ImportConfig):
    """Use any callable as a classification metric.

    Set ``name="import"`` and provide a ``target`` pointing to any
    callable to use it instead of a built-in option.
    """

    name: Literal["import"] = "import"


class BaseClassificationConfig(BaseConfig):
    include: List[str] | None = None
    exclude: List[str] | None = None


class BaseClassificationMetric:
    def __init__(
        self,
        targets: TargetProtocol,
        include: List[str] | None = None,
        exclude: List[str] | None = None,
    ):
        self.targets = targets
        self.include = include
        self.exclude = exclude

    def include_class(self, class_name: str) -> bool:
        if self.include is not None:
            return class_name in self.include

        if self.exclude is not None:
            return class_name not in self.exclude

        return True


class ClassificationAveragePrecisionConfig(BaseClassificationConfig):
    name: Literal["average_precision"] = "average_precision"
    ignore_non_predictions: bool = True
    ignore_generic: bool = True
    label: str = "average_precision"


class ClassificationAveragePrecision(BaseClassificationMetric):
    def __init__(
        self,
        targets: TargetProtocol,
        ignore_non_predictions: bool = True,
        ignore_generic: bool = True,
        label: str = "average_precision",
        include: List[str] | None = None,
        exclude: List[str] | None = None,
    ):
        super().__init__(include=include, exclude=exclude, targets=targets)
        self.ignore_non_predictions = ignore_non_predictions
        self.ignore_generic = ignore_generic
        self.label = label

    def __call__(
        self, clip_evaluations: Sequence[ClipEval]
    ) -> Dict[str, float]:
        y_true, y_score, num_positives = _extract_per_class_metric_data(
            clip_evaluations,
            ignore_non_predictions=self.ignore_non_predictions,
            ignore_generic=self.ignore_generic,
        )

        class_scores = {
            class_name: average_precision(
                y_true[class_name],
                y_score[class_name],
                num_positives=num_positives[class_name],
            )
            for class_name in self.targets.class_names
        }

        mean_score = float(
            np.mean([v for v in class_scores.values() if not np.isnan(v)])
        )

        return {
            f"mean_{self.label}": mean_score,
            **{
                f"{self.label}/{class_name}": score
                for class_name, score in class_scores.items()
                if self.include_class(class_name)
            },
        }

    @classification_metrics.register(ClassificationAveragePrecisionConfig)
    @staticmethod
    def from_config(
        config: ClassificationAveragePrecisionConfig,
        targets: TargetProtocol,
    ):
        return ClassificationAveragePrecision(
            targets=targets,
            ignore_non_predictions=config.ignore_non_predictions,
            ignore_generic=config.ignore_generic,
            label=config.label,
            include=config.include,
            exclude=config.exclude,
        )


class ClassificationROCAUCConfig(BaseClassificationConfig):
    name: Literal["roc_auc"] = "roc_auc"
    label: str = "roc_auc"
    ignore_non_predictions: bool = True
    ignore_generic: bool = True


class ClassificationROCAUC(BaseClassificationMetric):
    def __init__(
        self,
        targets: TargetProtocol,
        ignore_non_predictions: bool = True,
        ignore_generic: bool = True,
        label: str = "roc_auc",
        include: List[str] | None = None,
        exclude: List[str] | None = None,
    ):
        self.targets = targets
        self.ignore_non_predictions = ignore_non_predictions
        self.ignore_generic = ignore_generic
        self.label = label
        self.include = include
        self.exclude = exclude

    def __call__(
        self, clip_evaluations: Sequence[ClipEval]
    ) -> Dict[str, float]:
        y_true, y_score, _ = _extract_per_class_metric_data(
            clip_evaluations,
            ignore_non_predictions=self.ignore_non_predictions,
            ignore_generic=self.ignore_generic,
        )

        class_scores = {
            class_name: float(
                metrics.roc_auc_score(
                    y_true[class_name],
                    y_score[class_name],
                )
            )
            for class_name in self.targets.class_names
        }

        mean_score = float(
            np.mean([v for v in class_scores.values() if v != np.nan])
        )

        return {
            f"mean_{self.label}": mean_score,
            **{
                f"{self.label}/{class_name}": score
                for class_name, score in class_scores.items()
                if self.include_class(class_name)
            },
        }

    @classification_metrics.register(ClassificationROCAUCConfig)
    @staticmethod
    def from_config(
        config: ClassificationROCAUCConfig, targets: TargetProtocol
    ):
        return ClassificationROCAUC(
            targets=targets,
            ignore_non_predictions=config.ignore_non_predictions,
            ignore_generic=config.ignore_generic,
            label=config.label,
        )


ClassificationMetricConfig = Annotated[
    ClassificationAveragePrecisionConfig | ClassificationROCAUCConfig,
    Field(discriminator="name"),
]


def build_classification_metric(
    config: ClassificationMetricConfig,
    targets: TargetProtocol,
) -> ClassificationMetric:
    return classification_metrics.build(config, targets)


def _extract_per_class_metric_data(
    clip_evaluations: Sequence[ClipEval],
    ignore_non_predictions: bool = True,
    ignore_generic: bool = True,
):
    y_true = defaultdict(list)
    y_score = defaultdict(list)
    num_positives = defaultdict(lambda: 0)

    for clip_eval in clip_evaluations:
        for class_name, matches in clip_eval.matches.items():
            for m in matches:
                # Exclude matches with ground truth sounds where the class
                # is unknown
                if m.is_generic and ignore_generic:
                    continue

                is_class = m.true_class == class_name

                if is_class:
                    num_positives[class_name] += 1

                # Ignore matches that don't correspond to a prediction
                if not m.is_prediction and ignore_non_predictions:
                    continue

                y_true[class_name].append(is_class)
                y_score[class_name].append(m.score)

    return y_true, y_score, num_positives


def compute_precision_recall_curves(
    clip_evaluations: Sequence[ClipEval],
    ignore_non_predictions: bool = True,
    ignore_generic: bool = True,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    y_true, y_score, num_positives = _extract_per_class_metric_data(
        clip_evaluations,
        ignore_non_predictions=ignore_non_predictions,
        ignore_generic=ignore_generic,
    )

    return {
        class_name: compute_precision_recall(
            y_true[class_name],
            y_score[class_name],
            num_positives=num_positives[class_name],
        )
        for class_name in y_true
    }
