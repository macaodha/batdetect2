from typing import Annotated, Callable, Literal, Sequence, Union

from pydantic import Field
from sklearn import metrics

from batdetect2.core import BaseConfig, Registry
from batdetect2.evaluate.metrics.common import average_precision
from batdetect2.typing import (
    ClipMatches,
)

__all__ = []

PerClassMatchMetric = Callable[[Sequence[ClipMatches], str], float]


metrics_registry: Registry[PerClassMatchMetric, []] = Registry(
    "match_metric"
)


class ClassificationAveragePrecisionConfig(BaseConfig):
    name: Literal["classification_average_precision"] = (
        "classification_average_precision"
    )
    ignore_non_predictions: bool = True
    ignore_generic: bool = True


class ClassificationAveragePrecision:
    def __init__(
        self,
        ignore_non_predictions: bool = True,
        ignore_generic: bool = True,
    ):
        self.ignore_non_predictions = ignore_non_predictions
        self.ignore_generic = ignore_generic

    def __call__(
        self,
        clip_evaluations: Sequence[ClipMatches],
        class_name: str,
    ) -> float:
        y_true = []
        y_score = []
        num_positives = 0

        for clip_eval in clip_evaluations:
            for m in clip_eval.matches:
                is_class = m.gt_class == class_name

                if is_class:
                    num_positives += 1

                # Ignore matches that don't correspond to a prediction
                if not m.is_prediction and self.ignore_non_predictions:
                    continue

                # Exclude matches with ground truth sounds where the class is
                # unknown
                if m.is_generic and self.ignore_generic:
                    continue

                y_true.append(is_class)
                y_score.append(m.pred_class_scores.get(class_name, 0))

        return average_precision(y_true, y_score, num_positives=num_positives)

    @metrics_registry.register(ClassificationAveragePrecisionConfig)
    @staticmethod
    def from_config(config: ClassificationAveragePrecisionConfig):
        return ClassificationAveragePrecision(
            ignore_non_predictions=config.ignore_non_predictions,
            ignore_generic=config.ignore_generic,
        )


class ClassificationROCAUCConfig(BaseConfig):
    name: Literal["classification_roc_auc"] = "classification_roc_auc"
    ignore_non_predictions: bool = True
    ignore_generic: bool = True


class ClassificationROCAUC:
    def __init__(
        self,
        ignore_non_predictions: bool = True,
        ignore_generic: bool = True,
    ):
        self.ignore_non_predictions = ignore_non_predictions
        self.ignore_generic = ignore_generic

    def __call__(
        self,
        clip_evaluations: Sequence[ClipMatches],
        class_name: str,
    ) -> float:
        y_true = []
        y_score = []

        for clip_eval in clip_evaluations:
            for m in clip_eval.matches:
                # Exclude matches with ground truth sounds where the class is
                # unknown
                if m.is_generic and self.ignore_generic:
                    continue

                # Ignore matches that don't correspond to a prediction
                if not m.is_prediction and self.ignore_non_predictions:
                    continue

                y_true.append(m.gt_class == class_name)
                y_score.append(m.pred_class_scores.get(class_name, 0))

        return float(metrics.roc_auc_score(y_true, y_score))

    @metrics_registry.register(ClassificationROCAUCConfig)
    @staticmethod
    def from_config(config: ClassificationROCAUCConfig):
        return ClassificationROCAUC(
            ignore_non_predictions=config.ignore_non_predictions,
            ignore_generic=config.ignore_generic,
        )


PerClassMatchMetricConfig = Annotated[
    Union[
        ClassificationAveragePrecisionConfig,
        ClassificationROCAUCConfig,
    ],
    Field(discriminator="name"),
]


def build_per_class_matches_metric(config: PerClassMatchMetricConfig):
    return metrics_registry.build(config)
