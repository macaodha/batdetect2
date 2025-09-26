from collections import defaultdict
from dataclasses import dataclass
from typing import (
    Annotated,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Union,
)

import numpy as np
from pydantic import Field
from sklearn import metrics
from soundevent import data

from batdetect2.core import BaseConfig, Registry
from batdetect2.evaluate.metrics.common import average_precision
from batdetect2.typing import RawPrediction

__all__ = []


@dataclass
class MatchEval:
    gt: Optional[data.SoundEventAnnotation]
    pred: Optional[RawPrediction]

    is_prediction: bool
    is_ground_truth: bool
    is_generic: bool
    true_class: Optional[str]
    score: float


@dataclass
class ClipEval:
    clip: data.Clip
    matches: Mapping[str, List[MatchEval]]


ClassificationMetric = Callable[[Sequence[ClipEval]], Dict[str, float]]


classification_metrics: Registry[ClassificationMetric, []] = Registry(
    "classification_metric"
)


class BaseClassificationConfig(BaseConfig):
    include: Optional[List[str]] = None
    exclude: Optional[List[str]] = None


class BaseClassificationMetric:
    def __init__(
        self,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
    ):
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
        ignore_non_predictions: bool = True,
        ignore_generic: bool = True,
        label: str = "average_precision",
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
    ):
        super().__init__(include=include, exclude=exclude)
        self.ignore_non_predictions = ignore_non_predictions
        self.ignore_generic = ignore_generic
        self.label = label

    def __call__(
        self, clip_evaluations: Sequence[ClipEval]
    ) -> Dict[str, float]:
        y_true = defaultdict(list)
        y_score = defaultdict(list)
        num_positives = defaultdict(lambda: 0)

        class_names = set()

        for clip_eval in clip_evaluations:
            for class_name, matches in clip_eval.matches.items():
                class_names.add(class_name)

                for m in matches:
                    # Exclude matches with ground truth sounds where the class
                    # is unknown
                    if m.is_generic and self.ignore_generic:
                        continue

                    is_class = m.true_class == class_name

                    if is_class:
                        num_positives[class_name] += 1

                    # Ignore matches that don't correspond to a prediction
                    if not m.is_prediction and self.ignore_non_predictions:
                        continue

                    y_true[class_name].append(is_class)
                    y_score[class_name].append(m.score)

        class_scores = {
            class_name: average_precision(
                y_true[class_name],
                y_score[class_name],
                num_positives=num_positives[class_name],
            )
            for class_name in class_names
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

    @classification_metrics.register(ClassificationAveragePrecisionConfig)
    @staticmethod
    def from_config(config: ClassificationAveragePrecisionConfig):
        return ClassificationAveragePrecision(
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
        ignore_non_predictions: bool = True,
        ignore_generic: bool = True,
        label: str = "roc_auc",
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
    ):
        self.ignore_non_predictions = ignore_non_predictions
        self.ignore_generic = ignore_generic
        self.label = label
        self.include = include
        self.exclude = exclude

    def __call__(
        self, clip_evaluations: Sequence[ClipEval]
    ) -> Dict[str, float]:
        y_true = defaultdict(list)
        y_score = defaultdict(list)

        class_names = set()

        for clip_eval in clip_evaluations:
            for class_name, matches in clip_eval.matches.items():
                class_names.add(class_name)

                for m in matches:
                    # Exclude matches with ground truth sounds where the class
                    # is unknown
                    if m.is_generic and self.ignore_generic:
                        continue

                    # Ignore matches that don't correspond to a prediction
                    if not m.is_prediction and self.ignore_non_predictions:
                        continue

                    y_true[class_name].append(m.true_class == class_name)
                    y_score[class_name].append(m.score)

        class_scores = {
            class_name: float(
                metrics.roc_auc_score(
                    y_true[class_name],
                    y_score[class_name],
                )
            )
            for class_name in class_names
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
    def from_config(config: ClassificationROCAUCConfig):
        return ClassificationROCAUC(
            ignore_non_predictions=config.ignore_non_predictions,
            ignore_generic=config.ignore_generic,
            label=config.label,
        )


ClassificationMetricConfig = Annotated[
    Union[
        ClassificationAveragePrecisionConfig,
        ClassificationROCAUCConfig,
    ],
    Field(discriminator="name"),
]


def build_classification_metrics(config: ClassificationMetricConfig):
    return classification_metrics.build(config)
