from collections import defaultdict
from dataclasses import dataclass
from typing import Annotated, Callable, Dict, Literal, Sequence, Set

import numpy as np
from pydantic import Field
from sklearn import metrics

from batdetect2.core.configs import BaseConfig
from batdetect2.core.registries import (
    ImportConfig,
    Registry,
    add_import_config,
)
from batdetect2.evaluate.metrics.common import average_precision


@dataclass
class ClipEval:
    true_classes: Set[str]
    class_scores: Dict[str, float]


ClipClassificationMetric = Callable[[Sequence[ClipEval]], Dict[str, float]]

clip_classification_metrics: Registry[ClipClassificationMetric, []] = Registry(
    "clip_classification_metric"
)


@add_import_config(clip_classification_metrics)
class ClipClassificationMetricImportConfig(ImportConfig):
    """Use any callable as a clip classification metric.

    Set ``name="import"`` and provide a ``target`` pointing to any
    callable to use it instead of a built-in option.
    """

    name: Literal["import"] = "import"


class ClipClassificationAveragePrecisionConfig(BaseConfig):
    name: Literal["average_precision"] = "average_precision"
    label: str = "average_precision"


class ClipClassificationAveragePrecision:
    def __init__(self, label: str = "average_precision"):
        self.label = label

    def __call__(
        self,
        clip_evaluations: Sequence[ClipEval],
    ) -> Dict[str, float]:
        y_true = defaultdict(list)
        y_score = defaultdict(list)

        for clip_eval in clip_evaluations:
            for class_name, score in clip_eval.class_scores.items():
                y_true[class_name].append(class_name in clip_eval.true_classes)
                y_score[class_name].append(score)

        class_scores = {
            class_name: float(
                average_precision(
                    y_true=y_true[class_name],
                    y_score=y_score[class_name],
                )
            )
            for class_name in y_true
        }

        mean = np.mean([v for v in class_scores.values() if not np.isnan(v)])

        return {
            f"mean_{self.label}": float(mean),
            **{
                f"{self.label}/{class_name}": score
                for class_name, score in class_scores.items()
                if not np.isnan(score)
            },
        }

    @clip_classification_metrics.register(
        ClipClassificationAveragePrecisionConfig
    )
    @staticmethod
    def from_config(config: ClipClassificationAveragePrecisionConfig):
        return ClipClassificationAveragePrecision(label=config.label)


class ClipClassificationROCAUCConfig(BaseConfig):
    name: Literal["roc_auc"] = "roc_auc"
    label: str = "roc_auc"


class ClipClassificationROCAUC:
    def __init__(self, label: str = "roc_auc"):
        self.label = label

    def __call__(
        self,
        clip_evaluations: Sequence[ClipEval],
    ) -> Dict[str, float]:
        y_true = defaultdict(list)
        y_score = defaultdict(list)

        for clip_eval in clip_evaluations:
            for class_name, score in clip_eval.class_scores.items():
                y_true[class_name].append(class_name in clip_eval.true_classes)
                y_score[class_name].append(score)

        class_scores = {
            class_name: float(
                metrics.roc_auc_score(
                    y_true=y_true[class_name],
                    y_score=y_score[class_name],
                )
            )
            for class_name in y_true
        }

        mean = np.mean([v for v in class_scores.values() if not np.isnan(v)])

        return {
            f"mean_{self.label}": float(mean),
            **{
                f"{self.label}/{class_name}": score
                for class_name, score in class_scores.items()
                if not np.isnan(score)
            },
        }

    @clip_classification_metrics.register(ClipClassificationROCAUCConfig)
    @staticmethod
    def from_config(config: ClipClassificationROCAUCConfig):
        return ClipClassificationROCAUC(label=config.label)


ClipClassificationMetricConfig = Annotated[
    ClipClassificationAveragePrecisionConfig | ClipClassificationROCAUCConfig,
    Field(discriminator="name"),
]


def build_clip_metric(config: ClipClassificationMetricConfig):
    return clip_classification_metrics.build(config)
