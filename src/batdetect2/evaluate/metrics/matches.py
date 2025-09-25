from typing import Annotated, Callable, Literal, Sequence, Union

import numpy as np
from pydantic import Field
from sklearn import metrics

from batdetect2.core import BaseConfig, Registry
from batdetect2.evaluate.metrics.common import average_precision
from batdetect2.typing import (
    ClipMatches,
)

__all__ = [
    "MatchMetricConfig",
    "MatchesMetric",
    "build_match_metric",
]

MatchesMetric = Callable[[Sequence[ClipMatches]], float]


metrics_registry: Registry[MatchesMetric, []] = Registry("match_metric")


class DetectionAveragePrecisionConfig(BaseConfig):
    name: Literal["detection_average_precision"] = (
        "detection_average_precision"
    )
    ignore_non_predictions: bool = True


class DetectionAveragePrecision:
    def __init__(self, ignore_non_predictions: bool = True):
        self.ignore_non_predictions = ignore_non_predictions

    def __call__(
        self,
        clip_evaluations: Sequence[ClipMatches],
    ) -> float:
        y_true = []
        y_score = []
        num_positives = 0

        for clip_eval in clip_evaluations:
            for m in clip_eval.matches:
                num_positives += int(m.gt_det)

                # Ignore matches that don't correspond to a prediction
                if not m.is_prediction and self.ignore_non_predictions:
                    continue

                y_true.append(m.gt_det)
                y_score.append(m.pred_score)

        return average_precision(y_true, y_score, num_positives=num_positives)

    @metrics_registry.register(DetectionAveragePrecisionConfig)
    @staticmethod
    def from_config(config: DetectionAveragePrecisionConfig):
        return DetectionAveragePrecision(
            ignore_non_predictions=config.ignore_non_predictions
        )


class TopClassAveragePrecisionConfig(BaseConfig):
    name: Literal["top_class_average_precision"] = (
        "top_class_average_precision"
    )
    ignore_non_predictions: bool = True
    ignore_generic: bool = True


class TopClassAveragePrecision:
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
    ) -> float:
        y_true = []
        y_score = []
        num_positives = 0

        for clip_eval in clip_evaluations:
            for m in clip_eval.matches:
                if m.is_generic and self.ignore_generic:
                    # Ignore ground truth sounds with unknown class
                    continue

                num_positives += int(m.gt_det)

                if not m.is_prediction and self.ignore_non_predictions:
                    # Ignore matches that don't correspond to a prediction
                    continue

                y_true.append(m.gt_det & (m.top_class == m.gt_class))
                y_score.append(m.top_class_score)

        return average_precision(y_true, y_score, num_positives=num_positives)

    @metrics_registry.register(TopClassAveragePrecisionConfig)
    @staticmethod
    def from_config(config: TopClassAveragePrecisionConfig):
        return TopClassAveragePrecision(
            ignore_non_predictions=config.ignore_non_predictions
        )


class DetectionROCAUCConfig(BaseConfig):
    name: Literal["detection_roc_auc"] = "detection_roc_auc"
    ignore_non_predictions: bool = True


class DetectionROCAUC:
    def __init__(
        self,
        ignore_non_predictions: bool = True,
    ):
        self.ignore_non_predictions = ignore_non_predictions

    def __call__(self, clip_evaluations: Sequence[ClipMatches]) -> float:
        y_true = []
        y_score = []

        for clip_eval in clip_evaluations:
            for m in clip_eval.matches:
                if not m.is_prediction and self.ignore_non_predictions:
                    # Ignore matches that don't correspond to a prediction
                    continue

                y_true.append(m.gt_det)
                y_score.append(m.pred_score)

        return float(metrics.roc_auc_score(y_true, y_score))

    @metrics_registry.register(DetectionROCAUCConfig)
    @staticmethod
    def from_config(config: DetectionROCAUCConfig):
        return DetectionROCAUC(
            ignore_non_predictions=config.ignore_non_predictions
        )


class DetectionRecallConfig(BaseConfig):
    name: Literal["detection_recall"] = "detection_recall"
    threshold: float = 0.5


class DetectionRecall:
    def __init__(self, threshold: float):
        self.threshold = threshold

    def __call__(
        self,
        clip_evaluations: Sequence[ClipMatches],
    ) -> float:
        num_positives = 0
        true_positives = 0

        for clip_eval in clip_evaluations:
            for m in clip_eval.matches:
                if m.gt_det:
                    num_positives += 1

                if m.pred_score >= self.threshold and m.gt_det:
                    true_positives += 1

        if num_positives == 0:
            return 1

        return true_positives / num_positives

    @metrics_registry.register(DetectionRecallConfig)
    @staticmethod
    def from_config(config: DetectionRecallConfig):
        return DetectionRecall(threshold=config.threshold)


class DetectionPrecisionConfig(BaseConfig):
    name: Literal["detection_precision"] = "detection_precision"
    threshold: float = 0.5


class DetectionPrecision:
    def __init__(self, threshold: float):
        self.threshold = threshold

    def __call__(
        self,
        clip_evaluations: Sequence[ClipMatches],
    ) -> float:
        num_detections = 0
        true_positives = 0

        for clip_eval in clip_evaluations:
            for m in clip_eval.matches:
                is_detection = m.pred_score >= self.threshold

                if is_detection:
                    num_detections += 1

                if is_detection and m.gt_det:
                    true_positives += 1

        if num_detections == 0:
            return np.nan

        return true_positives / num_detections

    @metrics_registry.register(DetectionPrecisionConfig)
    @staticmethod
    def from_config(config: DetectionPrecisionConfig):
        return DetectionPrecision(threshold=config.threshold)


MatchMetricConfig = Annotated[
    Union[
        DetectionAveragePrecisionConfig,
        DetectionROCAUCConfig,
        DetectionRecallConfig,
        DetectionPrecisionConfig,
        TopClassAveragePrecisionConfig,
    ],
    Field(discriminator="name"),
]


def build_match_metric(config: MatchMetricConfig):
    return metrics_registry.build(config)
