from dataclasses import dataclass
from typing import (
    Annotated,
    Callable,
    Dict,
    List,
    Literal,
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

__all__ = [
    "DetectionMetricConfig",
    "DetectionMetric",
    "build_detection_metric",
]


@dataclass
class MatchEval:
    gt: data.SoundEventAnnotation | None
    pred: RawPrediction | None

    is_prediction: bool
    is_ground_truth: bool
    score: float


@dataclass
class ClipEval:
    clip: data.Clip
    matches: List[MatchEval]


DetectionMetric = Callable[[Sequence[ClipEval]], Dict[str, float]]


detection_metrics: Registry[DetectionMetric, []] = Registry("detection_metric")


class DetectionAveragePrecisionConfig(BaseConfig):
    name: Literal["average_precision"] = "average_precision"
    label: str = "average_precision"
    ignore_non_predictions: bool = True


class DetectionAveragePrecision:
    def __init__(self, label: str, ignore_non_predictions: bool = True):
        self.ignore_non_predictions = ignore_non_predictions
        self.label = label

    def __call__(
        self,
        clip_evals: Sequence[ClipEval],
    ) -> Dict[str, float]:
        y_true = []
        y_score = []
        num_positives = 0

        for clip_eval in clip_evals:
            for m in clip_eval.matches:
                num_positives += int(m.is_ground_truth)

                # Ignore matches that don't correspond to a prediction
                if not m.is_prediction and self.ignore_non_predictions:
                    continue

                y_true.append(m.is_ground_truth)
                y_score.append(m.score)

        ap = average_precision(y_true, y_score, num_positives=num_positives)
        return {self.label: ap}

    @detection_metrics.register(DetectionAveragePrecisionConfig)
    @staticmethod
    def from_config(config: DetectionAveragePrecisionConfig):
        return DetectionAveragePrecision(
            label=config.label,
            ignore_non_predictions=config.ignore_non_predictions,
        )


class DetectionROCAUCConfig(BaseConfig):
    name: Literal["roc_auc"] = "roc_auc"
    label: str = "roc_auc"
    ignore_non_predictions: bool = True


class DetectionROCAUC:
    def __init__(
        self,
        label: str = "roc_auc",
        ignore_non_predictions: bool = True,
    ):
        self.label = label
        self.ignore_non_predictions = ignore_non_predictions

    def __call__(self, clip_evals: Sequence[ClipEval]) -> Dict[str, float]:
        y_true: List[bool] = []
        y_score: List[float] = []

        for clip_eval in clip_evals:
            for m in clip_eval.matches:
                if not m.is_prediction and self.ignore_non_predictions:
                    # Ignore matches that don't correspond to a prediction
                    continue

                y_true.append(m.is_ground_truth)
                y_score.append(m.score)

        score = float(metrics.roc_auc_score(y_true, y_score))
        return {self.label: score}

    @detection_metrics.register(DetectionROCAUCConfig)
    @staticmethod
    def from_config(config: DetectionROCAUCConfig):
        return DetectionROCAUC(
            label=config.label,
            ignore_non_predictions=config.ignore_non_predictions,
        )


class DetectionRecallConfig(BaseConfig):
    name: Literal["recall"] = "recall"
    label: str = "recall"
    threshold: float = 0.5


class DetectionRecall:
    def __init__(self, threshold: float, label: str = "recall"):
        self.label = label
        self.threshold = threshold

    def __call__(
        self,
        clip_evaluations: Sequence[ClipEval],
    ) -> Dict[str, float]:
        num_positives = 0
        true_positives = 0

        for clip_eval in clip_evaluations:
            for m in clip_eval.matches:
                if m.is_ground_truth:
                    num_positives += 1

                if m.score >= self.threshold and m.is_ground_truth:
                    true_positives += 1

        if num_positives == 0:
            return {self.label: np.nan}

        score = true_positives / num_positives
        return {self.label: score}

    @detection_metrics.register(DetectionRecallConfig)
    @staticmethod
    def from_config(config: DetectionRecallConfig):
        return DetectionRecall(threshold=config.threshold, label=config.label)


class DetectionPrecisionConfig(BaseConfig):
    name: Literal["precision"] = "precision"
    label: str = "precision"
    threshold: float = 0.5


class DetectionPrecision:
    def __init__(self, threshold: float, label: str = "precision"):
        self.threshold = threshold
        self.label = label

    def __call__(
        self,
        clip_evaluations: Sequence[ClipEval],
    ) -> Dict[str, float]:
        num_detections = 0
        true_positives = 0

        for clip_eval in clip_evaluations:
            for m in clip_eval.matches:
                is_detection = m.score >= self.threshold

                if is_detection:
                    num_detections += 1

                if is_detection and m.is_ground_truth:
                    true_positives += 1

        if num_detections == 0:
            return {self.label: np.nan}

        score = true_positives / num_detections
        return {self.label: score}

    @detection_metrics.register(DetectionPrecisionConfig)
    @staticmethod
    def from_config(config: DetectionPrecisionConfig):
        return DetectionPrecision(
            threshold=config.threshold,
            label=config.label,
        )


DetectionMetricConfig = Annotated[
    DetectionAveragePrecisionConfig | DetectionROCAUCConfig | DetectionRecallConfig | DetectionPrecisionConfig,
    Field(discriminator="name"),
]


def build_detection_metric(config: DetectionMetricConfig):
    return detection_metrics.build(config)
