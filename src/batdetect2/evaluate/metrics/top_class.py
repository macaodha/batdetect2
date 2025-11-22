from collections import defaultdict
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
from sklearn import metrics, preprocessing
from soundevent import data

from batdetect2.core import BaseConfig, Registry
from batdetect2.evaluate.metrics.common import average_precision
from batdetect2.typing import RawPrediction
from batdetect2.typing.targets import TargetProtocol

__all__ = [
    "TopClassMetricConfig",
    "TopClassMetric",
    "build_top_class_metric",
]


@dataclass
class MatchEval:
    clip: data.Clip
    gt: Optional[data.SoundEventAnnotation]
    pred: Optional[RawPrediction]

    is_ground_truth: bool
    is_generic: bool
    is_prediction: bool
    pred_class: Optional[str]
    true_class: Optional[str]
    score: float


@dataclass
class ClipEval:
    clip: data.Clip
    matches: List[MatchEval]


TopClassMetric = Callable[[Sequence[ClipEval]], Dict[str, float]]


top_class_metrics: Registry[TopClassMetric, []] = Registry("top_class_metric")


class TopClassAveragePrecisionConfig(BaseConfig):
    name: Literal["average_precision"] = "average_precision"
    label: str = "average_precision"
    ignore_generic: bool = True
    ignore_non_predictions: bool = True


class TopClassAveragePrecision:
    def __init__(
        self,
        ignore_generic: bool = True,
        ignore_non_predictions: bool = True,
        label: str = "average_precision",
    ):
        self.ignore_generic = ignore_generic
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
                if m.is_generic and self.ignore_generic:
                    # Ignore gt sounds with unknown class
                    continue

                num_positives += int(m.is_ground_truth)

                if not m.is_prediction and self.ignore_non_predictions:
                    # Ignore non predictions
                    continue

                y_true.append(m.pred_class == m.true_class)
                y_score.append(m.score)

        score = average_precision(y_true, y_score, num_positives=num_positives)
        return {self.label: score}

    @top_class_metrics.register(TopClassAveragePrecisionConfig)
    @staticmethod
    def from_config(config: TopClassAveragePrecisionConfig):
        return TopClassAveragePrecision(
            ignore_generic=config.ignore_generic,
            label=config.label,
        )


class TopClassROCAUCConfig(BaseConfig):
    name: Literal["roc_auc"] = "roc_auc"
    ignore_generic: bool = True
    ignore_non_predictions: bool = True
    label: str = "roc_auc"


class TopClassROCAUC:
    def __init__(
        self,
        ignore_generic: bool = True,
        ignore_non_predictions: bool = True,
        label: str = "roc_auc",
    ):
        self.ignore_generic = ignore_generic
        self.ignore_non_predictions = ignore_non_predictions
        self.label = label

    def __call__(self, clip_evals: Sequence[ClipEval]) -> Dict[str, float]:
        y_true: List[bool] = []
        y_score: List[float] = []

        for clip_eval in clip_evals:
            for m in clip_eval.matches:
                if m.is_generic and self.ignore_generic:
                    # Ignore gt sounds with unknown class
                    continue

                if not m.is_prediction and self.ignore_non_predictions:
                    # Ignore non predictions
                    continue

                y_true.append(m.pred_class == m.true_class)
                y_score.append(m.score)

        score = float(metrics.roc_auc_score(y_true, y_score))
        return {self.label: score}

    @top_class_metrics.register(TopClassROCAUCConfig)
    @staticmethod
    def from_config(config: TopClassROCAUCConfig):
        return TopClassROCAUC(
            ignore_generic=config.ignore_generic,
            label=config.label,
        )


class TopClassRecallConfig(BaseConfig):
    name: Literal["recall"] = "recall"
    threshold: float = 0.5
    label: str = "recall"


class TopClassRecall:
    def __init__(self, threshold: float, label: str = "recall"):
        self.threshold = threshold
        self.label = label

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

                if m.score >= self.threshold and m.pred_class == m.true_class:
                    true_positives += 1

        if num_positives == 0:
            return {self.label: np.nan}

        score = true_positives / num_positives
        return {self.label: score}

    @top_class_metrics.register(TopClassRecallConfig)
    @staticmethod
    def from_config(config: TopClassRecallConfig):
        return TopClassRecall(
            threshold=config.threshold,
            label=config.label,
        )


class TopClassPrecisionConfig(BaseConfig):
    name: Literal["precision"] = "precision"
    threshold: float = 0.5
    label: str = "precision"


class TopClassPrecision:
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

                if is_detection and m.pred_class == m.true_class:
                    true_positives += 1

        if num_detections == 0:
            return {self.label: np.nan}

        score = true_positives / num_detections
        return {self.label: score}

    @top_class_metrics.register(TopClassPrecisionConfig)
    @staticmethod
    def from_config(config: TopClassPrecisionConfig):
        return TopClassPrecision(
            threshold=config.threshold,
            label=config.label,
        )


class BalancedAccuracyConfig(BaseConfig):
    name: Literal["balanced_accuracy"] = "balanced_accuracy"
    label: str = "balanced_accuracy"
    exclude_noise: bool = False
    noise_class: str = "noise"


class BalancedAccuracy:
    def __init__(
        self,
        exclude_noise: bool = True,
        noise_class: str = "noise",
        label: str = "balanced_accuracy",
    ):
        self.exclude_noise = exclude_noise
        self.noise_class = noise_class
        self.label = label

    def __call__(
        self,
        clip_evaluations: Sequence[ClipEval],
    ) -> Dict[str, float]:
        y_true: List[str] = []
        y_pred: List[str] = []

        for clip_eval in clip_evaluations:
            for m in clip_eval.matches:
                if m.is_generic:
                    # Ignore matches that correspond to a sound event
                    # with unknown class
                    continue

                if not m.is_ground_truth and self.exclude_noise:
                    # Ignore predictions that were not matched to a
                    # ground truth
                    continue

                if m.pred_class is None and self.exclude_noise:
                    # Ignore non-predictions
                    continue

                y_true.append(m.true_class or self.noise_class)
                y_pred.append(m.pred_class or self.noise_class)

        encoder = preprocessing.LabelEncoder()
        encoder.fit(list(set(y_true) | set(y_pred)))

        y_true = encoder.transform(y_true)
        y_pred = encoder.transform(y_pred)
        score = metrics.balanced_accuracy_score(y_true, y_pred)
        return {self.label: score}

    @top_class_metrics.register(BalancedAccuracyConfig)
    @staticmethod
    def from_config(config: BalancedAccuracyConfig):
        return BalancedAccuracy(
            exclude_noise=config.exclude_noise,
            noise_class=config.noise_class,
            label=config.label,
        )


TopClassMetricConfig = Annotated[
    Union[
        TopClassAveragePrecisionConfig,
        TopClassROCAUCConfig,
        TopClassRecallConfig,
        TopClassPrecisionConfig,
        BalancedAccuracyConfig,
    ],
    Field(discriminator="name"),
]


def build_top_class_metric(config: TopClassMetricConfig):
    return top_class_metrics.build(config)


def compute_confusion_matrix(
    clip_evaluations: Sequence[ClipEval],
    targets: TargetProtocol,
    threshold: float = 0.2,
    normalize: Literal["true", "pred", "all", "none"] = "true",
    exclude_generic: bool = True,
    exclude_false_positives: bool = True,
    exclude_false_negatives: bool = True,
    noise_class: str = "noise",
):
    y_true: List[str] = []
    y_pred: List[str] = []

    for clip_eval in clip_evaluations:
        for m in clip_eval.matches:
            true_class = m.true_class
            pred_class = m.pred_class

            if not m.is_prediction and exclude_false_negatives:
                # Ignore matches that don't correspond to a prediction
                continue

            if not m.is_ground_truth and exclude_false_positives:
                # Ignore matches that don't correspond to a ground truth
                continue

            if m.score < threshold:
                if exclude_false_negatives:
                    continue

                pred_class = noise_class

            if m.is_generic:
                if exclude_generic:
                    # Ignore gt sounds with unknown class
                    continue

                true_class = targets.detection_class_name

            y_true.append(true_class or noise_class)
            y_pred.append(pred_class or noise_class)

    labels = sorted(targets.class_names)

    if not exclude_generic:
        labels.append(targets.detection_class_name)

    if not exclude_false_positives or not exclude_false_negatives:
        labels.append(noise_class)

    return metrics.confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
        normalize=normalize,
    ), labels
