from dataclasses import dataclass
from typing import Annotated, Callable, Dict, Literal, Sequence

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
    gt_det: bool
    score: float


ClipDetectionMetric = Callable[[Sequence[ClipEval]], Dict[str, float]]

clip_detection_metrics: Registry[ClipDetectionMetric, []] = Registry(
    "clip_detection_metric"
)


@add_import_config(clip_detection_metrics)
class ClipDetectionMetricImportConfig(ImportConfig):
    """Use any callable as a clip detection metric.

    Set ``name="import"`` and provide a ``target`` pointing to any
    callable to use it instead of a built-in option.
    """

    name: Literal["import"] = "import"


class ClipDetectionAveragePrecisionConfig(BaseConfig):
    name: Literal["average_precision"] = "average_precision"
    label: str = "average_precision"


class ClipDetectionAveragePrecision:
    def __init__(self, label: str = "average_precision"):
        self.label = label

    def __call__(
        self,
        clip_evaluations: Sequence[ClipEval],
    ) -> Dict[str, float]:
        y_true = []
        y_score = []

        for clip_eval in clip_evaluations:
            y_true.append(clip_eval.gt_det)
            y_score.append(clip_eval.score)

        score = average_precision(y_true=y_true, y_score=y_score)
        return {self.label: score}

    @clip_detection_metrics.register(ClipDetectionAveragePrecisionConfig)
    @staticmethod
    def from_config(config: ClipDetectionAveragePrecisionConfig):
        return ClipDetectionAveragePrecision(label=config.label)


class ClipDetectionROCAUCConfig(BaseConfig):
    name: Literal["roc_auc"] = "roc_auc"
    label: str = "roc_auc"


class ClipDetectionROCAUC:
    def __init__(self, label: str = "roc_auc"):
        self.label = label

    def __call__(
        self,
        clip_evaluations: Sequence[ClipEval],
    ) -> Dict[str, float]:
        y_true = []
        y_score = []

        for clip_eval in clip_evaluations:
            y_true.append(clip_eval.gt_det)
            y_score.append(clip_eval.score)

        score = float(metrics.roc_auc_score(y_true=y_true, y_score=y_score))
        return {self.label: score}

    @clip_detection_metrics.register(ClipDetectionROCAUCConfig)
    @staticmethod
    def from_config(config: ClipDetectionROCAUCConfig):
        return ClipDetectionROCAUC(label=config.label)


class ClipDetectionRecallConfig(BaseConfig):
    name: Literal["recall"] = "recall"
    threshold: float = 0.5
    label: str = "recall"


class ClipDetectionRecall:
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
            if clip_eval.gt_det:
                num_positives += 1

            if clip_eval.score >= self.threshold and clip_eval.gt_det:
                true_positives += 1

        if num_positives == 0:
            return {self.label: np.nan}

        score = true_positives / num_positives
        return {self.label: score}

    @clip_detection_metrics.register(ClipDetectionRecallConfig)
    @staticmethod
    def from_config(config: ClipDetectionRecallConfig):
        return ClipDetectionRecall(
            threshold=config.threshold, label=config.label
        )


class ClipDetectionPrecisionConfig(BaseConfig):
    name: Literal["precision"] = "precision"
    threshold: float = 0.5
    label: str = "precision"


class ClipDetectionPrecision:
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
            if clip_eval.score >= self.threshold:
                num_detections += 1

            if clip_eval.score >= self.threshold and clip_eval.gt_det:
                true_positives += 1

        if num_detections == 0:
            return {self.label: np.nan}

        score = true_positives / num_detections
        return {self.label: score}

    @clip_detection_metrics.register(ClipDetectionPrecisionConfig)
    @staticmethod
    def from_config(config: ClipDetectionPrecisionConfig):
        return ClipDetectionPrecision(
            threshold=config.threshold, label=config.label
        )


ClipDetectionMetricConfig = Annotated[
    ClipDetectionAveragePrecisionConfig
    | ClipDetectionROCAUCConfig
    | ClipDetectionRecallConfig
    | ClipDetectionPrecisionConfig,
    Field(discriminator="name"),
]


def build_clip_metric(config: ClipDetectionMetricConfig):
    return clip_detection_metrics.build(config)
