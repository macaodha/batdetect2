from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Literal, Sequence, Set

from pydantic import Field, field_validator
from sklearn import metrics
from soundevent import data

from batdetect2.evaluate.evaluator.base import (
    BaseEvaluator,
    BaseEvaluatorConfig,
    evaluators,
)
from batdetect2.evaluate.metrics.common import average_precision
from batdetect2.typing.postprocess import RawPrediction
from batdetect2.typing.targets import TargetProtocol


@dataclass
class ClipInfo:
    gt_det: bool
    gt_classes: Set[str]
    pred_score: float
    pred_class_scores: Dict[str, float]


ClipMetric = Callable[[Sequence[ClipInfo]], float]


def clip_detection_average_precision(
    clip_evaluations: Sequence[ClipInfo],
) -> float:
    y_true = []
    y_score = []

    for clip_eval in clip_evaluations:
        y_true.append(clip_eval.gt_det)
        y_score.append(clip_eval.pred_score)

    return average_precision(y_true=y_true, y_score=y_score)


def clip_detection_roc_auc(
    clip_evaluations: Sequence[ClipInfo],
) -> float:
    y_true = []
    y_score = []

    for clip_eval in clip_evaluations:
        y_true.append(clip_eval.gt_det)
        y_score.append(clip_eval.pred_score)

    return float(metrics.roc_auc_score(y_true=y_true, y_score=y_score))


clip_metrics = {
    "average_precision": clip_detection_average_precision,
    "roc_auc": clip_detection_roc_auc,
}


class ClipMetricsConfig(BaseEvaluatorConfig):
    name: Literal["clip"] = "clip"
    prefix: str = "clip"
    metrics: List[str] = Field(
        default_factory=lambda: [
            "average_precision",
            "roc_auc",
        ]
    )

    @field_validator("metrics", mode="after")
    @classmethod
    def validate_metrics(cls, v: List[str]) -> List[str]:
        for metric_name in v:
            if metric_name not in clip_metrics:
                raise ValueError(f"Unknown metric {metric_name}")
        return v


class ClipEvaluator(BaseEvaluator):
    def __init__(self, *args, metrics: Dict[str, ClipMetric], **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = metrics

    def evaluate(
        self,
        clip_annotations: Sequence[data.ClipAnnotation],
        predictions: Sequence[Sequence[RawPrediction]],
    ) -> List[ClipInfo]:
        return [
            self.match_clip(clip_annotation, preds)
            for clip_annotation, preds in zip(clip_annotations, predictions)
        ]

    def compute_metrics(
        self,
        eval_outputs: List[ClipInfo],
    ) -> Dict[str, float]:
        scores = {
            name: metric(eval_outputs) for name, metric in self.metrics.items()
        }
        return {
            f"{self.prefix}/{name}": score for name, score in scores.items()
        }

    def match_clip(
        self,
        clip_annotation: data.ClipAnnotation,
        predictions: Sequence[RawPrediction],
    ) -> ClipInfo:
        clip = clip_annotation.clip

        gt_det = False
        gt_classes = set()
        for sound_event in clip_annotation.sound_events:
            if self.filter_sound_event_annotations(sound_event, clip):
                continue

            gt_det = True
            class_name = self.targets.encode_class(sound_event)

            if class_name is None:
                continue

            gt_classes.add(class_name)

        pred_score = 0
        pred_class_scores: defaultdict[str, float] = defaultdict(lambda: 0)
        for pred in predictions:
            if self.filter_predictions(pred, clip):
                continue

            pred_score = max(pred_score, pred.detection_score)

            for class_name, class_score in zip(
                self.targets.class_names,
                pred.class_scores,
            ):
                pred_class_scores[class_name] = max(
                    pred_class_scores[class_name],
                    class_score,
                )

        return ClipInfo(
            gt_det=gt_det,
            gt_classes=gt_classes,
            pred_score=pred_score,
            pred_class_scores=pred_class_scores,
        )

    @evaluators.register(ClipMetricsConfig)
    @staticmethod
    def from_config(
        config: ClipMetricsConfig,
        targets: TargetProtocol,
    ):
        metrics = {name: clip_metrics.get(name) for name in config.metrics}
        return ClipEvaluator.build(
            config=config,
            metrics=metrics,
            targets=targets,
        )
