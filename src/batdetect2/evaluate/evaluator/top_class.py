from typing import Dict, List, Literal, Sequence

from pydantic import Field, field_validator
from soundevent import data

from batdetect2.evaluate.match import match
from batdetect2.evaluate.metrics.base import (
    BaseMetric,
    BaseMetricConfig,
    metrics_registry,
)
from batdetect2.evaluate.metrics.common import average_precision
from batdetect2.evaluate.metrics.detection import DetectionMetric
from batdetect2.typing import ClipMatches, RawPrediction, TargetProtocol

__all__ = [
    "TopClassEvaluator",
    "TopClassEvaluatorConfig",
]


def top_class_average_precision(
    clip_evaluations: Sequence[ClipMatches],
) -> float:
    y_true = []
    y_score = []
    num_positives = 0

    for clip_eval in clip_evaluations:
        for m in clip_eval.matches:
            is_generic = m.gt_det and (m.gt_class is None)

            # Ignore ground truth sounds with unknown class
            if is_generic:
                continue

            num_positives += int(m.gt_det)

            # Ignore matches that don't correspond to a prediction
            if m.pred_geometry is None:
                continue

            y_true.append(m.gt_det & (m.top_class == m.gt_class))
            y_score.append(m.top_class_score)

    return average_precision(y_true, y_score, num_positives=num_positives)


top_class_metrics = {
    "average_precision": top_class_average_precision,
}


class TopClassEvaluatorConfig(BaseMetricConfig):
    name: Literal["top_class"] = "top_class"
    prefix: str = "top_class"
    metrics: List[str] = Field(default_factory=lambda: ["average_precision"])

    @field_validator("metrics", mode="after")
    @classmethod
    def validate_metrics(cls, v: List[str]) -> List[str]:
        for metric_name in v:
            if metric_name not in top_class_metrics:
                raise ValueError(f"Unknown metric {metric_name}")
        return v


class TopClassEvaluator(BaseMetric):
    def __init__(self, *args, metrics: Dict[str, DetectionMetric], **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = metrics

    def __call__(
        self,
        clip_annotations: Sequence[data.ClipAnnotation],
        predictions: Sequence[Sequence[RawPrediction]],
    ) -> Dict[str, float]:
        clip_evaluations = [
            self.match_clip(clip_annotation, preds)
            for clip_annotation, preds in zip(clip_annotations, predictions)
        ]
        scores = {
            name: metric(clip_evaluations)
            for name, metric in self.metrics.items()
        }
        return {
            f"{self.prefix}/{name}": score for name, score in scores.items()
        }

    def match_clip(
        self,
        clip_annotation: data.ClipAnnotation,
        predictions: Sequence[RawPrediction],
    ) -> ClipMatches:
        clip = clip_annotation.clip

        gts = [
            sound_event
            for sound_event in clip_annotation.sound_events
            if self.filter_sound_event_annotations(sound_event, clip)
        ]
        preds = [
            pred for pred in predictions if self.filter_predictions(pred, clip)
        ]
        # Use score of top class for matching
        scores = [pred.class_scores.max() for pred in preds]

        return match(
            gts,
            preds,
            scores=scores,
            clip=clip,
            targets=self.targets,
            matcher=self.matcher,
        )

    @classmethod
    def from_config(
        cls,
        config: TopClassEvaluatorConfig,
        targets: TargetProtocol,
    ):
        metrics = {
            name: top_class_metrics.get(name) for name in config.metrics
        }
        return super().build(
            config=config,
            metrics=metrics,
            targets=targets,
        )


metrics_registry.register(TopClassEvaluatorConfig, TopClassEvaluator)
