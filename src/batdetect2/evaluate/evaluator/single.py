from typing import Callable, Dict, List, Literal, Mapping, Sequence

from pydantic import Field
from soundevent import data

from batdetect2.evaluate.evaluator.base import (
    BaseEvaluator,
    BaseEvaluatorConfig,
    evaluators,
)
from batdetect2.evaluate.match import match
from batdetect2.evaluate.metrics.matches import (
    DetectionAveragePrecisionConfig,
    MatchesMetric,
    MatchMetricConfig,
    build_match_metric,
)
from batdetect2.typing import ClipMatches, RawPrediction, TargetProtocol

ScoreFn = Callable[[RawPrediction], float]


def score_by_detection_score(pred: RawPrediction) -> float:
    return pred.detection_score


def score_by_top_class_score(pred: RawPrediction) -> float:
    return pred.class_scores.max()


ScoreFunctionOption = Literal["detection_score", "top_class_score"]
score_functions: Mapping[ScoreFunctionOption, ScoreFn] = {
    "detection_score": score_by_detection_score,
    "top_class_score": score_by_top_class_score,
}


def get_score_fn(name: ScoreFunctionOption) -> ScoreFn:
    return score_functions[name]


class GlobalEvaluatorConfig(BaseEvaluatorConfig):
    name: Literal["detection"] = "detection"
    prefix: str = "detection"
    score_by: ScoreFunctionOption = "detection_score"
    metrics: List[MatchMetricConfig] = Field(
        default_factory=lambda: [DetectionAveragePrecisionConfig()]
    )


class GlobalEvaluator(BaseEvaluator):
    def __init__(
        self,
        *args,
        score_fn: ScoreFn,
        metrics: Dict[str, MatchesMetric],
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.metrics = metrics
        self.score_fn = score_fn

    def compute_metrics(
        self,
        eval_outputs: List[ClipMatches],
    ) -> Dict[str, float]:
        scores = {
            name: metric(eval_outputs) for name, metric in self.metrics.items()
        }
        return {
            f"{self.prefix}/{name}": score for name, score in scores.items()
        }

    def evaluate(
        self,
        clip_annotations: Sequence[data.ClipAnnotation],
        predictions: Sequence[Sequence[RawPrediction]],
    ) -> List[ClipMatches]:
        return [
            self.match_clip(clip_annotation, preds)
            for clip_annotation, preds in zip(clip_annotations, predictions)
        ]

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
        scores = [self.score_fn(pred) for pred in preds]

        return match(
            gts,
            preds,
            scores=scores,
            clip=clip,
            targets=self.targets,
            matcher=self.matcher,
        )

    @evaluators.register(GlobalEvaluatorConfig)
    @staticmethod
    def from_config(
        config: GlobalEvaluatorConfig,
        targets: TargetProtocol,
    ):
        metrics = {
            metric.name: build_match_metric(metric)
            for metric in config.metrics
        }
        score_fn = get_score_fn(config.score_by)
        return GlobalEvaluator.build(
            config=config,
            score_fn=score_fn,
            metrics=metrics,
            targets=targets,
        )
