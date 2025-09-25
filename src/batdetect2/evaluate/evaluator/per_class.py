from collections import defaultdict
from typing import (
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
)

import numpy as np
from pydantic import Field
from soundevent import data

from batdetect2.evaluate.evaluator.base import (
    BaseEvaluator,
    BaseEvaluatorConfig,
    evaluators,
)
from batdetect2.evaluate.match import match
from batdetect2.evaluate.metrics.per_class_matches import (
    ClassificationAveragePrecisionConfig,
    PerClassMatchMetric,
    PerClassMatchMetricConfig,
    build_per_class_matches_metric,
)
from batdetect2.typing import (
    ClipMatches,
    RawPrediction,
    TargetProtocol,
)

ScoreFn = Callable[[RawPrediction, int], float]


def score_by_class_score(pred: RawPrediction, class_index: int) -> float:
    return float(pred.class_scores[class_index])


def score_by_adjusted_class_score(
    pred: RawPrediction,
    class_index: int,
) -> float:
    return float(pred.class_scores[class_index]) * pred.detection_score


ScoreFunctionOption = Literal["class_score", "adjusted_class_score"]
score_functions: Mapping[ScoreFunctionOption, ScoreFn] = {
    "class_score": score_by_class_score,
    "adjusted_class_score": score_by_adjusted_class_score,
}


def get_score_fn(name: ScoreFunctionOption) -> ScoreFn:
    return score_functions[name]


class ClassificationMetricsConfig(BaseEvaluatorConfig):
    name: Literal["classification"] = "classification"
    prefix: str = "classification"
    include_generics: bool = True
    score_by: ScoreFunctionOption = "class_score"
    metrics: List[PerClassMatchMetricConfig] = Field(
        default_factory=lambda: [ClassificationAveragePrecisionConfig()]
    )
    include: Optional[List[str]] = None
    exclude: Optional[List[str]] = None


class PerClassEvaluator(BaseEvaluator):
    def __init__(
        self,
        *args,
        metrics: Dict[str, PerClassMatchMetric],
        score_fn: ScoreFn,
        include_generics: bool = True,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.score_fn = score_fn
        self.metrics = metrics

        self.include_generics = include_generics

        self.include = include
        self.exclude = exclude

        self.selected = self.targets.class_names
        if include is not None:
            self.selected = [
                class_name
                for class_name in self.selected
                if class_name in include
            ]

        if exclude is not None:
            self.selected = [
                class_name
                for class_name in self.selected
                if class_name not in exclude
            ]

    def evaluate(
        self,
        clip_annotations: Sequence[data.ClipAnnotation],
        predictions: Sequence[Sequence[RawPrediction]],
    ) -> Dict[str, List[ClipMatches]]:
        ret = defaultdict(list)

        for clip_annotation, preds in zip(clip_annotations, predictions):
            matches = self.match_clip(clip_annotation, preds)
            for class_name, clip_eval in matches.items():
                ret[class_name].append(clip_eval)

        return ret

    def compute_metrics(
        self,
        eval_outputs: Dict[str, List[ClipMatches]],
    ) -> Dict[str, float]:
        results = {}

        for metric_name, metric in self.metrics.items():
            class_scores = {
                class_name: metric(eval_outputs[class_name], class_name)
                for class_name in self.targets.class_names
            }
            mean = float(
                np.mean([v for v in class_scores.values() if v != np.nan])
            )

            results[f"{self.prefix}/mean_{metric_name}"] = mean

            for class_name, value in class_scores.items():
                if class_name not in self.selected:
                    continue

                results[f"{self.prefix}/{metric_name}/{class_name}"] = value

        return results

    def match_clip(
        self,
        clip_annotation: data.ClipAnnotation,
        predictions: Sequence[RawPrediction],
    ) -> Dict[str, ClipMatches]:
        clip = clip_annotation.clip

        preds = [
            pred for pred in predictions if self.filter_predictions(pred, clip)
        ]

        all_gts = [
            sound_event
            for sound_event in clip_annotation.sound_events
            if self.filter_sound_event_annotations(sound_event, clip)
        ]

        ret = {}

        for class_name in self.targets.class_names:
            class_idx = self.targets.class_names.index(class_name)

            # Only match to targets of the given class
            gts = [
                sound_event
                for sound_event in all_gts
                if self.is_class(sound_event, class_name)
            ]
            scores = [self.score_fn(pred, class_idx) for pred in preds]

            ret[class_name] = match(
                gts,
                preds,
                clip=clip,
                scores=scores,
                targets=self.targets,
                matcher=self.matcher,
            )

        return ret

    def is_class(
        self,
        sound_event: data.SoundEventAnnotation,
        class_name: str,
    ) -> bool:
        sound_event_class = self.targets.encode_class(sound_event)

        if sound_event_class is None and self.include_generics:
            # Sound events that are generic could be of the given
            # class
            return True

        return sound_event_class == class_name

    @evaluators.register(ClassificationMetricsConfig)
    @staticmethod
    def from_config(
        config: ClassificationMetricsConfig,
        targets: TargetProtocol,
    ):
        metrics = {
            metric.name: build_per_class_matches_metric(metric)
            for metric in config.metrics
        }
        return PerClassEvaluator.build(
            config=config,
            targets=targets,
            metrics=metrics,
            score_fn=get_score_fn(config.score_by),
            include_generics=config.include_generics,
            include=config.include,
            exclude=config.exclude,
        )
