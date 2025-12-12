from functools import partial
from typing import Literal

from pydantic import Field
from soundevent import data
from soundevent.evaluation import match_detections_and_gts

from batdetect2.evaluate.metrics.classification import (
    ClassificationAveragePrecisionConfig,
    ClassificationMetricConfig,
    ClipEval,
    MatchEval,
    build_classification_metric,
)
from batdetect2.evaluate.plots.classification import (
    ClassificationPlotConfig,
    build_classification_plotter,
)
from batdetect2.evaluate.tasks.base import (
    BaseSEDTask,
    BaseSEDTaskConfig,
    tasks_registry,
)
from batdetect2.typing import (
    BatDetect2Prediction,
    RawPrediction,
    TargetProtocol,
)


class ClassificationTaskConfig(BaseSEDTaskConfig):
    name: Literal["sound_event_classification"] = "sound_event_classification"
    prefix: str = "classification"
    metrics: list[ClassificationMetricConfig] = Field(
        default_factory=lambda: [ClassificationAveragePrecisionConfig()]
    )
    plots: list[ClassificationPlotConfig] = Field(default_factory=list)
    include_generics: bool = True


class ClassificationTask(BaseSEDTask[ClipEval]):
    def __init__(
        self,
        *args,
        include_generics: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.include_generics = include_generics

    def evaluate_clip(
        self,
        clip_annotation: data.ClipAnnotation,
        prediction: BatDetect2Prediction,
    ) -> ClipEval:
        clip = clip_annotation.clip

        preds = [
            pred
            for pred in prediction.predictions
            if self.include_prediction(pred, clip)
        ]

        all_gts = [
            sound_event
            for sound_event in clip_annotation.sound_events
            if self.include_sound_event_annotation(sound_event, clip)
        ]

        per_class_matches = {}

        for class_name in self.targets.class_names:
            class_idx = self.targets.class_names.index(class_name)

            # Only match to targets of the given class
            gts = [
                sound_event
                for sound_event in all_gts
                if is_target_class(
                    sound_event,
                    class_name,
                    self.targets,
                    include_generics=self.include_generics,
                )
            ]

            matches = []

            for match in match_detections_and_gts(
                detections=preds,
                ground_truths=gts,
                affinity=self.affinity,
                score=partial(get_class_score, class_idx=class_idx),
                strict_match=self.strict_match,
                affinity_threshold=self.affinity_threshold,
            ):
                true_class = (
                    self.targets.encode_class(match.annotation)
                    if match.annotation is not None
                    else None
                )
                matches.append(
                    MatchEval(
                        clip=clip,
                        gt=match.annotation,
                        pred=match.prediction,
                        is_prediction=match.prediction is not None,
                        is_ground_truth=match.annotation is not None,
                        is_generic=match.annotation is not None
                        and true_class is None,
                        true_class=true_class,
                        score=match.prediction_score,
                    )
                )

            per_class_matches[class_name] = matches

        return ClipEval(clip=clip, matches=per_class_matches)

    @tasks_registry.register(ClassificationTaskConfig)
    @staticmethod
    def from_config(
        config: ClassificationTaskConfig,
        targets: TargetProtocol,
    ):
        metrics = [
            build_classification_metric(metric, targets)
            for metric in config.metrics
        ]
        plots = [
            build_classification_plotter(plot, targets)
            for plot in config.plots
        ]
        return ClassificationTask.build(
            config=config,
            plots=plots,
            targets=targets,
            metrics=metrics,
            include_generics=config.include_generics,
        )


def get_class_score(pred: RawPrediction, class_idx: int) -> float:
    return pred.class_scores[class_idx]


def is_target_class(
    sound_event: data.SoundEventAnnotation,
    class_name: str,
    targets: TargetProtocol,
    include_generics: bool = True,
) -> bool:
    sound_event_class = targets.encode_class(sound_event)

    if sound_event_class is None and include_generics:
        # Sound events that are generic could be of the given
        # class
        return True

    return sound_event_class == class_name
