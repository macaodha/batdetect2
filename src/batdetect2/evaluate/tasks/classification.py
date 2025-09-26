from typing import (
    List,
    Literal,
    Sequence,
)

from pydantic import Field
from soundevent import data

from batdetect2.evaluate.metrics.classification import (
    ClassificationAveragePrecisionConfig,
    ClassificationMetricConfig,
    ClipEval,
    MatchEval,
    build_classification_metrics,
)
from batdetect2.evaluate.tasks.base import (
    BaseTask,
    BaseTaskConfig,
    tasks_registry,
)
from batdetect2.typing import RawPrediction, TargetProtocol


class ClassificationTaskConfig(BaseTaskConfig):
    name: Literal["sound_event_classification"] = "sound_event_classification"
    prefix: str = "classification"
    metrics: List[ClassificationMetricConfig] = Field(
        default_factory=lambda: [ClassificationAveragePrecisionConfig()]
    )
    include_generics: bool = True


class ClassificationTask(BaseTask[ClipEval]):
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
        predictions: Sequence[RawPrediction],
    ) -> ClipEval:
        clip = clip_annotation.clip

        preds = [
            pred for pred in predictions if self.include_prediction(pred, clip)
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
                if self.is_class(sound_event, class_name)
            ]
            scores = [float(pred.class_scores[class_idx]) for pred in preds]

            matches = []

            for pred_idx, gt_idx, _ in self.matcher(
                ground_truth=[se.sound_event.geometry for se in gts],  # type: ignore
                predictions=[pred.geometry for pred in preds],
                scores=scores,
            ):
                gt = gts[gt_idx] if gt_idx is not None else None
                pred = preds[pred_idx] if pred_idx is not None else None

                true_class = (
                    self.targets.encode_class(gt) if gt is not None else None
                )

                score = (
                    float(pred.class_scores[class_idx])
                    if pred is not None
                    else 0
                )

                matches.append(
                    MatchEval(
                        gt=gt,
                        pred=pred,
                        is_prediction=pred is not None,
                        is_ground_truth=gt is not None,
                        is_generic=gt is not None and true_class is None,
                        true_class=true_class,
                        score=score,
                    )
                )

            per_class_matches[class_name] = matches

        return ClipEval(clip=clip, matches=per_class_matches)

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

    @tasks_registry.register(ClassificationTaskConfig)
    @staticmethod
    def from_config(
        config: ClassificationTaskConfig,
        targets: TargetProtocol,
    ):
        metrics = [
            build_classification_metrics(metric) for metric in config.metrics
        ]
        return ClassificationTask.build(
            config=config,
            targets=targets,
            metrics=metrics,
        )
