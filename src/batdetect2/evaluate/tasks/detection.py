from typing import List, Literal, Sequence

from pydantic import Field
from soundevent import data

from batdetect2.evaluate.metrics.detection import (
    ClipEval,
    DetectionAveragePrecisionConfig,
    DetectionMetricConfig,
    MatchEval,
    build_detection_metric,
)
from batdetect2.evaluate.tasks.base import (
    BaseTask,
    BaseTaskConfig,
    tasks_registry,
)
from batdetect2.typing import RawPrediction, TargetProtocol


class DetectionTaskConfig(BaseTaskConfig):
    name: Literal["sound_event_detection"] = "sound_event_detection"
    prefix: str = "detection"
    metrics: List[DetectionMetricConfig] = Field(
        default_factory=lambda: [DetectionAveragePrecisionConfig()]
    )


class DetectionTask(BaseTask[ClipEval]):
    def evaluate_clip(
        self,
        clip_annotation: data.ClipAnnotation,
        predictions: Sequence[RawPrediction],
    ) -> ClipEval:
        clip = clip_annotation.clip

        gts = [
            sound_event
            for sound_event in clip_annotation.sound_events
            if self.include_sound_event_annotation(sound_event, clip)
        ]
        preds = [
            pred for pred in predictions if self.include_prediction(pred, clip)
        ]
        scores = [pred.detection_score for pred in preds]

        matches = []
        for pred_idx, gt_idx, _ in self.matcher(
            ground_truth=[se.sound_event.geometry for se in gts],  # type: ignore
            predictions=[pred.geometry for pred in preds],
            scores=scores,
        ):
            gt = gts[gt_idx] if gt_idx is not None else None
            pred = preds[pred_idx] if pred_idx is not None else None

            matches.append(
                MatchEval(
                    gt=gt,
                    pred=pred,
                    is_prediction=pred is not None,
                    is_ground_truth=gt is not None,
                    score=pred.detection_score if pred is not None else 0,
                )
            )

        return ClipEval(clip=clip, matches=matches)

    @tasks_registry.register(DetectionTaskConfig)
    @staticmethod
    def from_config(
        config: DetectionTaskConfig,
        targets: TargetProtocol,
    ):
        metrics = [build_detection_metric(metric) for metric in config.metrics]
        return DetectionTask.build(
            config=config,
            metrics=metrics,
            targets=targets,
        )
