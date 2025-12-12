from typing import Literal

from pydantic import Field
from soundevent import data
from soundevent.evaluation import match_detections_and_gts

from batdetect2.evaluate.metrics.detection import (
    ClipEval,
    DetectionAveragePrecisionConfig,
    DetectionMetricConfig,
    MatchEval,
    build_detection_metric,
)
from batdetect2.evaluate.plots.detection import (
    DetectionPlotConfig,
    build_detection_plotter,
)
from batdetect2.evaluate.tasks.base import (
    BaseSEDTask,
    BaseSEDTaskConfig,
    tasks_registry,
)
from batdetect2.typing import TargetProtocol
from batdetect2.typing.postprocess import ClipDetections


class DetectionTaskConfig(BaseSEDTaskConfig):
    name: Literal["sound_event_detection"] = "sound_event_detection"
    prefix: str = "detection"
    metrics: list[DetectionMetricConfig] = Field(
        default_factory=lambda: [DetectionAveragePrecisionConfig()]
    )
    plots: list[DetectionPlotConfig] = Field(default_factory=list)


class DetectionTask(BaseSEDTask[ClipEval]):
    def evaluate_clip(
        self,
        clip_annotation: data.ClipAnnotation,
        prediction: ClipDetections,
    ) -> ClipEval:
        clip = clip_annotation.clip

        gts = [
            sound_event
            for sound_event in clip_annotation.sound_events
            if self.include_sound_event_annotation(sound_event, clip)
        ]
        preds = [
            pred
            for pred in prediction.detections
            if self.include_prediction(pred, clip)
        ]

        matches = []
        for match in match_detections_and_gts(
            detections=preds,
            ground_truths=gts,
            affinity=self.affinity,
            score=lambda pred: pred.detection_score,
            strict_match=self.strict_match,
            affinity_threshold=self.affinity_threshold,
        ):
            matches.append(
                MatchEval(
                    gt=match.annotation,
                    pred=match.prediction,
                    is_prediction=match.prediction is not None,
                    is_ground_truth=match.annotation is not None,
                    score=match.prediction_score,
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
        plots = [
            build_detection_plotter(plot, targets) for plot in config.plots
        ]
        return DetectionTask.build(
            config=config,
            metrics=metrics,
            targets=targets,
            plots=plots,
        )
