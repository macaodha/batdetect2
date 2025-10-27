from typing import List, Literal

from pydantic import Field
from soundevent import data

from batdetect2.evaluate.metrics.clip_detection import (
    ClipDetectionAveragePrecisionConfig,
    ClipDetectionMetricConfig,
    ClipEval,
    build_clip_metric,
)
from batdetect2.evaluate.plots.clip_detection import (
    ClipDetectionPlotConfig,
    build_clip_detection_plotter,
)
from batdetect2.evaluate.tasks.base import (
    BaseTask,
    BaseTaskConfig,
    tasks_registry,
)
from batdetect2.typing import TargetProtocol
from batdetect2.typing.postprocess import BatDetect2Prediction


class ClipDetectionTaskConfig(BaseTaskConfig):
    name: Literal["clip_detection"] = "clip_detection"
    prefix: str = "clip_detection"
    metrics: List[ClipDetectionMetricConfig] = Field(
        default_factory=lambda: [
            ClipDetectionAveragePrecisionConfig(),
        ]
    )
    plots: List[ClipDetectionPlotConfig] = Field(default_factory=list)


class ClipDetectionTask(BaseTask[ClipEval]):
    def evaluate_clip(
        self,
        clip_annotation: data.ClipAnnotation,
        prediction: BatDetect2Prediction,
    ) -> ClipEval:
        clip = clip_annotation.clip

        gt_det = any(
            self.include_sound_event_annotation(sound_event, clip)
            for sound_event in clip_annotation.sound_events
        )

        pred_score = 0
        for pred in prediction.predictions:
            if not self.include_prediction(pred, clip):
                continue

            pred_score = max(pred_score, pred.detection_score)

        return ClipEval(
            gt_det=gt_det,
            score=pred_score,
        )

    @tasks_registry.register(ClipDetectionTaskConfig)
    @staticmethod
    def from_config(
        config: ClipDetectionTaskConfig,
        targets: TargetProtocol,
    ):
        metrics = [build_clip_metric(metric) for metric in config.metrics]
        plots = [
            build_clip_detection_plotter(plot, targets)
            for plot in config.plots
        ]
        return ClipDetectionTask.build(
            config=config,
            metrics=metrics,
            targets=targets,
            plots=plots,
        )
