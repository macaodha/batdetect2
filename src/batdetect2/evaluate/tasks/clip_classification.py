from collections import defaultdict
from typing import Literal

from pydantic import Field
from soundevent import data

from batdetect2.evaluate.metrics.clip_classification import (
    ClipClassificationAveragePrecisionConfig,
    ClipClassificationMetricConfig,
    ClipEval,
    build_clip_metric,
)
from batdetect2.evaluate.plots.clip_classification import (
    ClipClassificationPlotConfig,
    build_clip_classification_plotter,
)
from batdetect2.evaluate.tasks.base import (
    BaseTask,
    BaseTaskConfig,
    tasks_registry,
)
from batdetect2.postprocess.types import ClipDetections
from batdetect2.targets.types import TargetProtocol


class ClipClassificationTaskConfig(BaseTaskConfig):
    name: Literal["clip_classification"] = "clip_classification"
    prefix: str = "clip_classification"
    metrics: list[ClipClassificationMetricConfig] = Field(
        default_factory=lambda: [
            ClipClassificationAveragePrecisionConfig(),
        ]
    )
    plots: list[ClipClassificationPlotConfig] = Field(default_factory=list)


class ClipClassificationTask(BaseTask[ClipEval]):
    def evaluate_clip(
        self,
        clip_annotation: data.ClipAnnotation,
        prediction: ClipDetections,
    ) -> ClipEval:
        clip = clip_annotation.clip

        gt_classes = set()
        for sound_event in clip_annotation.sound_events:
            if not self.include_sound_event_annotation(sound_event, clip):
                continue

            class_name = self.targets.encode_class(sound_event)

            if class_name is None:
                continue

            gt_classes.add(class_name)

        pred_scores = defaultdict(float)
        for pred in prediction.detections:
            if not self.include_prediction(pred, clip):
                continue

            for class_idx, class_name in enumerate(self.targets.class_names):
                pred_scores[class_name] = max(
                    float(pred.class_scores[class_idx]),
                    pred_scores[class_name],
                )

        return ClipEval(true_classes=gt_classes, class_scores=pred_scores)

    @tasks_registry.register(ClipClassificationTaskConfig)
    @staticmethod
    def from_config(
        config: ClipClassificationTaskConfig,
        targets: TargetProtocol,
    ):
        metrics = [build_clip_metric(metric) for metric in config.metrics]
        plots = [
            build_clip_classification_plotter(plot, targets)
            for plot in config.plots
        ]
        return ClipClassificationTask(
            prefix=config.prefix,
            plots=plots,
            metrics=metrics,
            targets=targets,
        )
