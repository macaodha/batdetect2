from typing import Literal

from pydantic import Field
from soundevent import data
from soundevent.evaluation import match_detections_and_gts

from batdetect2.evaluate.metrics.top_class import (
    ClipEval,
    MatchEval,
    TopClassAveragePrecisionConfig,
    TopClassMetricConfig,
    build_top_class_metric,
)
from batdetect2.evaluate.plots.top_class import (
    TopClassPlotConfig,
    build_top_class_plotter,
)
from batdetect2.evaluate.tasks.base import (
    BaseSEDTask,
    BaseSEDTaskConfig,
    tasks_registry,
)
from batdetect2.postprocess.types import ClipDetections
from batdetect2.targets.types import TargetProtocol


def _default_metrics() -> list[TopClassMetricConfig]:
    return [TopClassAveragePrecisionConfig()]


class TopClassDetectionTaskConfig(BaseSEDTaskConfig):
    name: Literal["top_class_detection"] = "top_class_detection"
    prefix: str = "top_class"
    metrics: list[TopClassMetricConfig] = Field(
        default_factory=_default_metrics
    )
    plots: list[TopClassPlotConfig] = Field(default_factory=list)


class TopClassDetectionTask(BaseSEDTask[ClipEval]):
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
            ground_truths=gts,
            detections=preds,
            affinity=self.affinity,
            score=lambda pred: pred.class_scores.max(),
            strict_match=self.strict_match,
            affinity_threshold=self.affinity_threshold,
        ):
            gt = match.annotation
            pred = match.prediction
            true_class = (
                self.targets.encode_class(gt) if gt is not None else None
            )

            class_idx = (
                pred.class_scores.argmax() if pred is not None else None
            )
            pred_class = (
                self.targets.class_names[class_idx]
                if class_idx is not None
                else None
            )

            matches.append(
                MatchEval(
                    clip=clip,
                    gt=gt,
                    pred=pred,
                    is_ground_truth=gt is not None,
                    is_prediction=pred is not None,
                    true_class=true_class,
                    is_generic=gt is not None and true_class is None,
                    pred_class=pred_class,
                    score=match.prediction_score,
                )
            )

        return ClipEval(clip=clip, matches=matches)

    @tasks_registry.register(TopClassDetectionTaskConfig)
    @staticmethod
    def from_config(
        config: TopClassDetectionTaskConfig,
        targets: TargetProtocol,
    ):
        metrics = [build_top_class_metric(metric) for metric in config.metrics]
        plots = [
            build_top_class_plotter(plot, targets) for plot in config.plots
        ]
        return TopClassDetectionTask.build(
            config=config,
            plots=plots,
            metrics=metrics,
            targets=targets,
        )
