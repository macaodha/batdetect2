from typing import List, Literal, Sequence

from pydantic import Field
from soundevent import data

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
    BaseTask,
    BaseTaskConfig,
    tasks_registry,
)
from batdetect2.typing import RawPrediction, TargetProtocol


class TopClassDetectionTaskConfig(BaseTaskConfig):
    name: Literal["top_class_detection"] = "top_class_detection"
    prefix: str = "top_class"
    metrics: List[TopClassMetricConfig] = Field(
        default_factory=lambda: [TopClassAveragePrecisionConfig()]
    )
    plots: List[TopClassPlotConfig] = Field(default_factory=list)


class TopClassDetectionTask(BaseTask[ClipEval]):
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
        # Take the highest score for each prediction
        scores = [pred.class_scores.max() for pred in preds]

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

            class_idx = (
                pred.class_scores.argmax() if pred is not None else None
            )

            score = (
                float(pred.class_scores[class_idx]) if pred is not None else 0
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
                    score=score,
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
