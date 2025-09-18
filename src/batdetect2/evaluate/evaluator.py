from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from matplotlib.figure import Figure
from soundevent import data
from soundevent.geometry import compute_bounds

from batdetect2.evaluate.config import EvaluationConfig
from batdetect2.evaluate.match import build_matcher, match
from batdetect2.evaluate.metrics import build_metric
from batdetect2.evaluate.plots import build_plotter
from batdetect2.targets import build_targets
from batdetect2.typing.evaluate import (
    ClipEvaluation,
    EvaluatorProtocol,
    MatcherProtocol,
    MetricsProtocol,
    PlotterProtocol,
)
from batdetect2.typing.postprocess import RawPrediction
from batdetect2.typing.targets import TargetProtocol

__all__ = [
    "Evaluator",
    "build_evaluator",
]


class Evaluator:
    def __init__(
        self,
        config: EvaluationConfig,
        targets: TargetProtocol,
        matcher: MatcherProtocol,
        metrics: List[MetricsProtocol],
        plots: List[PlotterProtocol],
    ):
        self.config = config
        self.targets = targets
        self.matcher = matcher
        self.metrics = metrics
        self.plots = plots

    def match(
        self,
        clip_annotation: data.ClipAnnotation,
        predictions: Sequence[RawPrediction],
    ) -> ClipEvaluation:
        clip = clip_annotation.clip
        ground_truth = [
            sound_event
            for sound_event in clip_annotation.sound_events
            if self.filter_sound_event_annotations(sound_event, clip)
        ]
        predictions = [
            prediction
            for prediction in predictions
            if self.filter_predictions(prediction, clip)
        ]
        return ClipEvaluation(
            clip=clip_annotation.clip,
            matches=match(
                ground_truth,
                predictions,
                clip=clip,
                targets=self.targets,
                matcher=self.matcher,
            ),
        )

    def filter_sound_event_annotations(
        self,
        sound_event_annotation: data.SoundEventAnnotation,
        clip: data.Clip,
    ) -> bool:
        if not self.targets.filter(sound_event_annotation):
            return False

        geometry = sound_event_annotation.sound_event.geometry
        if geometry is None:
            return False

        return is_in_bounds(
            geometry,
            clip,
            self.config.ignore_start_end,
        )

    def filter_predictions(
        self,
        prediction: RawPrediction,
        clip: data.Clip,
    ) -> bool:
        return is_in_bounds(
            prediction.geometry,
            clip,
            self.config.ignore_start_end,
        )

    def evaluate(
        self,
        clip_annotations: Sequence[data.ClipAnnotation],
        predictions: Sequence[Sequence[RawPrediction]],
    ) -> List[ClipEvaluation]:
        if len(clip_annotations) != len(predictions):
            raise ValueError(
                "Number of annotated clips and sets of predictions do not match"
            )

        return [
            self.match(clip_annotation, preds)
            for clip_annotation, preds in zip(clip_annotations, predictions)
        ]

    def compute_metrics(
        self,
        clip_evaluations: Sequence[ClipEvaluation],
    ) -> Dict[str, float]:
        results = {}

        for metric in self.metrics:
            results.update(metric(clip_evaluations))

        return results

    def generate_plots(
        self, clip_evaluations: Sequence[ClipEvaluation]
    ) -> Iterable[Tuple[str, Figure]]:
        for plotter in self.plots:
            for name, fig in plotter(clip_evaluations):
                yield name, fig


def build_evaluator(
    config: Optional[EvaluationConfig] = None,
    targets: Optional[TargetProtocol] = None,
    matcher: Optional[MatcherProtocol] = None,
    plots: Optional[List[PlotterProtocol]] = None,
    metrics: Optional[List[MetricsProtocol]] = None,
) -> EvaluatorProtocol:
    config = config or EvaluationConfig()
    targets = targets or build_targets()
    matcher = matcher or build_matcher(config.match_strategy)

    if metrics is None:
        metrics = [
            build_metric(config, targets.class_names)
            for config in config.metrics
        ]

    if plots is None:
        plots = [
            build_plotter(config, targets.class_names)
            for config in config.plots
        ]

    return Evaluator(
        config=config,
        targets=targets,
        matcher=matcher,
        metrics=metrics,
        plots=plots,
    )


def is_in_bounds(
    geometry: data.Geometry,
    clip: data.Clip,
    buffer: float,
) -> bool:
    start_time = compute_bounds(geometry)[0]
    return (start_time >= clip.start_time + buffer) and (
        start_time <= clip.end_time - buffer
    )
