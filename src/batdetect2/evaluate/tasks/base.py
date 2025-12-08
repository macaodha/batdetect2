from typing import (
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Sequence,
    Tuple,
    TypeVar,
)

from matplotlib.figure import Figure
from pydantic import Field
from soundevent import data
from soundevent.geometry import compute_bounds

from batdetect2.core import BaseConfig
from batdetect2.core.registries import Registry
from batdetect2.evaluate.match import (
    MatchConfig,
    StartTimeMatchConfig,
    build_matcher,
)
from batdetect2.typing.evaluate import EvaluatorProtocol, MatcherProtocol
from batdetect2.typing.postprocess import BatDetect2Prediction, RawPrediction
from batdetect2.typing.targets import TargetProtocol

__all__ = [
    "BaseTaskConfig",
    "BaseTask",
]

tasks_registry: Registry[EvaluatorProtocol, [TargetProtocol]] = Registry(
    "tasks"
)


T_Output = TypeVar("T_Output")


class BaseTaskConfig(BaseConfig):
    prefix: str
    ignore_start_end: float = 0.01
    matching_strategy: MatchConfig = Field(
        default_factory=StartTimeMatchConfig
    )


class BaseTask(EvaluatorProtocol, Generic[T_Output]):
    targets: TargetProtocol

    matcher: MatcherProtocol

    metrics: List[Callable[[Sequence[T_Output]], Dict[str, float]]]

    plots: List[Callable[[Sequence[T_Output]], Iterable[Tuple[str, Figure]]]]

    ignore_start_end: float

    prefix: str

    def __init__(
        self,
        matcher: MatcherProtocol,
        targets: TargetProtocol,
        metrics: List[Callable[[Sequence[T_Output]], Dict[str, float]]],
        prefix: str,
        ignore_start_end: float = 0.01,
        plots: List[Callable[[Sequence[T_Output]], Iterable[Tuple[str, Figure]]]] | None = None,
    ):
        self.matcher = matcher
        self.metrics = metrics
        self.plots = plots or []
        self.targets = targets
        self.prefix = prefix
        self.ignore_start_end = ignore_start_end

    def compute_metrics(
        self,
        eval_outputs: List[T_Output],
    ) -> Dict[str, float]:
        scores = [metric(eval_outputs) for metric in self.metrics]
        return {
            f"{self.prefix}/{name}": score
            for metric_output in scores
            for name, score in metric_output.items()
        }

    def generate_plots(
        self, eval_outputs: List[T_Output]
    ) -> Iterable[Tuple[str, Figure]]:
        for plot in self.plots:
            for name, fig in plot(eval_outputs):
                yield f"{self.prefix}/{name}", fig

    def evaluate(
        self,
        clip_annotations: Sequence[data.ClipAnnotation],
        predictions: Sequence[BatDetect2Prediction],
    ) -> List[T_Output]:
        return [
            self.evaluate_clip(clip_annotation, preds)
            for clip_annotation, preds in zip(clip_annotations, predictions, strict=False)
        ]

    def evaluate_clip(
        self,
        clip_annotation: data.ClipAnnotation,
        prediction: BatDetect2Prediction,
    ) -> T_Output: ...

    def include_sound_event_annotation(
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
            self.ignore_start_end,
        )

    def include_prediction(
        self,
        prediction: RawPrediction,
        clip: data.Clip,
    ) -> bool:
        return is_in_bounds(
            prediction.geometry,
            clip,
            self.ignore_start_end,
        )

    @classmethod
    def build(
        cls,
        config: BaseTaskConfig,
        targets: TargetProtocol,
        metrics: List[Callable[[Sequence[T_Output]], Dict[str, float]]],
        plots: List[Callable[[Sequence[T_Output]], Iterable[Tuple[str, Figure]]]] | None = None,
        **kwargs,
    ):
        matcher = build_matcher(config.matching_strategy)
        return cls(
            matcher=matcher,
            targets=targets,
            metrics=metrics,
            plots=plots,
            prefix=config.prefix,
            ignore_start_end=config.ignore_start_end,
            **kwargs,
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
