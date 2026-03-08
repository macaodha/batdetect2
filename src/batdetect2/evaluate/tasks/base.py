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

from batdetect2.core import BaseConfig, Registry
from batdetect2.evaluate.affinity import (
    AffinityConfig,
    TimeAffinityConfig,
    build_affinity_function,
)
from batdetect2.typing import (
    AffinityFunction,
    ClipDetections,
    Detection,
    EvaluatorProtocol,
    TargetProtocol,
)

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


class BaseTask(EvaluatorProtocol, Generic[T_Output]):
    targets: TargetProtocol

    metrics: List[Callable[[Sequence[T_Output]], Dict[str, float]]]

    plots: List[Callable[[Sequence[T_Output]], Iterable[Tuple[str, Figure]]]]

    prefix: str

    ignore_start_end: float

    def __init__(
        self,
        targets: TargetProtocol,
        metrics: List[Callable[[Sequence[T_Output]], Dict[str, float]]],
        prefix: str,
        plots: List[
            Callable[[Sequence[T_Output]], Iterable[Tuple[str, Figure]]]
        ]
        | None = None,
        ignore_start_end: float = 0.01,
    ):
        self.prefix = prefix
        self.targets = targets
        self.metrics = metrics
        self.plots = plots or []
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
        predictions: Sequence[ClipDetections],
    ) -> List[T_Output]:
        return [
            self.evaluate_clip(clip_annotation, preds)
            for clip_annotation, preds in zip(
                clip_annotations, predictions, strict=False
            )
        ]

    def evaluate_clip(
        self,
        clip_annotation: data.ClipAnnotation,
        prediction: ClipDetections,
    ) -> T_Output: ...  # ty: ignore[empty-body]

    def include_sound_event_annotation(
        self,
        sound_event_annotation: data.SoundEventAnnotation,
        clip: data.Clip,
    ) -> bool:
        if not self.targets.filter(sound_event_annotation):
            return False

        geometry = sound_event_annotation.sound_event.geometry
        return is_in_bounds(
            geometry,
            clip,
            self.ignore_start_end,
        )

    def include_prediction(
        self,
        prediction: Detection,
        clip: data.Clip,
    ) -> bool:
        return is_in_bounds(
            prediction.geometry,
            clip,
            self.ignore_start_end,
        )


class BaseSEDTaskConfig(BaseTaskConfig):
    affinity: AffinityConfig = Field(default_factory=TimeAffinityConfig)
    affinity_threshold: float = 0
    strict_match: bool = True


class BaseSEDTask(BaseTask[T_Output]):
    affinity: AffinityFunction

    def __init__(
        self,
        prefix: str,
        targets: TargetProtocol,
        metrics: List[Callable[[Sequence[T_Output]], Dict[str, float]]],
        affinity: AffinityFunction,
        plots: List[
            Callable[[Sequence[T_Output]], Iterable[Tuple[str, Figure]]]
        ]
        | None = None,
        affinity_threshold: float = 0,
        ignore_start_end: float = 0.01,
        strict_match: bool = True,
    ):
        super().__init__(
            prefix=prefix,
            metrics=metrics,
            plots=plots,
            targets=targets,
            ignore_start_end=ignore_start_end,
        )
        self.affinity = affinity
        self.affinity_threshold = affinity_threshold
        self.strict_match = strict_match

    @classmethod
    def build(
        cls,
        config: BaseSEDTaskConfig,
        targets: TargetProtocol,
        **kwargs,
    ):
        affinity = build_affinity_function(config.affinity)
        return cls(
            affinity=affinity,
            affinity_threshold=config.affinity_threshold,
            prefix=config.prefix,
            ignore_start_end=config.ignore_start_end,
            strict_match=config.strict_match,
            targets=targets,
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
