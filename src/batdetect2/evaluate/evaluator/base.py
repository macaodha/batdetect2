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
from batdetect2.typing.postprocess import RawPrediction
from batdetect2.typing.targets import TargetProtocol

__all__ = [
    "BaseEvaluatorConfig",
    "BaseEvaluator",
]

evaluators: Registry[EvaluatorProtocol, [TargetProtocol]] = Registry("metric")


class BaseEvaluatorConfig(BaseConfig):
    prefix: str
    ignore_start_end: float = 0.01
    matching_strategy: MatchConfig = Field(
        default_factory=StartTimeMatchConfig
    )


class BaseEvaluator(EvaluatorProtocol):
    targets: TargetProtocol

    matcher: MatcherProtocol

    ignore_start_end: float

    prefix: str

    def __init__(
        self,
        matcher: MatcherProtocol,
        targets: TargetProtocol,
        prefix: str,
        ignore_start_end: float = 0.01,
    ):
        self.matcher = matcher
        self.targets = targets
        self.prefix = prefix
        self.ignore_start_end = ignore_start_end

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
            self.ignore_start_end,
        )

    def filter_predictions(
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
        config: BaseEvaluatorConfig,
        targets: TargetProtocol,
        **kwargs,
    ):
        matcher = build_matcher(config.matching_strategy)
        return cls(
            matcher=matcher,
            targets=targets,
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
