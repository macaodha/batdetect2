from dataclasses import dataclass
from typing import (
    Dict,
    Generic,
    Iterable,
    List,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
)

from matplotlib.figure import Figure
from soundevent import data

from batdetect2.typing.postprocess import ClipDetections, Detection
from batdetect2.typing.targets import TargetProtocol

__all__ = [
    "EvaluatorProtocol",
    "MetricsProtocol",
    "MatchEvaluation",
]


@dataclass
class MatchEvaluation:
    clip: data.Clip

    sound_event_annotation: data.SoundEventAnnotation | None
    gt_det: bool
    gt_class: str | None
    gt_geometry: data.Geometry | None

    pred_score: float
    pred_class_scores: Dict[str, float]
    pred_geometry: data.Geometry | None

    affinity: float

    @property
    def top_class(self) -> str | None:
        if not self.pred_class_scores:
            return None

        return max(self.pred_class_scores, key=self.pred_class_scores.get)  # type: ignore

    @property
    def is_prediction(self) -> bool:
        return self.pred_geometry is not None

    @property
    def is_generic(self) -> bool:
        return self.gt_det and self.gt_class is None

    @property
    def top_class_score(self) -> float:
        pred_class = self.top_class

        if pred_class is None:
            return 0

        return self.pred_class_scores[pred_class]


@dataclass
class ClipMatches:
    clip: data.Clip
    matches: List[MatchEvaluation]


class MatcherProtocol(Protocol):
    def __call__(
        self,
        ground_truth: Sequence[data.Geometry],
        predictions: Sequence[data.Geometry],
        scores: Sequence[float],
    ) -> Iterable[Tuple[int | None, int | None, float]]: ...


Geom = TypeVar("Geom", bound=data.Geometry, contravariant=True)


class AffinityFunction(Protocol):
    def __call__(
        self,
        detection: Detection,
        ground_truth: data.SoundEventAnnotation,
    ) -> float: ...


class MetricsProtocol(Protocol):
    def __call__(
        self,
        clip_annotations: Sequence[data.ClipAnnotation],
        predictions: Sequence[Sequence[Detection]],
    ) -> Dict[str, float]: ...


class PlotterProtocol(Protocol):
    def __call__(
        self,
        clip_annotations: Sequence[data.ClipAnnotation],
        predictions: Sequence[Sequence[Detection]],
    ) -> Iterable[Tuple[str, Figure]]: ...


EvaluationOutput = TypeVar("EvaluationOutput")


class EvaluatorProtocol(Protocol, Generic[EvaluationOutput]):
    targets: TargetProtocol

    def evaluate(
        self,
        clip_annotations: Sequence[data.ClipAnnotation],
        predictions: Sequence[ClipDetections],
    ) -> EvaluationOutput: ...

    def compute_metrics(
        self, eval_outputs: EvaluationOutput
    ) -> Dict[str, float]: ...

    def generate_plots(
        self, eval_outputs: EvaluationOutput
    ) -> Iterable[Tuple[str, Figure]]: ...
