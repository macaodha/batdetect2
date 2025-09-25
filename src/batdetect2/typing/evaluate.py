from dataclasses import dataclass
from typing import (
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
)

from matplotlib.figure import Figure
from soundevent import data

from batdetect2.typing.postprocess import RawPrediction
from batdetect2.typing.targets import TargetProtocol

__all__ = [
    "EvaluatorProtocol",
    "MetricsProtocol",
    "MatchEvaluation",
]


@dataclass
class MatchEvaluation:
    clip: data.Clip

    sound_event_annotation: Optional[data.SoundEventAnnotation]
    gt_det: bool
    gt_class: Optional[str]
    gt_geometry: Optional[data.Geometry]

    pred_score: float
    pred_class_scores: Dict[str, float]
    pred_geometry: Optional[data.Geometry]

    affinity: float

    @property
    def top_class(self) -> Optional[str]:
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
    ) -> Iterable[Tuple[Optional[int], Optional[int], float]]: ...


Geom = TypeVar("Geom", bound=data.Geometry, contravariant=True)


class AffinityFunction(Protocol, Generic[Geom]):
    def __call__(
        self,
        geometry1: Geom,
        geometry2: Geom,
    ) -> float: ...


class MetricsProtocol(Protocol):
    def __call__(
        self,
        clip_annotations: Sequence[data.ClipAnnotation],
        predictions: Sequence[Sequence[RawPrediction]],
    ) -> Dict[str, float]: ...


class PlotterProtocol(Protocol):
    def __call__(
        self,
        clip_annotations: Sequence[data.ClipAnnotation],
        predictions: Sequence[Sequence[RawPrediction]],
    ) -> Iterable[Tuple[str, Figure]]: ...


EvaluationOutput = TypeVar("EvaluationOutput")


class EvaluatorProtocol(Protocol, Generic[EvaluationOutput]):
    targets: TargetProtocol

    def evaluate(
        self,
        clip_annotations: Sequence[data.ClipAnnotation],
        predictions: Sequence[Sequence[RawPrediction]],
    ) -> EvaluationOutput: ...

    def compute_metrics(
        self, eval_outputs: EvaluationOutput
    ) -> Dict[str, float]: ...

    def generate_plots(
        self, eval_outputs: EvaluationOutput
    ) -> Iterable[Tuple[str, Figure]]: ...
