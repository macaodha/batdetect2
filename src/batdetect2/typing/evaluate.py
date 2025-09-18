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

    pred_score: float
    pred_class_scores: Dict[str, float]
    pred_geometry: Optional[data.Geometry]

    affinity: float

    @property
    def pred_class(self) -> Optional[str]:
        if not self.pred_class_scores:
            return None

        return max(self.pred_class_scores, key=self.pred_class_scores.get)  # type: ignore

    @property
    def pred_class_score(self) -> float:
        pred_class = self.pred_class

        if pred_class is None:
            return 0

        return self.pred_class_scores[pred_class]

    def is_true_positive(self, threshold: float = 0) -> bool:
        return (
            self.gt_det
            and self.pred_score > threshold
            and self.gt_class == self.pred_class
        )

    def is_false_positive(self, threshold: float = 0) -> bool:
        return self.gt_det is None and self.pred_score > threshold

    def is_false_negative(self, threshold: float = 0) -> bool:
        return self.gt_det and self.pred_score <= threshold

    def is_cross_trigger(self, threshold: float = 0) -> bool:
        return (
            self.gt_det
            and self.pred_score > threshold
            and self.gt_class != self.pred_class
        )


@dataclass
class ClipEvaluation:
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
        self, clip_evaluations: Sequence[ClipEvaluation]
    ) -> Dict[str, float]: ...


class PlotterProtocol(Protocol):
    def __call__(
        self, clip_evaluations: Sequence[ClipEvaluation]
    ) -> Iterable[Tuple[str, Figure]]: ...


class EvaluatorProtocol(Protocol):
    targets: TargetProtocol

    def evaluate(
        self,
        clip_annotations: Sequence[data.ClipAnnotation],
        predictions: Sequence[Sequence[RawPrediction]],
    ) -> List[ClipEvaluation]: ...

    def compute_metrics(
        self, clip_evaluations: Sequence[ClipEvaluation]
    ) -> Dict[str, float]: ...

    def generate_plots(
        self, clip_evaluations: Sequence[ClipEvaluation]
    ) -> Iterable[Tuple[str, Figure]]: ...
