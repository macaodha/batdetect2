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

from soundevent import data

__all__ = [
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
        time_buffer: float = 0.01,
        freq_buffer: float = 1000,
    ) -> float: ...


class MetricsProtocol(Protocol):
    def __call__(self, matches: List[MatchEvaluation]) -> Dict[str, float]: ...
