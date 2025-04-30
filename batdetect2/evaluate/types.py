from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol
from uuid import UUID

__all__ = [
    "MetricsProtocol",
    "Match",
]


@dataclass
class Match:
    gt_uuid: Optional[UUID]
    gt_det: bool
    gt_class: Optional[str]
    pred_score: float
    affinity: float
    class_scores: Dict[str, float]


class MetricsProtocol(Protocol):
    def __call__(self, matches: List[Match]) -> Dict[str, float]: ...
