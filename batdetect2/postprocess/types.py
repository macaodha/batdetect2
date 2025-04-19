from typing import Dict, NamedTuple, Protocol

import numpy as np

__all__ = [
    "BatDetect2Prediction",
]


class BatDetect2Prediction(NamedTuple):
    start_time: float
    end_time: float
    low_freq: float
    high_freq: float
    detection_score: float
    class_scores: Dict[str, float]
    features: np.ndarray


class PostprocessorProtocol(Protocol):
    pass
