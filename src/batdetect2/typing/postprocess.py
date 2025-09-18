"""Defines shared interfaces and data structures for postprocessing.

This module centralizes the Protocol definitions and common data structures
used throughout the `batdetect2.postprocess` module.

The main component is the `PostprocessorProtocol`, which outlines the standard
interface for an object responsible for executing the entire postprocessing
pipeline. This pipeline transforms raw neural network outputs into interpretable
detections represented as `soundevent` objects. Using protocols ensures
modularity and consistent interaction between different parts of the BatDetect2
system that deal with model predictions.
"""

from dataclasses import dataclass
from typing import List, NamedTuple, Optional, Protocol, Sequence

import numpy as np
import torch
from soundevent import data

from batdetect2.typing.models import ModelOutput
from batdetect2.typing.targets import Position, Size

__all__ = [
    "RawPrediction",
    "PostprocessorProtocol",
    "GeometryDecoder",
]


# TODO: update the docstring
class GeometryDecoder(Protocol):
    """Type alias for a function that recovers geometry from position and size.

    This callable takes:
    1.  A position tuple `(time, frequency)`.
    2.  A NumPy array of size dimensions (e.g., `[width, height]`).
    3.  Optionally a class name of the highest scoring class. This is to accomodate
        different ways of decoding geometry that depend on the predicted class.
    It should return the reconstructed `soundevent.data.Geometry` (typically a
    `BoundingBox`).
    """

    def __call__(
        self, position: Position, size: Size, class_name: Optional[str] = None
    ) -> data.Geometry: ...


class RawPrediction(NamedTuple):
    geometry: data.Geometry
    detection_score: float
    class_scores: np.ndarray
    features: np.ndarray


class ClipDetectionsArray(NamedTuple):
    scores: np.ndarray
    sizes: np.ndarray
    class_scores: np.ndarray
    times: np.ndarray
    frequencies: np.ndarray
    features: np.ndarray


class ClipDetectionsTensor(NamedTuple):
    scores: torch.Tensor
    sizes: torch.Tensor
    class_scores: torch.Tensor
    times: torch.Tensor
    frequencies: torch.Tensor
    features: torch.Tensor

    def numpy(self) -> ClipDetectionsArray:
        return ClipDetectionsArray(
            scores=self.scores.detach().cpu().numpy(),
            sizes=self.sizes.detach().cpu().numpy(),
            class_scores=self.class_scores.detach().cpu().numpy(),
            times=self.times.detach().cpu().numpy(),
            frequencies=self.frequencies.detach().cpu().numpy(),
            features=self.features.detach().cpu().numpy(),
        )


@dataclass
class BatDetect2Prediction:
    raw: RawPrediction
    sound_event_prediction: data.SoundEventPrediction


class PostprocessorProtocol(Protocol):
    """Protocol defining the interface for the full postprocessing pipeline."""

    def __call__(
        self,
        output: ModelOutput,
        start_times: Optional[Sequence[float]] = None,
    ) -> List[ClipDetectionsTensor]: ...
