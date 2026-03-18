from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple, Protocol

import numpy as np
import torch
from soundevent import data

from batdetect2.targets.types import Position, Size

if TYPE_CHECKING:
    from batdetect2.models.types import ModelOutput

__all__ = [
    "ClipDetections",
    "ClipDetectionsArray",
    "ClipDetectionsTensor",
    "ClipPrediction",
    "Detection",
    "GeometryDecoder",
    "PostprocessorProtocol",
]


class GeometryDecoder(Protocol):
    def __call__(
        self,
        position: Position,
        size: Size,
        class_name: str | None = None,
    ) -> data.Geometry: ...


@dataclass
class Detection:
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
class ClipDetections:
    clip: data.Clip
    detections: list[Detection]


@dataclass
class ClipPrediction:
    clip: data.Clip
    detection_score: float
    class_scores: np.ndarray


class PostprocessorProtocol(Protocol):
    def __call__(
        self, output: "ModelOutput"
    ) -> list[ClipDetectionsTensor]: ...
