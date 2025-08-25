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
from typing import List, NamedTuple, Optional, Protocol

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
    """Intermediate representation of a single detected sound event.

    Holds extracted information about a detection after initial processing
    (like peak finding, coordinate remapping, geometry recovery) but before
    final class decoding and conversion into a `SoundEventPrediction`. This
    can be useful for evaluation or simpler data handling formats.

    Attributes
    ----------
    geometry: data.Geometry
        The recovered estimated geometry of the detected sound event.
        Usually a bounding box.
    detection_score : float
        The confidence score associated with this detection, typically from
        the detection heatmap peak.
    class_scores : xr.DataArray
        An xarray DataArray containing the predicted probabilities or scores
        for each target class at the detection location. Indexed by a
        'category' coordinate containing class names.
    features : xr.DataArray
        An xarray DataArray containing extracted feature vectors at the
        detection location. Indexed by a 'feature' coordinate.
    """

    geometry: data.Geometry
    detection_score: float
    class_scores: np.ndarray
    features: np.ndarray


class Detections(NamedTuple):
    scores: torch.Tensor
    sizes: torch.Tensor
    class_scores: torch.Tensor
    times: torch.Tensor
    frequencies: torch.Tensor
    features: torch.Tensor


@dataclass
class BatDetect2Prediction:
    raw: RawPrediction
    sound_event_prediction: data.SoundEventPrediction


class PostprocessorProtocol(Protocol):
    """Protocol defining the interface for the full postprocessing pipeline."""

    def get_detections(
        self,
        output: ModelOutput,
        clips: Optional[List[data.Clip]] = None,
    ) -> List[Detections]: ...

    def get_raw_predictions(
        self,
        output: ModelOutput,
        clips: List[data.Clip],
    ) -> List[List[RawPrediction]]:
        """Extract intermediate RawPrediction objects for a batch.

        Processes the raw model output for a batch through remapping, NMS,
        detection, data extraction, and geometry recovery to produce a list of
        `RawPrediction` objects for each corresponding input clip. This provides
        a simplified, intermediate representation before final tag decoding.

        Parameters
        ----------
        output : ModelOutput
            The raw output from the neural network model for a batch.
        clips : List[data.Clip]
            A list of `soundevent.data.Clip` objects corresponding to the batch
            items, providing context. Must match the batch size of `output`.

        Returns
        -------
        List[List[RawPrediction]]
            A list of lists (one inner list per input clip, in order). Each
            inner list contains the `RawPrediction` objects extracted for the
            corresponding input clip.
        """
        ...

    def get_sound_event_predictions(
        self, output: ModelOutput, clips: List[data.Clip]
    ) -> List[List[BatDetect2Prediction]]: ...

    def get_predictions(
        self,
        output: ModelOutput,
        clips: List[data.Clip],
    ) -> List[data.ClipPrediction]:
        """Perform the full postprocessing pipeline for a batch.

        Takes raw model output for a batch and corresponding clips, applies the
        entire postprocessing chain, and returns the final, interpretable
        predictions as a list of `soundevent.data.ClipPrediction` objects.

        Parameters
        ----------
        output : ModelOutput
            The raw output from the neural network model for a batch.
        clips : List[data.Clip]
            A list of `soundevent.data.Clip` objects corresponding to the batch
            items, providing context. Must match the batch size of `output`.

        Returns
        -------
        List[data.ClipPrediction]
            A list containing one `ClipPrediction` object for each input clip
            (in the same order), populated with `SoundEventPrediction` objects
            representing the final detections with decoded tags and geometry.
        """
        ...
