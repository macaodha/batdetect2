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

from typing import Callable, List, NamedTuple, Protocol

import numpy as np
import xarray as xr
from soundevent import data

from batdetect2.models.types import ModelOutput

__all__ = [
    "RawPrediction",
    "PostprocessorProtocol",
    "GeometryBuilder",
]


GeometryBuilder = Callable[[tuple[float, float], np.ndarray], data.Geometry]
"""Type alias for a function that recovers geometry from position and size.

This callable takes:
1.  A position tuple `(time, frequency)`.
2.  A NumPy array of size dimensions (e.g., `[width, height]`).
It should return the reconstructed `soundevent.data.Geometry` (typically a
`BoundingBox`).
"""


class RawPrediction(NamedTuple):
    """Intermediate representation of a single detected sound event.

    Holds extracted information about a detection after initial processing
    (like peak finding, coordinate remapping, geometry recovery) but before
    final class decoding and conversion into a `SoundEventPrediction`. This
    can be useful for evaluation or simpler data handling formats.

    Attributes
    ----------
    start_time : float
        Start time of the recovered bounding box in seconds.
    end_time : float
        End time of the recovered bounding box in seconds.
    low_freq : float
        Lowest frequency of the recovered bounding box in Hz.
    high_freq : float
        Highest frequency of the recovered bounding box in Hz.
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

    start_time: float
    end_time: float
    low_freq: float
    high_freq: float
    detection_score: float
    class_scores: xr.DataArray
    features: xr.DataArray


class PostprocessorProtocol(Protocol):
    """Protocol defining the interface for the full postprocessing pipeline.

    This protocol outlines the standard methods for an object that takes raw
    output from a BatDetect2 model and the corresponding input clip metadata,
    and processes it through various stages (e.g., coordinate remapping, NMS,
    detection extraction, data extraction, decoding) to produce interpretable
    results at different levels of completion.

    Implementations manage the configured logic for all postprocessing steps.
    """

    def get_feature_arrays(
        self,
        output: ModelOutput,
        clips: List[data.Clip],
    ) -> List[xr.DataArray]:
        """Remap feature tensors to coordinate-aware DataArrays.

        Parameters
        ----------
        output : ModelOutput
            The raw output from the neural network model for a batch, expected
            to contain the necessary feature tensors.
        clips : List[data.Clip]
            A list of `soundevent.data.Clip` objects, one for each item in the
            processed batch. This list provides the timing, recording, and
            other metadata context needed to calculate real-world coordinates
            (seconds, Hz) for the output arrays. The length of this list must
            correspond to the batch size of the `output`.

        Returns
        -------
        List[xr.DataArray]
            A list of xarray DataArrays, one for each input clip in the batch,
            in the same order. Each DataArray contains the feature vectors
            with dimensions like ('feature', 'time', 'frequency') and
            corresponding real-world coordinates.
        """
        ...

    def get_detection_arrays(
        self,
        output: ModelOutput,
        clips: List[data.Clip],
    ) -> List[xr.DataArray]:
        """Remap detection tensors to coordinate-aware DataArrays.

        Parameters
        ----------
        output : ModelOutput
            The raw output from the neural network model for a batch,
            containing detection heatmaps.
        clips : List[data.Clip]
            A list of `soundevent.data.Clip` objects corresponding to the batch
            items, providing coordinate context. Must match the batch size of
            `output`.

        Returns
        -------
        List[xr.DataArray]
            A list of 2D xarray DataArrays (one per input clip, in order),
            representing the detection heatmap with 'time' and 'frequency'
            coordinates. Values typically indicate detection confidence.
        """
        ...

    def get_classification_arrays(
        self,
        output: ModelOutput,
        clips: List[data.Clip],
    ) -> List[xr.DataArray]:
        """Remap classification tensors to coordinate-aware DataArrays.

        Parameters
        ----------
        output : ModelOutput
            The raw output from the neural network model for a batch,
            containing class probability tensors.
        clips : List[data.Clip]
            A list of `soundevent.data.Clip` objects corresponding to the batch
            items, providing coordinate context. Must match the batch size of
            `output`.

        Returns
        -------
        List[xr.DataArray]
            A list of 3D xarray DataArrays (one per input clip, in order),
            representing class probabilities with 'category', 'time', and
            'frequency' dimensions and coordinates.
        """
        ...

    def get_sizes_arrays(
        self,
        output: ModelOutput,
        clips: List[data.Clip],
    ) -> List[xr.DataArray]:
        """Remap size prediction tensors to coordinate-aware DataArrays.

        Parameters
        ----------
        output : ModelOutput
            The raw output from the neural network model for a batch,
            containing predicted size tensors (e.g., width and height).
        clips : List[data.Clip]
            A list of `soundevent.data.Clip` objects corresponding to the batch
            items, providing coordinate context. Must match the batch size of
            `output`.

        Returns
        -------
        List[xr.DataArray]
            A list of 3D xarray DataArrays (one per input clip, in order),
            representing predicted sizes with 'dimension'
            (e.g., ['width', 'height']), 'time', and 'frequency' dimensions and
            coordinates. Values represent estimated detection sizes.
        """
        ...

    def get_detection_datasets(
        self,
        output: ModelOutput,
        clips: List[data.Clip],
    ) -> List[xr.Dataset]:
        """Perform remapping, NMS, detection, and data extraction for a batch.

        Processes the raw model output for a batch to identify detection peaks
        and extract all associated information (score, position, size, class
        probs, features) at those peak locations, returning a structured
        dataset for each input clip in the batch.

        Parameters
        ----------
        output : ModelOutput
            The raw output from the neural network model for a batch.
        clips : List[data.Clip]
            A list of `soundevent.data.Clip` objects corresponding to the batch
            items, providing context. Must match the batch size of `output`.

        Returns
        -------
        List[xr.Dataset]
            A list of xarray Datasets (one per input clip, in order). Each
            Dataset contains multiple DataArrays ('scores', 'dimensions',
            'classes', 'features') sharing a common 'detection' dimension,
            providing aligned data for each detected event in that clip.
        """
        ...

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
