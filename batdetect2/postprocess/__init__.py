"""Main entry point for the BatDetect2 Postprocessing pipeline.

This package (`batdetect2.postprocess`) takes the raw outputs from a trained
BatDetect2 neural network model and transforms them into meaningful, structured
predictions, typically in the form of `soundevent.data.ClipPrediction` objects
containing detected sound events with associated class tags and geometry.

The pipeline involves several configurable steps, implemented in submodules:
1.  Non-Maximum Suppression (`.nms`): Isolates distinct detection peaks.
2.  Coordinate Remapping (`.remapping`): Adds real-world time/frequency
    coordinates to raw model output arrays.
3.  Detection Extraction (`.detection`): Identifies candidate detection points
    (location and score) based on thresholds and score ranking (top-k).
4.  Data Extraction (`.extraction`): Gathers associated model outputs (size,
    class probabilities, features) at the detected locations.
5.  Decoding & Formatting (`.decoding`): Converts extracted numerical data and
    class predictions into interpretable `soundevent` objects, including
    recovering geometry (ROIs) and decoding class names back to standard tags.

This module provides the primary interface:
- `PostprocessConfig`: A configuration object for postprocessing parameters
  (thresholds, NMS kernel size, etc.).
- `load_postprocess_config`: Function to load the configuration from a file.
- `Postprocessor`: The main class (implementing `PostprocessorProtocol`) that
  holds the configured pipeline logic.
- `build_postprocessor`: A factory function to create a `Postprocessor`
  instance, linking it to the necessary target definitions (`TargetProtocol`).
It also re-exports key components from submodules for convenience.
"""

from typing import List, Optional

import xarray as xr
from pydantic import Field
from soundevent import data

from batdetect2.configs import BaseConfig, load_config
from batdetect2.models.types import ModelOutput
from batdetect2.postprocess.decoding import (
    DEFAULT_CLASSIFICATION_THRESHOLD,
    convert_raw_predictions_to_clip_prediction,
    convert_xr_dataset_to_raw_prediction,
)
from batdetect2.postprocess.detection import (
    DEFAULT_DETECTION_THRESHOLD,
    TOP_K_PER_SEC,
    extract_detections_from_array,
    get_max_detections,
)
from batdetect2.postprocess.extraction import (
    extract_detection_xr_dataset,
)
from batdetect2.postprocess.nms import (
    NMS_KERNEL_SIZE,
    non_max_suppression,
)
from batdetect2.postprocess.remapping import (
    classification_to_xarray,
    detection_to_xarray,
    features_to_xarray,
    sizes_to_xarray,
)
from batdetect2.postprocess.types import PostprocessorProtocol, RawPrediction
from batdetect2.preprocess import MAX_FREQ, MIN_FREQ
from batdetect2.targets.types import TargetProtocol

__all__ = [
    "DEFAULT_CLASSIFICATION_THRESHOLD",
    "DEFAULT_DETECTION_THRESHOLD",
    "MAX_FREQ",
    "MIN_FREQ",
    "ModelOutput",
    "NMS_KERNEL_SIZE",
    "PostprocessConfig",
    "Postprocessor",
    "PostprocessorProtocol",
    "RawPrediction",
    "TOP_K_PER_SEC",
    "build_postprocessor",
    "classification_to_xarray",
    "convert_raw_predictions_to_clip_prediction",
    "convert_xr_dataset_to_raw_prediction",
    "detection_to_xarray",
    "extract_detection_xr_dataset",
    "extract_detections_from_array",
    "features_to_xarray",
    "get_max_detections",
    "load_postprocess_config",
    "non_max_suppression",
    "sizes_to_xarray",
]


class PostprocessConfig(BaseConfig):
    """Configuration settings for the postprocessing pipeline.

    Defines tunable parameters that control how raw model outputs are
    converted into final detections.

    Attributes
    ----------
    nms_kernel_size : int, default=NMS_KERNEL_SIZE
        Size (pixels) of the kernel/neighborhood for Non-Maximum Suppression.
        Used to suppress weaker detections near stronger peaks. Must be
        positive.
    detection_threshold : float, default=DEFAULT_DETECTION_THRESHOLD
        Minimum confidence score from the detection heatmap required to
        consider a point as a potential detection. Must be >= 0.
    classification_threshold : float, default=DEFAULT_CLASSIFICATION_THRESHOLD
        Minimum confidence score for a specific class prediction to be included
        in the decoded tags for a detection. Must be >= 0.
    top_k_per_sec : int, default=TOP_K_PER_SEC
        Desired maximum number of detections per second of audio. Used by
        `get_max_detections` to calculate an absolute limit based on clip
        duration before applying `extract_detections_from_array`. Must be
        positive.
    """

    nms_kernel_size: int = Field(default=NMS_KERNEL_SIZE, gt=0)
    detection_threshold: float = Field(
        default=DEFAULT_DETECTION_THRESHOLD,
        ge=0,
    )
    classification_threshold: float = Field(
        default=DEFAULT_CLASSIFICATION_THRESHOLD,
        ge=0,
    )
    top_k_per_sec: int = Field(default=TOP_K_PER_SEC, gt=0)


def load_postprocess_config(
    path: data.PathLike,
    field: Optional[str] = None,
) -> PostprocessConfig:
    """Load the postprocessing configuration from a file.

    Reads a configuration file (YAML) and validates it against the
    `PostprocessConfig` schema, potentially extracting data from a nested
    field.

    Parameters
    ----------
    path : PathLike
        Path to the configuration file.
    field : str, optional
        Dot-separated path to a nested section within the file containing the
        postprocessing configuration (e.g., "inference.postprocessing").
        If None, the entire file content is used.

    Returns
    -------
    PostprocessConfig
        The loaded and validated postprocessing configuration object.

    Raises
    ------
    FileNotFoundError
        If the config file path does not exist.
    yaml.YAMLError
        If the file content is not valid YAML.
    pydantic.ValidationError
        If the loaded configuration data does not conform to the
        `PostprocessConfig` schema.
    KeyError, TypeError
        If `field` specifies an invalid path within the loaded data.
    """
    return load_config(path, schema=PostprocessConfig, field=field)


def build_postprocessor(
    targets: TargetProtocol,
    config: Optional[PostprocessConfig] = None,
    max_freq: float = MAX_FREQ,
    min_freq: float = MIN_FREQ,
) -> PostprocessorProtocol:
    """Factory function to build the standard postprocessor.

    Creates and initializes the `Postprocessor` instance, providing it with the
    necessary `targets` object and the `PostprocessConfig`.

    Parameters
    ----------
    targets : TargetProtocol
        An initialized object conforming to the `TargetProtocol`, providing
        methods like `.decode()` and `.recover_roi()`, and attributes like
        `.class_names` and `.generic_class_tags`. This links postprocessing
        to the defined target semantics and geometry mappings.
    config : PostprocessConfig, optional
        Configuration object specifying postprocessing parameters (thresholds,
        NMS kernel size, etc.). If None, default settings defined in
        `PostprocessConfig` will be used.
    min_freq : int, default=MIN_FREQ
        The minimum frequency (Hz) corresponding to the frequency axis of the
        model outputs. Required for coordinate remapping. Consider setting via
        `PostprocessConfig` instead for better encapsulation.
    max_freq : int, default=MAX_FREQ
        The maximum frequency (Hz) corresponding to the frequency axis of the
        model outputs. Required for coordinate remapping. Consider setting via
        `PostprocessConfig`.

    Returns
    -------
    PostprocessorProtocol
        An initialized `Postprocessor` instance ready to process model outputs.
    """
    return Postprocessor(
        targets=targets,
        config=config or PostprocessConfig(),
        min_freq=min_freq,
        max_freq=max_freq,
    )


class Postprocessor(PostprocessorProtocol):
    """Standard implementation of the postprocessing pipeline.

    This class orchestrates the steps required to convert raw model outputs
    into interpretable `soundevent` predictions. It uses configured parameters
    and leverages functions from the `batdetect2.postprocess` submodules for
    each stage (NMS, remapping, detection, extraction, decoding).

    It requires a `TargetProtocol` object during initialization to access
    necessary decoding information (class name to tag mapping,
    ROI recovery logic) ensuring consistency with the target definitions used
    during training or specified for inference.

    Instances are typically created using the `build_postprocessor` factory
    function.

    Attributes
    ----------
    targets : TargetProtocol
        The configured target definition object providing decoding and ROI
        recovery.
    config : PostprocessConfig
        Configuration object holding parameters for NMS, thresholds, etc.
    min_freq : float
        Minimum frequency (Hz) assumed for the model output's frequency axis.
    max_freq : float
        Maximum frequency (Hz) assumed for the model output's frequency axis.
    """

    targets: TargetProtocol

    def __init__(
        self,
        targets: TargetProtocol,
        config: PostprocessConfig,
        min_freq: float = MIN_FREQ,
        max_freq: float = MAX_FREQ,
    ):
        """Initialize the Postprocessor.

        Parameters
        ----------
        targets : TargetProtocol
            Initialized target definition object.
        config : PostprocessConfig
            Configuration for postprocessing parameters.
        min_freq : int, default=MIN_FREQ
            Minimum frequency (Hz) for coordinate remapping.
        max_freq : int, default=MAX_FREQ
            Maximum frequency (Hz) for coordinate remapping.
        """
        self.targets = targets
        self.config = config
        self.min_freq = min_freq
        self.max_freq = max_freq

    def get_feature_arrays(
        self,
        output: ModelOutput,
        clips: List[data.Clip],
    ) -> List[xr.DataArray]:
        """Extract and remap raw feature tensors for a batch.

        Parameters
        ----------
        output : ModelOutput
            Raw model output containing `output.features` tensor for the batch.
        clips : List[data.Clip]
            List of Clip objects corresponding to the batch items.

        Returns
        -------
        List[xr.DataArray]
            List of coordinate-aware feature DataArrays, one per clip.

        Raises
        ------
        ValueError
            If batch sizes of `output.features` and `clips` do not match.
        """
        if len(clips) != len(output.features):
            raise ValueError(
                "Number of clips and batch size of feature array"
                "do not match. "
                f"(clips: {len(clips)}, features: {len(output.features)})"
            )

        return [
            features_to_xarray(
                feats,
                start_time=clip.start_time,
                end_time=clip.end_time,
                min_freq=self.min_freq,
                max_freq=self.max_freq,
            )
            for feats, clip in zip(output.features, clips)
        ]

    def get_detection_arrays(
        self,
        output: ModelOutput,
        clips: List[data.Clip],
    ) -> List[xr.DataArray]:
        """Apply NMS and remap detection heatmaps for a batch.

        Parameters
        ----------
        output : ModelOutput
            Raw model output containing `output.detection_probs` tensor for the
            batch.
        clips : List[data.Clip]
            List of Clip objects corresponding to the batch items.

        Returns
        -------
        List[xr.DataArray]
            List of NMS-applied, coordinate-aware detection heatmaps, one per
            clip.

        Raises
        ------
        ValueError
            If batch sizes of `output.detection_probs` and `clips` do not match.
        """
        detections = output.detection_probs

        if len(clips) != len(output.detection_probs):
            raise ValueError(
                "Number of clips and batch size of detection array "
                "do not match. "
                f"(clips: {len(clips)}, detection: {len(detections)})"
            )

        detections = non_max_suppression(
            detections,
            kernel_size=self.config.nms_kernel_size,
        )

        return [
            detection_to_xarray(
                dets,
                start_time=clip.start_time,
                end_time=clip.end_time,
                min_freq=self.min_freq,
                max_freq=self.max_freq,
            )
            for dets, clip in zip(detections, clips)
        ]

    def get_classification_arrays(
        self, output: ModelOutput, clips: List[data.Clip]
    ) -> List[xr.DataArray]:
        """Extract and remap raw classification tensors for a batch.

        Parameters
        ----------
        output : ModelOutput
            Raw model output containing `output.class_probs` tensor for the
            batch.
        clips : List[data.Clip]
            List of Clip objects corresponding to the batch items.

        Returns
        -------
        List[xr.DataArray]
            List of coordinate-aware class probability maps, one per clip.

        Raises
        ------
        ValueError
            If batch sizes of `output.class_probs` and `clips` do not match, or
            if number of classes mismatches `self.targets.class_names`.
        """
        classifications = output.class_probs

        if len(clips) != len(classifications):
            raise ValueError(
                "Number of clips and batch size of classification array "
                "do not match. "
                f"(clips: {len(clips)}, classification: {len(classifications)})"
            )

        return [
            classification_to_xarray(
                class_probs,
                start_time=clip.start_time,
                end_time=clip.end_time,
                class_names=self.targets.class_names,
                min_freq=self.min_freq,
                max_freq=self.max_freq,
            )
            for class_probs, clip in zip(classifications, clips)
        ]

    def get_sizes_arrays(
        self, output: ModelOutput, clips: List[data.Clip]
    ) -> List[xr.DataArray]:
        """Extract and remap raw size prediction tensors for a batch.

        Parameters
        ----------
        output : ModelOutput
            Raw model output containing `output.size_preds` tensor for the
            batch.
        clips : List[data.Clip]
            List of Clip objects corresponding to the batch items.

        Returns
        -------
        List[xr.DataArray]
            List of coordinate-aware size prediction maps, one per clip.

        Raises
        ------
        ValueError
            If batch sizes of `output.size_preds` and `clips` do not match.
        """
        sizes = output.size_preds

        if len(clips) != len(sizes):
            raise ValueError(
                "Number of clips and batch size of sizes array do not match. "
                f"(clips: {len(clips)}, sizes: {len(sizes)})"
            )

        return [
            sizes_to_xarray(
                size_preds,
                start_time=clip.start_time,
                end_time=clip.end_time,
                min_freq=self.min_freq,
                max_freq=self.max_freq,
            )
            for size_preds, clip in zip(sizes, clips)
        ]

    def get_detection_datasets(
        self, output: ModelOutput, clips: List[data.Clip]
    ) -> List[xr.Dataset]:
        """Perform NMS, remapping, detection, and data extraction for a batch.

        Parameters
        ----------
        output : ModelOutput
            Raw output from the neural network model for a batch.
        clips : List[data.Clip]
            List of `soundevent.data.Clip` objects corresponding to the batch.

        Returns
        -------
        List[xr.Dataset]
            List of xarray Datasets (one per clip). Each Dataset contains
            aligned scores, dimensions, class probabilities, and features for
            detections found in that clip.
        """
        detection_arrays = self.get_detection_arrays(output, clips)
        classification_arrays = self.get_classification_arrays(output, clips)
        size_arrays = self.get_sizes_arrays(output, clips)
        features_arrays = self.get_feature_arrays(output, clips)

        datasets = []
        for det_array, class_array, sizes_array, feats_array in zip(
            detection_arrays,
            classification_arrays,
            size_arrays,
            features_arrays,
        ):
            max_detections = get_max_detections(
                det_array,
                top_k_per_sec=self.config.top_k_per_sec,
            )

            positions = extract_detections_from_array(
                det_array,
                max_detections=max_detections,
                threshold=self.config.detection_threshold,
            )

            datasets.append(
                extract_detection_xr_dataset(
                    positions,
                    sizes_array,
                    class_array,
                    feats_array,
                )
            )

        return datasets

    def get_raw_predictions(
        self, output: ModelOutput, clips: List[data.Clip]
    ) -> List[List[RawPrediction]]:
        """Extract intermediate RawPrediction objects for a batch.

        Processes raw model output through remapping, NMS, detection, data
        extraction, and geometry recovery via the configured
        `targets.recover_roi`.

        Parameters
        ----------
        output : ModelOutput
            Raw output from the neural network model for a batch.
        clips : List[data.Clip]
            List of `soundevent.data.Clip` objects corresponding to the batch.

        Returns
        -------
        List[List[RawPrediction]]
            List of lists (one inner list per input clip). Each inner list
            contains `RawPrediction` objects for detections in that clip.
        """
        detection_datasets = self.get_detection_datasets(output, clips)
        return [
            convert_xr_dataset_to_raw_prediction(
                dataset,
                self.targets.recover_roi,
            )
            for dataset in detection_datasets
        ]

    def get_predictions(
        self, output: ModelOutput, clips: List[data.Clip]
    ) -> List[data.ClipPrediction]:
        """Perform the full postprocessing pipeline for a batch.

        Takes raw model output and corresponding clips, applies the entire
        configured chain (NMS, remapping, extraction, geometry recovery, class
        decoding), producing final `soundevent.data.ClipPrediction` objects.

        Parameters
        ----------
        output : ModelOutput
            Raw output from the neural network model for a batch.
        clips : List[data.Clip]
            List of `soundevent.data.Clip` objects corresponding to the batch.

        Returns
        -------
        List[data.ClipPrediction]
            List containing one `ClipPrediction` object for each input clip,
            populated with `SoundEventPrediction` objects.
        """
        raw_predictions = self.get_raw_predictions(output, clips)
        return [
            convert_raw_predictions_to_clip_prediction(
                prediction,
                clip,
                sound_event_decoder=self.targets.decode,
                generic_class_tags=self.targets.generic_class_tags,
                classification_threshold=self.config.classification_threshold,
            )
            for prediction, clip in zip(raw_predictions, clips)
        ]
