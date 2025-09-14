"""Main entry point for the BatDetect2 Postprocessing pipeline."""

from typing import List, Optional

import torch
from loguru import logger
from pydantic import Field
from soundevent import data

from batdetect2.configs import BaseConfig, load_config
from batdetect2.postprocess.decoding import (
    DEFAULT_CLASSIFICATION_THRESHOLD,
    convert_raw_prediction_to_sound_event_prediction,
    convert_raw_predictions_to_clip_prediction,
    to_raw_predictions,
)
from batdetect2.postprocess.extraction import extract_prediction_tensor
from batdetect2.postprocess.nms import (
    NMS_KERNEL_SIZE,
    non_max_suppression,
)
from batdetect2.postprocess.remapping import map_detection_to_clip
from batdetect2.preprocess import MAX_FREQ, MIN_FREQ
from batdetect2.typing import ModelOutput
from batdetect2.typing.postprocess import (
    BatDetect2Prediction,
    DetectionsTensor,
    PostprocessorProtocol,
    RawPrediction,
)
from batdetect2.typing.preprocess import PreprocessorProtocol
from batdetect2.typing.targets import TargetProtocol

__all__ = [
    "DEFAULT_CLASSIFICATION_THRESHOLD",
    "DEFAULT_DETECTION_THRESHOLD",
    "MAX_FREQ",
    "MIN_FREQ",
    "ModelOutput",
    "NMS_KERNEL_SIZE",
    "PostprocessConfig",
    "Postprocessor",
    "TOP_K_PER_SEC",
    "build_postprocessor",
    "convert_raw_predictions_to_clip_prediction",
    "to_raw_predictions",
    "load_postprocess_config",
    "non_max_suppression",
]

DEFAULT_DETECTION_THRESHOLD = 0.01


TOP_K_PER_SEC = 100


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
    preprocessor: PreprocessorProtocol,
    config: Optional[PostprocessConfig] = None,
) -> PostprocessorProtocol:
    """Factory function to build the standard postprocessor."""
    config = config or PostprocessConfig()
    logger.opt(lazy=True).debug(
        "Building postprocessor with config: \n{}",
        lambda: config.to_yaml_string(),
    )
    return Postprocessor(
        samplerate=preprocessor.output_samplerate,
        min_freq=preprocessor.min_freq,
        max_freq=preprocessor.max_freq,
        top_k_per_sec=config.top_k_per_sec,
        detection_threshold=config.detection_threshold,
    )


class Postprocessor(torch.nn.Module, PostprocessorProtocol):
    """Standard implementation of the postprocessing pipeline."""

    def __init__(
        self,
        samplerate: float,
        min_freq: float,
        max_freq: float,
        top_k_per_sec: int = 200,
        detection_threshold: float = 0.01,
    ):
        """Initialize the Postprocessor."""
        super().__init__()
        self.samplerate = samplerate
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.top_k_per_sec = top_k_per_sec
        self.detection_threshold = detection_threshold

    def forward(self, output: ModelOutput) -> List[DetectionsTensor]:
        width = output.detection_probs.shape[-1]
        duration = width / self.samplerate
        max_detections = int(self.top_k_per_sec * duration)
        detections = extract_prediction_tensor(
            output,
            max_detections=max_detections,
            threshold=self.detection_threshold,
        )
        return [
            map_detection_to_clip(
                detection,
                start_time=0,
                end_time=duration,
                min_freq=self.min_freq,
                max_freq=self.max_freq,
            )
            for detection in detections
        ]

    def get_detections(
        self,
        output: ModelOutput,
        start_times: Optional[List[float]] = None,
    ) -> List[DetectionsTensor]:
        width = output.detection_probs.shape[-1]
        duration = width / self.samplerate
        max_detections = int(self.top_k_per_sec * duration)

        detections = extract_prediction_tensor(
            output,
            max_detections=max_detections,
            threshold=self.detection_threshold,
        )

        if start_times is None:
            return detections

        width = output.detection_probs.shape[-1]
        duration = width / self.samplerate
        return [
            map_detection_to_clip(
                detection,
                start_time=start_time,
                end_time=start_time + duration,
                min_freq=self.min_freq,
                max_freq=self.max_freq,
            )
            for detection, start_time in zip(detections, start_times)
        ]


def get_raw_predictions(
    output: ModelOutput,
    targets: TargetProtocol,
    postprocessor: PostprocessorProtocol,
    start_times: Optional[List[float]] = None,
) -> List[List[RawPrediction]]:
    """Extract intermediate RawPrediction objects for a batch."""
    detections = postprocessor.get_detections(output, start_times)
    return [
        to_raw_predictions(detection.numpy(), targets=targets)
        for detection in detections
    ]


def get_sound_event_predictions(
    output: ModelOutput,
    targets: TargetProtocol,
    postprocessor: PostprocessorProtocol,
    clips: List[data.Clip],
    classification_threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD,
) -> List[List[BatDetect2Prediction]]:
    raw_predictions = get_raw_predictions(
        output,
        targets=targets,
        postprocessor=postprocessor,
        start_times=[clip.start_time for clip in clips],
    )
    return [
        [
            BatDetect2Prediction(
                raw=raw,
                sound_event_prediction=convert_raw_prediction_to_sound_event_prediction(
                    raw,
                    recording=clip.recording,
                    targets=targets,
                    classification_threshold=classification_threshold,
                ),
            )
            for raw in predictions
        ]
        for predictions, clip in zip(raw_predictions, clips)
    ]


def get_predictions(
    output: ModelOutput,
    clips: List[data.Clip],
    targets: TargetProtocol,
    postprocessor: PostprocessorProtocol,
    classification_threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD,
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
    raw_predictions = get_raw_predictions(
        output,
        targets=targets,
        postprocessor=postprocessor,
        start_times=[clip.start_time for clip in clips],
    )
    return [
        convert_raw_predictions_to_clip_prediction(
            prediction,
            clip,
            targets=targets,
            classification_threshold=classification_threshold,
        )
        for prediction, clip in zip(raw_predictions, clips)
    ]
