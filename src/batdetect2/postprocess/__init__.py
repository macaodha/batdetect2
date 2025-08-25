"""Main entry point for the BatDetect2 Postprocessing pipeline."""

from typing import List, Optional

from loguru import logger
from pydantic import Field
from soundevent import data

from batdetect2.configs import BaseConfig, load_config
from batdetect2.postprocess.decoding import (
    DEFAULT_CLASSIFICATION_THRESHOLD,
    convert_detections_to_raw_predictions,
    convert_raw_prediction_to_sound_event_prediction,
    convert_raw_predictions_to_clip_prediction,
)
from batdetect2.postprocess.extraction import extract_prediction_tensor
from batdetect2.postprocess.nms import (
    NMS_KERNEL_SIZE,
    non_max_suppression,
)
from batdetect2.postprocess.remapping import map_detection_to_clip
from batdetect2.preprocess import MAX_FREQ, MIN_FREQ
from batdetect2.typing import ModelOutput, PreprocessorProtocol, TargetProtocol
from batdetect2.typing.postprocess import (
    BatDetect2Prediction,
    Detections,
    PostprocessorProtocol,
    RawPrediction,
)

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
    "convert_detections_to_raw_predictions",
    "load_postprocess_config",
    "non_max_suppression",
]

DEFAULT_DETECTION_THRESHOLD = 0.01


TOP_K_PER_SEC = 200


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
        targets=targets,
        preprocessor=preprocessor,
        config=config,
    )


class Postprocessor(PostprocessorProtocol):
    """Standard implementation of the postprocessing pipeline."""

    targets: TargetProtocol

    preprocessor: PreprocessorProtocol

    def __init__(
        self,
        targets: TargetProtocol,
        preprocessor: PreprocessorProtocol,
        config: PostprocessConfig,
    ):
        """Initialize the Postprocessor."""
        self.targets = targets
        self.preprocessor = preprocessor
        self.config = config

    def get_detections(
        self,
        output: ModelOutput,
        clips: Optional[List[data.Clip]] = None,
    ) -> List[Detections]:
        width = output.detection_probs.shape[-1]
        duration = width / self.preprocessor.output_samplerate
        max_detections = int(self.config.top_k_per_sec * duration)

        detections = extract_prediction_tensor(
            output,
            max_detections=max_detections,
            threshold=self.config.detection_threshold,
        )

        if clips is None:
            return detections

        return [
            map_detection_to_clip(
                detection,
                start_time=clip.start_time,
                end_time=clip.end_time,
                min_freq=self.preprocessor.min_freq,
                max_freq=self.preprocessor.max_freq,
            )
            for detection, clip in zip(detections, clips)
        ]

    def get_raw_predictions(
        self,
        output: ModelOutput,
        clips: List[data.Clip],
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
        detections = self.get_detections(output, clips)
        return [
            convert_detections_to_raw_predictions(
                dataset,
                targets=self.targets,
            )
            for dataset in detections
        ]

    def get_sound_event_predictions(
        self,
        output: ModelOutput,
        clips: List[data.Clip],
    ) -> List[List[BatDetect2Prediction]]:
        raw_predictions = self.get_raw_predictions(output, clips)
        return [
            [
                BatDetect2Prediction(
                    raw=raw,
                    sound_event_prediction=convert_raw_prediction_to_sound_event_prediction(
                        raw,
                        recording=clip.recording,
                        targets=self.targets,
                        classification_threshold=self.config.classification_threshold,
                    ),
                )
                for raw in predictions
            ]
            for predictions, clip in zip(raw_predictions, clips)
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
                targets=self.targets,
                classification_threshold=self.config.classification_threshold,
            )
            for prediction, clip in zip(raw_predictions, clips)
        ]
