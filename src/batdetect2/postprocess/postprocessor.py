from typing import List, Optional

import torch
from loguru import logger
from soundevent import data

from batdetect2.postprocess.config import (
    PostprocessConfig,
)
from batdetect2.postprocess.decoding import (
    DEFAULT_CLASSIFICATION_THRESHOLD,
    convert_raw_prediction_to_sound_event_prediction,
    convert_raw_predictions_to_clip_prediction,
    to_raw_predictions,
)
from batdetect2.postprocess.extraction import extract_prediction_tensor
from batdetect2.postprocess.remapping import map_detection_to_clip
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
    "build_postprocessor",
    "Postprocessor",
]


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
