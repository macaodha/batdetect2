from typing import List, Optional, Tuple, Union

import torch
from loguru import logger

from batdetect2.postprocess.config import (
    PostprocessConfig,
)
from batdetect2.postprocess.extraction import extract_detection_peaks
from batdetect2.postprocess.nms import NMS_KERNEL_SIZE, non_max_suppression
from batdetect2.postprocess.remapping import map_detection_to_clip
from batdetect2.typing import ModelOutput
from batdetect2.typing.postprocess import (
    ClipDetectionsTensor,
    PostprocessorProtocol,
)
from batdetect2.typing.preprocess import PreprocessorProtocol

__all__ = [
    "build_postprocessor",
    "Postprocessor",
]


def build_postprocessor(
    preprocessor: PreprocessorProtocol,
    config: PostprocessConfig | None = None,
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
        nms_kernel_size: int | Tuple[int, int] = NMS_KERNEL_SIZE,
    ):
        """Initialize the Postprocessor."""
        super().__init__()

        self.output_samplerate = samplerate
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.top_k_per_sec = top_k_per_sec
        self.detection_threshold = detection_threshold
        self.nms_kernel_size = nms_kernel_size

    def forward(
        self,
        output: ModelOutput,
        start_times: List[float] | None = None,
    ) -> List[ClipDetectionsTensor]:
        detection_heatmap = non_max_suppression(
            output.detection_probs.detach(),
            kernel_size=self.nms_kernel_size,
        )

        width = output.detection_probs.shape[-1]
        duration = width / self.output_samplerate
        max_detections = int(self.top_k_per_sec * duration)
        detections = extract_detection_peaks(
            detection_heatmap,
            size_heatmap=output.size_preds,
            feature_heatmap=output.features,
            classification_heatmap=output.class_probs,
            max_detections=max_detections,
            threshold=self.detection_threshold,
        )

        if start_times is None:
            start_times = [0 for _ in range(len(detections))]

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
