from pydantic import Field

from batdetect2.core.configs import BaseConfig
from batdetect2.postprocess.nms import NMS_KERNEL_SIZE

__all__ = [
    "PostprocessConfig",
]

DEFAULT_DETECTION_THRESHOLD = 0.01
DEFAULT_CLASSIFICATION_THRESHOLD = 0.1


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
