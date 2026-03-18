from pydantic import Field
from soundevent import data

from batdetect2.core.configs import BaseConfig, load_config
from batdetect2.postprocess.nms import NMS_KERNEL_SIZE

__all__ = [
    "PostprocessConfig",
    "load_postprocess_config",
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


def load_postprocess_config(
    path: data.PathLike,
    field: str | None = None,
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
