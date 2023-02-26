"""Types used in the code base."""
from typing import List, NamedTuple, Optional

import numpy as np
import torch

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict


try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol


try:
    from typing import NotRequired
except ImportError:
    from typing_extensions import NotRequired


__all__ = [
    "Annotation",
    "DetectionModel",
    "FileAnnotations",
    "ModelOutput",
    "ModelParameters",
    "NonMaximumSuppressionConfig",
    "PredictionResults",
    "ProcessingConfiguration",
    "ResultParams",
    "RunResults",
    "SpectrogramParameters",
]


class SpectrogramParameters(TypedDict):
    """Parameters for generating spectrograms."""

    fft_win_length: float
    """Length of the FFT window in seconds."""

    fft_overlap: float
    """Percentage of overlap between FFT windows."""

    spec_height: int
    """Height of the spectrogram in pixels."""

    resize_factor: float
    """Factor to resize the spectrogram by."""

    spec_divide_factor: int
    """Factor to divide the spectrogram by."""

    max_freq: int
    """Maximum frequency to display in the spectrogram."""

    min_freq: int
    """Minimum frequency to display in the spectrogram."""

    spec_scale: str
    """Scale to use for the spectrogram."""

    denoise_spec_avg: bool
    """Whether to denoise the spectrogram by averaging."""

    max_scale_spec: bool
    """Whether to scale the spectrogram so that its max is 1."""


class ModelParameters(TypedDict):
    """Model parameters."""

    model_name: str
    """Model name."""

    num_filters: int
    """Number of filters."""

    emb_dim: int
    """Embedding dimension."""

    ip_height: int
    """Input height in pixels."""

    resize_factor: float
    """Resize factor."""

    class_names: List[str]
    """Class names. The model is trained to detect these classes."""


DictWithClass = TypedDict("DictWithClass", {"class": str})


class Annotation(DictWithClass):
    """Format of annotations.

    This is the format of a single annotation as  expected by the annotation
    tool.
    """

    start_time: float
    """Start time in seconds."""

    end_time: float
    """End time in seconds."""

    low_freq: int
    """Low frequency in Hz."""

    high_freq: int
    """High frequency in Hz."""

    class_prob: float
    """Probability of class assignment."""

    det_prob: float
    """Probability of detection."""

    individual: str
    """Individual ID."""

    event: str
    """Type of detected event."""


class FileAnnotations(TypedDict):
    """Format of results.

    This is the format of the results expected by the annotation tool.
    """

    id: str
    """File ID."""

    annotated: bool
    """Whether file has been annotated."""

    duration: float
    """Duration of audio file."""

    issues: bool
    """Whether file has issues."""

    time_exp: float
    """Time expansion factor."""

    class_name: str
    """Class predicted at file level"""

    notes: str
    """Notes of file."""

    annotation: List[Annotation]
    """List of annotations."""


class RunResults(TypedDict):
    """Run results."""

    pred_dict: FileAnnotations
    """Predictions in the format expected by the annotation tool."""

    spec_feats: NotRequired[List[np.ndarray]]
    """Spectrogram features."""

    spec_feat_names: NotRequired[List[str]]
    """Spectrogram feature names."""

    cnn_feats: NotRequired[List[np.ndarray]]
    """CNN features."""

    cnn_feat_names: NotRequired[List[str]]
    """CNN feature names."""

    spec_slices: NotRequired[List[np.ndarray]]
    """Spectrogram slices."""


class ResultParams(TypedDict):
    """Result parameters."""

    class_names: List[str]
    """Class names."""

    spec_features: bool
    """Whether to return spectrogram features."""

    cnn_features: bool
    """Whether to return CNN features."""

    spec_slices: bool
    """Whether to return spectrogram slices."""


class ProcessingConfiguration(TypedDict):
    """Parameters for processing audio files."""

    # audio parameters
    target_samp_rate: int
    """Target sampling rate of the audio."""

    fft_win_length: float
    """Length of the FFT window in seconds."""

    fft_overlap: float
    """Length of the FFT window in samples."""

    resize_factor: float
    """Factor to resize the spectrogram by."""

    spec_divide_factor: int
    """Factor to divide the spectrogram by."""

    spec_height: int
    """Height of the spectrogram in pixels."""

    spec_scale: str
    """Scale to use for the spectrogram."""

    denoise_spec_avg: bool
    """Whether to denoise the spectrogram by averaging."""

    max_scale_spec: bool
    """Whether to scale the spectrogram so that its max is 1."""

    scale_raw_audio: bool
    """Whether to scale the raw audio to be between -1 and 1."""

    class_names: List[str]
    """Names of the classes the model can detect."""

    detection_threshold: float
    """Threshold for detection probability."""

    time_expansion: Optional[float]
    """Time expansion factor of the processed recordings."""

    top_n: int
    """Number of top detections to keep."""

    return_raw_preds: bool
    """Whether to return raw predictions."""

    max_duration: Optional[float]
    """Maximum duration of audio file to process in seconds."""

    nms_kernel_size: int
    """Size of the kernel for non-maximum suppression."""

    max_freq: int
    """Maximum frequency to consider in Hz."""

    min_freq: int
    """Minimum frequency to consider in Hz."""

    nms_top_k_per_sec: float
    """Number of top detections to keep per second."""

    quiet: bool
    """Whether to suppress output."""

    chunk_size: float
    """Size of chunks to process in seconds."""

    cnn_features: bool
    """Whether to return CNN features."""

    spec_features: bool
    """Whether to return spectrogram features."""

    spec_slices: bool
    """Whether to return spectrogram slices."""


class ModelOutput(NamedTuple):
    """Output of the detection model."""

    pred_det: torch.Tensor
    """Tensor with predict detection probabilities."""

    pred_size: torch.Tensor
    """Tensor with predicted bounding box sizes."""

    pred_class: torch.Tensor
    """Tensor with predicted class probabilities."""

    pred_class_un_norm: torch.Tensor
    """Tensor with predicted class probabilities before softmax."""

    features: torch.Tensor
    """Tensor with intermediate features."""


class PredictionResults(TypedDict):
    """Results of the prediction.

    Each key is a list of length `num_detections` containing the
    corresponding values for each detection.
    """

    det_probs: np.ndarray
    """Detection probabilities."""

    x_pos: np.ndarray
    """X position of the detection in pixels."""

    y_pos: np.ndarray
    """Y position of the detection in pixels."""

    bb_width: np.ndarray
    """Width of the detection in pixels."""

    bb_height: np.ndarray
    """Height of the detection in pixels."""

    start_times: np.ndarray
    """Start times of the detections in seconds."""

    end_times: np.ndarray
    """End times of the detections in seconds."""

    low_freqs: np.ndarray
    """Low frequencies of the detections in Hz."""

    high_freqs: np.ndarray
    """High frequencies of the detections in Hz."""

    class_probs: Optional[np.ndarray]
    """Class probabilities."""


class DetectionModel(Protocol):
    """Protocol for detection models.

    This protocol is used to define the interface for the detection models.
    This allows us to use the same code for training and inference, even
    though the models are different.
    """

    num_classes: int
    """Number of classes the model can classify."""

    emb_dim: int
    """Dimension of the embedding vector."""

    num_filts: int
    """Number of filters in the model."""

    resize_factor: float
    """Factor by which the input is resized."""

    ip_height_rs: int
    """Height of the input image."""

    def forward(
        self,
        ip: torch.Tensor,
        return_feats: bool = False,
    ) -> ModelOutput:
        """Forward pass of the model."""
        ...

    def __call__(
        self,
        ip: torch.Tensor,
        return_feats: bool = False,
    ) -> ModelOutput:
        """Forward pass of the model."""
        ...


class NonMaximumSuppressionConfig(TypedDict):
    """Configuration for non-maximum suppression."""

    nms_kernel_size: int
    """Size of the kernel for non-maximum suppression."""

    max_freq: int
    """Maximum frequency to consider in Hz."""

    min_freq: int
    """Minimum frequency to consider in Hz."""

    fft_win_length: float
    """Length of the FFT window in seconds."""

    fft_overlap: float
    """Overlap of the FFT windows in seconds."""

    resize_factor: float
    """Factor by which the input was resized."""

    nms_top_k_per_sec: float
    """Number of top detections to keep per second."""

    detection_threshold: float
    """Threshold for detection probability."""
