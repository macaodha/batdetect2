"""Post-processing of the output of the model."""
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import nn

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict

np.seterr(divide="ignore", invalid="ignore")


def x_coords_to_time(
    x_pos: float,
    sampling_rate: int,
    fft_win_length: float,
    fft_overlap: float,
) -> float:
    """Convert x coordinates of spectrogram to time in seconds.

    Args:
        x_pos: X position of the detection in pixels.
        sampling_rate: Sampling rate of the audio in Hz.
        fft_win_length: Length of the FFT window in seconds.
        fft_overlap: Overlap of the FFT windows in seconds.

    Returns:
        Time in seconds.
    """
    nfft = int(fft_win_length * sampling_rate)
    noverlap = int(fft_overlap * nfft)
    return ((x_pos * (nfft - noverlap)) + noverlap) / sampling_rate


def overall_class_pred(det_prob, class_prob):
    weighted_pred = (class_prob * det_prob).sum(1)
    return weighted_pred / weighted_pred.sum()


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


class ModelOutputs(TypedDict):
    """Outputs of the model."""

    pred_det: torch.Tensor
    """Detection probabilities."""

    pred_size: torch.Tensor
    """Box sizes."""

    pred_class: Optional[torch.Tensor]
    """Class probabilities."""

    features: Optional[torch.Tensor]
    """Features extracted by the model."""


def run_nms(
    outputs: ModelOutputs,
    params: NonMaximumSuppressionConfig,
    sampling_rate: np.ndarray,
) -> Tuple[List[PredictionResults], List[np.ndarray]]:
    """Run non-maximum suppression on the output of the model.

    Model outputs processed are expected to have a batch dimension.
    Each element of the batch is processed independently. The
    result is a pair of lists, one for the predictions and one for
    the features. Each element of the lists corresponds to one
    element of the batch.
    """

    pred_det = outputs["pred_det"]  # probability of box
    pred_size = outputs["pred_size"]  # box size

    pred_det_nms = non_max_suppression(pred_det, params["nms_kernel_size"])
    freq_rescale = (params["max_freq"] - params["min_freq"]) / pred_det.shape[
        -2
    ]

    # NOTE there will be small differences depending on which sampling rate is chosen
    # as we are choosing the same sampling rate for the entire batch
    duration = x_coords_to_time(
        pred_det.shape[-1],
        int(sampling_rate[0].item()),
        params["fft_win_length"],
        params["fft_overlap"],
    )
    top_k = int(duration * params["nms_top_k_per_sec"])
    scores, y_pos, x_pos = get_topk_scores(pred_det_nms, top_k)

    # loop over batch to save outputs
    preds: List[PredictionResults] = []
    feats: List[np.ndarray] = []
    for num_detection in range(pred_det_nms.shape[0]):
        # get valid indices
        inds_ord = torch.argsort(x_pos[num_detection, :])
        valid_inds = (
            scores[num_detection, inds_ord] > params["detection_threshold"]
        )
        valid_inds = inds_ord[valid_inds]

        # create result dictionary
        pred = {}
        pred["det_probs"] = scores[num_detection, valid_inds]
        pred["x_pos"] = x_pos[num_detection, valid_inds]
        pred["y_pos"] = y_pos[num_detection, valid_inds]
        pred["bb_width"] = pred_size[
            num_detection, 0, pred["y_pos"], pred["x_pos"]
        ]
        pred["bb_height"] = pred_size[
            num_detection, 1, pred["y_pos"], pred["x_pos"]
        ]
        pred["start_times"] = x_coords_to_time(
            pred["x_pos"].float() / params["resize_factor"],
            int(sampling_rate[num_detection].item()),
            params["fft_win_length"],
            params["fft_overlap"],
        )
        pred["end_times"] = x_coords_to_time(
            (pred["x_pos"].float() + pred["bb_width"])
            / params["resize_factor"],
            int(sampling_rate[num_detection].item()),
            params["fft_win_length"],
            params["fft_overlap"],
        )
        pred["low_freqs"] = (
            pred_size[num_detection].shape[1] - pred["y_pos"].float()
        ) * freq_rescale + params["min_freq"]
        pred["high_freqs"] = (
            pred["low_freqs"] + pred["bb_height"] * freq_rescale
        )

        # extract the per class votes
        pred_class = outputs.get("pred_class")
        if pred_class is not None:
            pred["class_probs"] = pred_class[
                num_detection,
                :,
                y_pos[num_detection, valid_inds],
                x_pos[num_detection, valid_inds],
            ]

        # extract the model features
        features = outputs.get("features")
        if features is not None:
            feat = features[
                num_detection,
                :,
                y_pos[num_detection, valid_inds],
                x_pos[num_detection, valid_inds],
            ].transpose(0, 1)
            feat = feat.cpu().numpy().astype(np.float32)
            feats.append(feat)

        # convert to numpy
        for key, value in pred.items():
            pred[key] = value.cpu().numpy().astype(np.float32)

        preds.append(pred)

    return preds, feats


def non_max_suppression(heat, kernel_size):
    # kernel can be an int or list/tuple
    if isinstance(kernel_size, int):
        kernel_size_h = kernel_size
        kernel_size_w = kernel_size

    pad_h = (kernel_size_h - 1) // 2
    pad_w = (kernel_size_w - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel_size_h, kernel_size_w), stride=1, padding=(pad_h, pad_w)
    )
    keep = (hmax == heat).float()

    return heat * keep


def get_topk_scores(scores, K):
    # expects input of size:  batch x 1 x height x width
    batch, _, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), K)
    topk_inds = topk_inds % (height * width)
    topk_ys = torch.div(topk_inds, width, rounding_mode="floor").long()
    topk_xs = (topk_inds % width).long()

    return topk_scores, topk_ys, topk_xs
