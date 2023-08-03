"""Functions to compute features from predictions."""
from typing import Dict, Optional

import numpy as np

from batdetect2 import types
from batdetect2.detector.parameters import MAX_FREQ_HZ, MIN_FREQ_HZ


def convert_int_to_freq(spec_ind, spec_height, min_freq, max_freq):
    """Convert spectrogram index to frequency in Hz.""" ""
    spec_ind = spec_height - spec_ind
    return round(
        (spec_ind / float(spec_height)) * (max_freq - min_freq) + min_freq, 2
    )


def extract_spec_slices(spec, pred_nms):
    """Extract spectrogram slices from spectrogram.

    The slices are extracted based on detected call locations.
    """
    x_pos = pred_nms["x_pos"]
    bb_width = pred_nms["bb_width"]
    slices = []

    # add 20% padding either side of call
    pad = bb_width * 0.2
    x_pos_pad = x_pos - pad
    bb_width_pad = bb_width + 2 * pad

    for ff in range(len(pred_nms["det_probs"])):
        x_start = int(np.maximum(0, x_pos_pad[ff]))
        x_end = int(
            np.minimum(
                spec.shape[1] - 1, np.round(x_pos_pad[ff] + bb_width_pad[ff])
            )
        )
        slices.append(spec[:, x_start:x_end].astype(np.float16))
    return slices


def compute_duration(
    prediction: types.Prediction,
    **_,
) -> float:
    """Compute duration of call in seconds."""
    return round(prediction["end_time"] - prediction["start_time"], 5)


def compute_low_freq(
    prediction: types.Prediction,
    **_,
) -> float:
    """Compute lowest frequency in call in Hz."""
    return int(prediction["low_freq"])


def compute_high_freq(
    prediction: types.Prediction,
    **_,
) -> float:
    """Compute highest frequency in call in Hz."""
    return int(prediction["high_freq"])


def compute_bandwidth(
    prediction: types.Prediction,
    **_,
) -> float:
    """Compute bandwidth of call in Hz."""
    return int(prediction["high_freq"] - prediction["low_freq"])


def compute_max_power_bb(
    prediction: types.Prediction,
    spec: Optional[np.ndarray] = None,
    min_freq: int = MIN_FREQ_HZ,
    max_freq: int = MAX_FREQ_HZ,
    **_,
) -> float:
    """Compute frequency with maximum power in call in Hz.

    This is the frequency with the maximum power in the bounding box of the
    call.
    """
    if spec is None:
        return np.nan

    x_start = max(0, prediction["x_pos"])
    x_end = min(
        spec.shape[1] - 1, prediction["x_pos"] + prediction["bb_width"]
    )

    # y low is the lowest freq but it will have a higher value due to array
    # starting at 0 at top
    y_low = min(spec.shape[0] - 1, prediction["y_pos"])
    y_high = max(0, prediction["y_pos"] - prediction["bb_height"])

    spec_bb = spec[y_high:y_low, x_start:x_end]
    power_per_freq_band = np.sum(spec_bb, axis=1)

    try:
        max_power_ind = np.argmax(power_per_freq_band)
    except ValueError:
        # If the call is too short, the bounding box might be empty.
        # In this case, return NaN.
        return np.nan

    return int(
        convert_int_to_freq(
            y_high + max_power_ind,
            spec.shape[0],
            min_freq,
            max_freq,
        )
    )


def compute_max_power(
    prediction: types.Prediction,
    spec: Optional[np.ndarray] = None,
    min_freq: int = MIN_FREQ_HZ,
    max_freq: int = MAX_FREQ_HZ,
    **_,
) -> float:
    """Compute frequency with maximum power in during the call in Hz."""
    if spec is None:
        return np.nan

    x_start = max(0, prediction["x_pos"])
    x_end = min(
        spec.shape[1] - 1, prediction["x_pos"] + prediction["bb_width"]
    )
    spec_call = spec[:, x_start:x_end]
    power_per_freq_band = np.sum(spec_call, axis=1)
    max_power_ind = np.argmax(power_per_freq_band)
    return int(
        convert_int_to_freq(
            max_power_ind,
            spec.shape[0],
            min_freq,
            max_freq,
        )
    )


def compute_max_power_first(
    prediction: types.Prediction,
    spec: Optional[np.ndarray] = None,
    min_freq: int = MIN_FREQ_HZ,
    max_freq: int = MAX_FREQ_HZ,
    **_,
) -> float:
    """Compute frequency with maximum power in first half of call in Hz."""
    if spec is None:
        return np.nan

    x_start = max(0, prediction["x_pos"])
    x_end = min(
        spec.shape[1] - 1, prediction["x_pos"] + prediction["bb_width"]
    )
    spec_call = spec[:, x_start:x_end]
    first_half = spec_call[:, : int(spec_call.shape[1] / 2)]
    power_per_freq_band = np.sum(first_half, axis=1)
    max_power_ind = np.argmax(power_per_freq_band)
    return int(
        convert_int_to_freq(
            max_power_ind,
            spec.shape[0],
            min_freq,
            max_freq,
        )
    )


def compute_max_power_second(
    prediction: types.Prediction,
    spec: Optional[np.ndarray] = None,
    min_freq: int = MIN_FREQ_HZ,
    max_freq: int = MAX_FREQ_HZ,
    **_,
) -> float:
    """Compute frequency with maximum power in second half of call in Hz."""
    if spec is None:
        return np.nan

    x_start = max(0, prediction["x_pos"])
    x_end = min(
        spec.shape[1] - 1, prediction["x_pos"] + prediction["bb_width"]
    )
    spec_call = spec[:, x_start:x_end]
    second_half = spec_call[:, int(spec_call.shape[1] / 2) :]
    power_per_freq_band = np.sum(second_half, axis=1)
    max_power_ind = np.argmax(power_per_freq_band)
    return int(
        convert_int_to_freq(
            max_power_ind,
            spec.shape[0],
            min_freq,
            max_freq,
        )
    )


def compute_call_interval(
    prediction: types.Prediction,
    previous: Optional[types.Prediction] = None,
    **_,
) -> float:
    """Compute time between this call and the previous call in seconds."""
    if previous is None:
        return np.nan
    return round(prediction["start_time"] - previous["end_time"], 5)


# NOTE: The order of the features in this dictionary is important. The
# features are extracted in this order and the order of the columns in the
# output csv file is determined by this order. In order to avoid breaking
# changes in the output csv file, new features should be added to the end of
# this dictionary.
FEATURES: Dict[str, types.FeatureExtractor] = {
    "duration": compute_duration,
    "low_freq_bb": compute_low_freq,
    "high_freq_bb": compute_high_freq,
    "bandwidth": compute_bandwidth,
    "max_power_bb": compute_max_power_bb,
    "max_power": compute_max_power,
    "max_power_first": compute_max_power_first,
    "max_power_second": compute_max_power_second,
    "call_interval": compute_call_interval,
}


def get_feats(
    spec: np.ndarray,
    pred_nms: types.PredictionResults,
    params: types.FeatureExtractionParameters,
):
    """Extract features from spectrogram based on detected call locations.

    The features extracted are:

    - duration: duration of call in seconds
    - low_freq: lowest frequency in call in kHz
    - high_freq: highest frequency in call in kHz
    - bandwidth: high_freq - low_freq
    - max_power_bb: frequency with maximum power in call in kHz
    - max_power: frequency with maximum power in spectrogram in kHz
    - max_power_first: frequency with maximum power in first half of call in
    kHz.
    - max_power_second: frequency with maximum power in second half of call in
    kHz.
    - call_interval: time between this call and the previous call in seconds

    Consider re-extracting spectrogram for this to get better temporal
    resolution.

    For more possible features check out:
    https://github.com/YvesBas/Tadarida-D/blob/master/Manual_Tadarida-D.odt

    Parameters
    ----------
    spec : np.ndarray
        Spectrogram from which to extract features.

    pred_nms : types.PredictionResults
        Information about detected calls from which to extract features.

    params : types.FeatureExtractionParameters
        Parameters for feature extraction.

    Returns
    -------
    features : np.ndarray
        Extracted features for each detected call. Shape is
        (num_detections, num_features).
    """
    num_detections = len(pred_nms["det_probs"])
    features = np.empty((num_detections, len(FEATURES)), dtype=np.float32)
    previous = None

    for row in range(num_detections):
        prediction: types.Prediction = {
            "det_prob": float(pred_nms["det_probs"][row]),
            "class_prob": pred_nms["class_probs"][:, row],
            "start_time": float(pred_nms["start_times"][row]),
            "end_time": float(pred_nms["end_times"][row]),
            "low_freq": float(pred_nms["low_freqs"][row]),
            "high_freq": float(pred_nms["high_freqs"][row]),
            "x_pos": int(pred_nms["x_pos"][row]),
            "y_pos": int(pred_nms["y_pos"][row]),
            "bb_width": int(pred_nms["bb_width"][row]),
            "bb_height": int(pred_nms["bb_height"][row]),
        }

        for col, feature in enumerate(FEATURES.values()):
            features[row, col] = feature(
                prediction,
                previous=previous,
                spec=spec,
                **params,
            )

        previous = prediction

    return features


def get_feature_names():
    """Get names of features in the order they are extracted."""
    return list(FEATURES.keys())
