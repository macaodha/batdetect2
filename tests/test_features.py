"""Test suite for feature extraction functions."""

import numpy as np

import batdetect2.detector.compute_features as feats
from batdetect2 import types


def index_to_freq(
    index: int,
    spec_height: int,
    min_freq: int,
    max_freq: int,
) -> float:
    """Convert spectrogram index to frequency in Hz."""
    index = spec_height - index
    return round(
        (index / float(spec_height)) * (max_freq - min_freq) + min_freq, 2
    )


def index_to_time(
    index: int,
    spec_width: int,
    spec_duration: float,
) -> float:
    """Convert spectrogram index to time in seconds."""
    return round((index / float(spec_width)) * spec_duration, 2)


def test_get_feats_function_with_empty_spectrogram():
    spec_duration = 3
    spec_width = 100
    spec_height = 100
    min_freq = 10_000
    max_freq = 120_000
    spectrogram = np.zeros((spec_height, spec_width))

    x_pos = 20
    y_pos = 80
    bb_width = 20
    bb_height = 20

    start_time = index_to_time(x_pos, spec_width, spec_duration)
    end_time = index_to_time(x_pos + bb_width, spec_width, spec_duration)
    high_freq = index_to_freq(y_pos, spec_height, min_freq, max_freq)
    low_freq = index_to_freq(y_pos + bb_height, spec_height, min_freq, max_freq)

    pred_nms: types.PredictionResults = {
        "det_probs": np.array([1]),
        "class_probs": np.array([1]),
        "x_pos": np.array([x_pos]),
        "y_pos": np.array([y_pos]),
        "bb_width": np.array([bb_width]),
        "bb_height": np.array([bb_height]),
        "start_times": np.array([start_time]),
        "end_times": np.array([end_time]),
        "low_freqs": np.array([low_freq]),
        "high_freqs": np.array([high_freq]),
    }

    params: types.FeatureExtractionParameters = {
        "min_freq": min_freq,
        "max_freq": max_freq,
    }

    features = feats.get_feats(spectrogram, pred_nms, params)
    assert low_freq < high_freq
    assert isinstance(features, np.ndarray)
    assert features.shape == (len(pred_nms["det_probs"]), 9)
    assert np.isclose(
        features[0],
        np.array(
            [
                end_time - start_time,
                low_freq,
                high_freq,
                high_freq - low_freq,
                max_freq,
                max_freq,
                max_freq,
                max_freq,
                np.nan,
            ]
        ),
        equal_nan=True,
    ).all()
