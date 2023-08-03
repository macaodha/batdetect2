"""Test suite for feature extraction functions."""

import logging

import librosa
import numpy as np
import pytest

import batdetect2.detector.compute_features as feats
from batdetect2 import api, types
from batdetect2.utils import audio_utils as au

numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.WARNING)


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
    """Test get_feats function with empty spectrogram.

    This tests that the overall flow of the function works, even if the
    spectrogram is empty.
    """
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
    low_freq = index_to_freq(y_pos, spec_height, min_freq, max_freq)
    high_freq = index_to_freq(
        y_pos - bb_height, spec_height, min_freq, max_freq
    )

    pred_nms: types.PredictionResults = {
        "det_probs": np.array([1]),
        "class_probs": np.array([[1]]),
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
                high_freq,
                max_freq,
                max_freq,
                max_freq,
                np.nan,
            ]
        ),
        equal_nan=True,
    ).all()


@pytest.mark.parametrize(
    "max_power",
    [
        30_000,
        31_000,
        32_000,
        33_000,
        34_000,
        35_000,
        36_000,
        37_000,
        38_000,
        39_000,
        40_000,
    ],
)
def test_compute_max_power_bb(max_power: int):
    """Test compute_max_power_bb function."""
    duration = 1
    samplerate = 256_000
    min_freq = 0
    max_freq = 128_000

    start_time = 0.3
    end_time = 0.6
    low_freq = 30_000
    high_freq = 40_000

    audio = np.zeros((int(duration * samplerate),))

    # Add a signal during the time and frequency range of interest
    audio[
        int(start_time * samplerate) : int(end_time * samplerate)
    ] = 0.5 * librosa.tone(
        max_power, sr=samplerate, duration=end_time - start_time
    )

    # Add a more powerful signal outside frequency range of interest
    audio[
        int(start_time * samplerate) : int(end_time * samplerate)
    ] += 2 * librosa.tone(
        80_000, sr=samplerate, duration=end_time - start_time
    )

    params = api.get_config(
        min_freq=min_freq,
        max_freq=max_freq,
        target_samp_rate=samplerate,
    )

    spec, _ = au.generate_spectrogram(
        audio,
        samplerate,
        params,
    )

    x_start = int(
        au.time_to_x_coords(
            start_time,
            samplerate,
            params["fft_win_length"],
            params["fft_overlap"],
        )
    )

    x_end = int(
        au.time_to_x_coords(
            end_time,
            samplerate,
            params["fft_win_length"],
            params["fft_overlap"],
        )
    )

    num_freq_bins = spec.shape[0]
    y_low = num_freq_bins - int(num_freq_bins * low_freq / max_freq)
    y_high = num_freq_bins - int(num_freq_bins * high_freq / max_freq)

    prediction: types.Prediction = {
        "det_prob": 1,
        "class_prob": np.ones((1,)),
        "x_pos": x_start,
        "y_pos": int(y_low),
        "bb_width": int(x_end - x_start),
        "bb_height": int(y_low - y_high),
        "start_time": start_time,
        "end_time": end_time,
        "low_freq": low_freq,
        "high_freq": high_freq,
    }

    print(prediction)

    max_power_bb = feats.compute_max_power_bb(
        prediction,
        spec,
        min_freq=min_freq,
        max_freq=max_freq,
    )

    assert abs(max_power_bb - max_power) <= 500


def test_compute_max_power():
    """Test compute_max_power_bb function."""
    duration = 3
    samplerate = 16_000
    min_freq = 0
    max_freq = 8_000

    start_time = 1
    end_time = 2
    low_freq = 3_000
    high_freq = 4_000
    max_power = 5_000

    audio = np.zeros((int(duration * samplerate),))

    # Add a signal during the time and frequency range of interest
    audio[
        int(start_time * samplerate) : int(end_time * samplerate)
    ] = 0.5 * librosa.tone(
        3_500, sr=samplerate, duration=end_time - start_time
    )

    # Add a more powerful signal outside frequency range of interest
    audio[
        int(start_time * samplerate) : int(end_time * samplerate)
    ] += 2 * librosa.tone(
        max_power, sr=samplerate, duration=end_time - start_time
    )

    params = api.get_config(
        min_freq=min_freq,
        max_freq=max_freq,
        target_samp_rate=samplerate,
    )

    spec, _ = au.generate_spectrogram(
        audio,
        samplerate,
        params,
    )

    x_start = int(
        au.time_to_x_coords(
            start_time,
            samplerate,
            params["fft_win_length"],
            params["fft_overlap"],
        )
    )

    x_end = int(
        au.time_to_x_coords(
            end_time,
            samplerate,
            params["fft_win_length"],
            params["fft_overlap"],
        )
    )

    num_freq_bins = spec.shape[0]
    y_low = int(num_freq_bins * low_freq / max_freq)
    y_high = int(num_freq_bins * high_freq / max_freq)

    prediction: types.Prediction = {
        "det_prob": 1,
        "class_prob": np.ones((1,)),
        "x_pos": x_start,
        "y_pos": int(y_high),
        "bb_width": int(x_end - x_start),
        "bb_height": int(y_high - y_low),
        "start_time": start_time,
        "end_time": end_time,
        "low_freq": low_freq,
        "high_freq": high_freq,
    }

    computed_max_power = feats.compute_max_power(
        prediction,
        spec,
        min_freq=min_freq,
        max_freq=max_freq,
    )

    assert abs(computed_max_power - max_power) < 100
