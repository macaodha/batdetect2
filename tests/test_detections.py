"""Test suite to ensure that model detections are not incorrect."""

import os

from batdetect2 import api

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def test_no_detections_above_nyquist():
    """Test that no detections are made above the nyquist frequency."""
    # Recording donated by @@kdarras
    path = os.path.join(DATA_DIR, "20230322_172000_selec2.wav")

    # This recording has a sampling rate of 192 kHz
    nyquist = 192_000 / 2

    output = api.process_file(path)
    predictions = output["pred_dict"]
    assert len(predictions["annotation"]) != 0
    assert all(
        pred["high_freq"] < nyquist for pred in predictions["annotation"]
    )
