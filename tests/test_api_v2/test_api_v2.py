from pathlib import Path

import numpy as np
import pytest
import torch
from soundevent.geometry import compute_bounds

from batdetect2.api_v2 import BatDetect2API
from batdetect2.config import BatDetect2Config


@pytest.fixture
def api_v2() -> BatDetect2API:
    """User story: users can create a ready-to-use API from config."""

    config = BatDetect2Config()
    config.inference.loader.batch_size = 2
    return BatDetect2API.from_config(config)


def test_process_file_returns_recording_level_predictions(
    api_v2: BatDetect2API,
    example_audio_files: list[Path],
) -> None:
    """User story: process a file and get detections in recording time."""

    prediction = api_v2.process_file(example_audio_files[0])

    assert prediction.clip.recording.path == example_audio_files[0]
    assert prediction.clip.start_time == 0
    assert prediction.clip.end_time == prediction.clip.recording.duration

    for detection in prediction.detections:
        start, low, end, high = compute_bounds(detection.geometry)
        assert 0 <= start <= end <= prediction.clip.recording.duration
        assert prediction.clip.recording.samplerate > 2 * low
        assert prediction.clip.recording.samplerate > 2 * high
        assert detection.class_scores.shape[0] == len(
            api_v2.targets.class_names
        )


def test_process_files_is_batch_size_invariant(
    api_v2: BatDetect2API,
    example_audio_files: list[Path],
) -> None:
    """User story: changing batch size should not change predictions."""

    preds_batch_1 = api_v2.process_files(example_audio_files, batch_size=1)
    preds_batch_3 = api_v2.process_files(example_audio_files, batch_size=3)

    assert len(preds_batch_1) == len(preds_batch_3)

    by_key_1 = {
        (
            str(pred.clip.recording.path),
            pred.clip.start_time,
            pred.clip.end_time,
        ): pred
        for pred in preds_batch_1
    }
    by_key_3 = {
        (
            str(pred.clip.recording.path),
            pred.clip.start_time,
            pred.clip.end_time,
        ): pred
        for pred in preds_batch_3
    }

    assert set(by_key_1) == set(by_key_3)

    for key in by_key_1:
        pred_1 = by_key_1[key]
        pred_3 = by_key_3[key]
        assert pred_1.clip.start_time == pred_3.clip.start_time
        assert pred_1.clip.end_time == pred_3.clip.end_time
        assert len(pred_1.detections) == len(pred_3.detections)


def test_process_audio_matches_process_spectrogram(
    api_v2: BatDetect2API,
    example_audio_files: list[Path],
) -> None:
    """User story: users can call either audio or spectrogram entrypoint."""

    audio = api_v2.load_audio(example_audio_files[0])
    from_audio = api_v2.process_audio(audio)

    spec = api_v2.generate_spectrogram(audio)
    from_spec = api_v2.process_spectrogram(spec)

    assert len(from_audio) == len(from_spec)

    for det_audio, det_spec in zip(from_audio, from_spec, strict=True):
        bounds_audio = np.array(compute_bounds(det_audio.geometry))
        bounds_spec = np.array(compute_bounds(det_spec.geometry))
        np.testing.assert_allclose(bounds_audio, bounds_spec, atol=1e-6)
        assert np.isclose(det_audio.detection_score, det_spec.detection_score)
        np.testing.assert_allclose(
            det_audio.class_scores,
            det_spec.class_scores,
            atol=1e-6,
        )


def test_process_spectrogram_rejects_batched_input(
    api_v2: BatDetect2API,
) -> None:
    """User story: invalid batched input gives a clear error."""

    spec = torch.zeros((2, 1, 128, 64), dtype=torch.float32)

    with pytest.raises(ValueError, match="Batched spectrograms not supported"):
        api_v2.process_spectrogram(spec)
