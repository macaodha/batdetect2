import numpy as np
from soundevent import data

from batdetect2.outputs.transforms.clip_transforms import (
    RemoveAboveNyquist,
    RemoveAtEdges,
)
from batdetect2.postprocess.types import ClipDetections, Detection


def _detection(
    start_time: float,
    low_freq: float,
    end_time: float,
    high_freq: float,
) -> Detection:
    return Detection(
        geometry=data.BoundingBox(
            coordinates=[start_time, low_freq, end_time, high_freq]
        ),
        detection_score=0.9,
        class_scores=np.array([0.9]),
        features=np.array([1.0, 2.0]),
    )


def test_remove_above_nyquist_high_freq_mode(clip: data.Clip) -> None:
    # Nyquist should be at 128kHz
    assert clip.recording.samplerate == 256_000

    transform = RemoveAboveNyquist(mode="high_freq", buffer=0)
    prediction = ClipDetections(
        clip=clip,
        detections=[
            _detection(0.1, 10_000, 0.2, 120_000),
            _detection(0.1, 10_000, 0.2, 130_000),
        ],
    )

    out = transform(prediction)

    assert len(out.detections) == 1


def test_remove_above_nyquist_low_freq_mode(clip: data.Clip) -> None:
    transform = RemoveAboveNyquist(mode="low_freq", buffer=0)
    prediction = ClipDetections(
        clip=clip,
        detections=[
            _detection(0.1, 120_000, 0.2, 140_000),
            _detection(0.1, 130_000, 0.2, 140_000),
        ],
    )

    out = transform(prediction)

    assert len(out.detections) == 1


def test_remove_above_nyquist_respects_buffer(clip: data.Clip) -> None:
    transform = RemoveAboveNyquist(mode="high_freq", buffer=5_000)
    prediction = ClipDetections(
        clip=clip,
        detections=[
            _detection(0.1, 10_000, 0.2, 122_000),
            _detection(0.1, 10_000, 0.2, 124_000),
        ],
    )

    out = transform(prediction)

    assert len(out.detections) == 1


def test_remove_at_edges_start_mode(clip: data.Clip) -> None:
    clip = clip.model_copy(update={"start_time": 10.0, "end_time": 20.0})
    transform = RemoveAtEdges(buffer=1.0, mode="start_time")
    prediction = ClipDetections(
        clip=clip,
        detections=[
            _detection(10.2, 20_000, 10.4, 30_000),
            _detection(11.2, 20_000, 11.4, 30_000),
        ],
    )

    out = transform(prediction)

    assert len(out.detections) == 1


def test_remove_at_edges_end_mode(clip: data.Clip) -> None:
    clip = clip.model_copy(update={"start_time": 10.0, "end_time": 20.0})
    transform = RemoveAtEdges(buffer=1.0, mode="end_time")
    prediction = ClipDetections(
        clip=clip,
        detections=[
            _detection(11.2, 20_000, 19.8, 30_000),
            _detection(11.2, 20_000, 18.6, 30_000),
        ],
    )

    out = transform(prediction)

    assert len(out.detections) == 1


def test_remove_at_edges_both_mode(clip: data.Clip) -> None:
    clip = clip.model_copy(update={"start_time": 10.0, "end_time": 20.0})
    transform = RemoveAtEdges(buffer=1.0, mode="both")
    prediction = ClipDetections(
        clip=clip,
        detections=[
            _detection(10.2, 20_000, 18.5, 30_000),
            _detection(11.2, 20_000, 18.5, 30_000),
            _detection(11.2, 20_000, 19.8, 30_000),
        ],
    )

    out = transform(prediction)

    assert len(out.detections) == 1
