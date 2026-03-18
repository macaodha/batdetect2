import numpy as np
from soundevent import data
from soundevent.geometry import compute_bounds

from batdetect2.outputs.transforms.detection_transforms import (
    FilterByDuration,
    FilterByDurationConfig,
    FilterByFrequency,
    FilterByFrequencyConfig,
    detection_transforms,
    shift_detection_time,
    shift_detections_to_start_time,
)
from batdetect2.postprocess.types import Detection


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


def test_shift_detection_time_moves_geometry_by_offset() -> None:
    detection = _detection(0.1, 20_000, 0.2, 30_000)

    shifted = shift_detection_time(detection, time=2.5)
    start, low, end, high = compute_bounds(shifted.geometry)

    assert np.isclose(start, 2.6)
    assert np.isclose(end, 2.7)
    assert np.isclose(low, 20_000)
    assert np.isclose(high, 30_000)


def test_shift_detections_to_start_time_zero_is_identity() -> None:
    detections = [_detection(0.1, 20_000, 0.2, 30_000)]

    shifted = shift_detections_to_start_time(detections, start_time=0)

    assert len(shifted) == 1
    assert shifted[0] is detections[0]


def test_filter_by_frequency_low_freq_mode() -> None:
    transform = FilterByFrequency(
        min_freq=20_000,
        max_freq=40_000,
        mode="low_freq",
    )

    assert transform(_detection(0.1, 25_000, 0.2, 60_000)) is not None
    assert transform(_detection(0.1, 10_000, 0.2, 60_000)) is None


def test_filter_by_frequency_high_freq_mode() -> None:
    transform = FilterByFrequency(
        min_freq=20_000,
        max_freq=40_000,
        mode="high_freq",
    )

    assert transform(_detection(0.1, 10_000, 0.2, 35_000)) is not None
    assert transform(_detection(0.1, 10_000, 0.2, 60_000)) is None


def test_filter_by_frequency_both_mode_current_semantics() -> None:
    transform = FilterByFrequency(
        min_freq=20_000,
        max_freq=40_000,
        mode="both",
    )

    # low >= min passes
    assert transform(_detection(0.1, 25_000, 0.2, 80_000)) is not None
    # high <= max passes
    assert transform(_detection(0.1, 10_000, 0.2, 35_000)) is not None
    # neither condition passes
    assert transform(_detection(0.1, 10_000, 0.2, 80_000)) is None


def test_filter_by_duration_keeps_within_range() -> None:
    transform = FilterByDuration(min_duration=0.04, max_duration=0.06)

    kept = transform(_detection(0.1, 20_000, 0.15, 30_000))
    removed = transform(_detection(0.1, 20_000, 0.2, 30_000))

    assert kept is not None
    assert removed is None


def test_detection_transform_registry_builds_builtin_transforms() -> None:
    frequency_transform = detection_transforms.build(
        FilterByFrequencyConfig(
            min_freq=20_000,
            max_freq=40_000,
            mode="high_freq",
        )
    )
    duration_transform = detection_transforms.build(
        FilterByDurationConfig(
            min_duration=0.01,
            max_duration=0.2,
        )
    )

    assert callable(frequency_transform)
    assert callable(duration_transform)
