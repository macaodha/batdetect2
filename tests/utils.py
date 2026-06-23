import numpy as np
from soundevent.geometry import compute_bounds

from batdetect2.postprocess.types import ClipDetections


def assert_clip_detections_equal(
    detections: ClipDetections,
    other: ClipDetections,
) -> None:
    """Assert two clip-detection objects are numerically equivalent."""
    assert detections.clip.recording.path == other.clip.recording.path
    assert detections.clip.start_time == other.clip.start_time
    assert detections.clip.end_time == other.clip.end_time
    assert len(detections.detections) == len(other.detections)

    sorted_detections = sorted(
        detections.detections,
        key=lambda det: (
            compute_bounds(det.geometry)[0],
            compute_bounds(det.geometry)[1],
        ),
    )

    sorted_other = sorted(
        other.detections,
        key=lambda det: (
            compute_bounds(det.geometry)[0],
            compute_bounds(det.geometry)[1],
        ),
    )

    for det, other_det in zip(
        sorted_detections,
        sorted_other,
        strict=True,
    ):
        np.testing.assert_allclose(
            np.array(compute_bounds(det.geometry)),
            np.array(compute_bounds(other_det.geometry)),
            atol=2e-2,
        )
        assert np.isclose(
            det.detection_score,
            other_det.detection_score,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            det.class_scores,
            other_det.class_scores,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            det.features,
            other_det.features,
            atol=2e-6,
        )
