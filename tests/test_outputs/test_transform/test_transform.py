import numpy as np
from soundevent import data
from soundevent.geometry import compute_bounds

from batdetect2.outputs import build_output_transform
from batdetect2.typing import ClipDetections, Detection


def test_shift_time_to_clip_start(clip: data.Clip):
    clip = clip.model_copy(update={"start_time": 2.5, "end_time": 3.0})

    detection = Detection(
        geometry=data.BoundingBox(coordinates=[0.1, 10_000, 0.2, 12_000]),
        detection_score=0.9,
        class_scores=np.array([0.9]),
        features=np.array([1.0, 2.0]),
    )

    transformed = build_output_transform()(
        [ClipDetections(clip=clip, detections=[detection])]
    )[0]

    start_time, _, end_time, _ = compute_bounds(
        transformed.detections[0].geometry
    )

    assert np.isclose(start_time, 2.6)
    assert np.isclose(end_time, 2.7)


def test_transform_identity_when_disabled(clip: data.Clip):
    clip = clip.model_copy(update={"start_time": 2.5, "end_time": 3.0})

    detection = Detection(
        geometry=data.BoundingBox(coordinates=[0.1, 10_000, 0.2, 12_000]),
        detection_score=0.9,
        class_scores=np.array([0.9]),
        features=np.array([1.0, 2.0]),
    )

    transform = build_output_transform(
        config={"shift_time_to_clip_start": False}
    )
    transformed = transform(
        [ClipDetections(clip=clip, detections=[detection])]
    )[0]

    assert transformed.detections[0].geometry == detection.geometry
