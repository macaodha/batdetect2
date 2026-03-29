from dataclasses import replace

import numpy as np
import torch
from soundevent import data
from soundevent.geometry import compute_bounds

from batdetect2.outputs import build_output_transform
from batdetect2.outputs.transforms import OutputTransform
from batdetect2.postprocess.types import (
    ClipDetections,
    ClipDetectionsTensor,
    Detection,
)
from batdetect2.targets import TargetConfig, build_roi_mapping
from batdetect2.targets.types import TargetProtocol


def _mock_clip_detections_tensor() -> ClipDetectionsTensor:
    return ClipDetectionsTensor(
        scores=torch.tensor([0.9], dtype=torch.float32),
        # NOTE: Time is scaled by 1000
        sizes=torch.tensor([[100, 1_000.0]], dtype=torch.float32),
        class_scores=torch.tensor([[0.8, 0.2]], dtype=torch.float32),
        times=torch.tensor([0.2], dtype=torch.float32),
        frequencies=torch.tensor([60_000.0], dtype=torch.float32),
        features=torch.tensor([[1.0, 2.0]], dtype=torch.float32),
    )


def _build_roi_mapper(targets: TargetProtocol):
    config_obj = getattr(targets, "config", None)
    target_config = (
        config_obj if isinstance(config_obj, TargetConfig) else None
    )
    return build_roi_mapping(
        config=(target_config.roi if target_config is not None else None),
    )


def test_shift_time_to_clip_start(sample_targets: TargetProtocol):
    raw = _mock_clip_detections_tensor()
    transform = build_output_transform(
        targets=sample_targets,
        roi_mapper=_build_roi_mapper(sample_targets),
    )

    transformed = transform.to_detections(raw, start_time=2.5)
    start_time, _, end_time, _ = compute_bounds(transformed[0].geometry)

    assert np.isclose(start_time, 2.7)
    assert np.isclose(end_time, 2.8)


def test_to_clip_detections_shifts_by_clip_start(
    clip: data.Clip,
    sample_targets: TargetProtocol,
):
    clip = clip.model_copy(update={"start_time": 2.5, "end_time": 3.0})
    transform = build_output_transform(
        targets=sample_targets,
        roi_mapper=_build_roi_mapper(sample_targets),
    )
    raw = _mock_clip_detections_tensor()
    shifted = transform.to_clip_detections(detections=raw, clip=clip)
    start_time, _, end_time, _ = compute_bounds(shifted.detections[0].geometry)
    assert np.isclose(start_time, 2.7)
    assert np.isclose(end_time, 2.8)


def test_detection_and_clip_transforms_applied_in_order(
    clip: data.Clip,
    sample_targets: TargetProtocol,
):
    clip = clip.model_copy(update={"start_time": 2.5, "end_time": 3.0})

    detection_1 = Detection(
        geometry=data.BoundingBox(coordinates=[0.1, 10_000, 0.2, 12_000]),
        detection_score=0.5,
        class_scores=np.array([0.9]),
        features=np.array([1.0, 2.0]),
    )
    detection_2 = Detection(
        geometry=data.BoundingBox(coordinates=[0.2, 10_000, 0.3, 12_000]),
        detection_score=0.7,
        class_scores=np.array([0.9]),
        features=np.array([1.0, 2.0]),
    )

    def boost_score(detection: Detection) -> Detection:
        return replace(
            detection,
            detection_score=detection.detection_score + 0.2,
        )

    def keep_high_score(detection: Detection) -> Detection | None:
        if detection.detection_score < 0.8:
            return None
        return detection

    def tag_clip_transform(prediction: ClipDetections) -> ClipDetections:
        detections = [
            replace(detection, detection_score=1.0)
            for detection in prediction.detections
        ]
        return replace(prediction, detections=detections)

    transform = OutputTransform(
        targets=sample_targets,
        roi_mapper=_build_roi_mapper(sample_targets),
        detection_transform_steps=[boost_score, keep_high_score],
        clip_transform_steps=[tag_clip_transform],
    )
    transformed = transform(
        [ClipDetections(clip=clip, detections=[detection_1, detection_2])]
    )[0]

    assert len(transformed.detections) == 1
    assert transformed.detections[0].detection_score == 1.0

    start_time, _, _, _ = compute_bounds(transformed.detections[0].geometry)
    assert np.isclose(start_time, 2.7)
