import numpy as np
import torch
from soundevent import data
from soundevent.geometry import compute_bounds

from batdetect2.outputs import build_output_transform
from batdetect2.postprocess.types import ClipDetectionsTensor
from batdetect2.targets.types import TargetProtocol


def _mock_clip_detections_tensor(
    *,
    time: float,
    duration: float,
    frequency: float,
    bandwidth: float,
) -> ClipDetectionsTensor:
    # NOTE: size time is represented in milliseconds.
    return ClipDetectionsTensor(
        scores=torch.tensor([0.9], dtype=torch.float32),
        sizes=torch.tensor(
            [[duration * 1_000, bandwidth]], dtype=torch.float32
        ),
        class_scores=torch.tensor([[0.8, 0.2]], dtype=torch.float32),
        times=torch.tensor([time], dtype=torch.float32),
        frequencies=torch.tensor([frequency], dtype=torch.float32),
        features=torch.tensor([[1.0, 2.0]], dtype=torch.float32),
    )


def test_pipeline_from_config_applies_detection_and_clip_transforms(
    clip: data.Clip,
    sample_targets: TargetProtocol,
) -> None:
    clip = clip.model_copy(update={"start_time": 10.0, "end_time": 11.0})
    transform = build_output_transform(
        targets=sample_targets,
        config={
            "detection_transforms": [
                {
                    "name": "filter_by_duration",
                    "min_duration": 0.08,
                    "max_duration": 0.12,
                }
            ],
            "clip_transforms": [
                {
                    "name": "remove_at_edges",
                    "buffer": 0.1,
                    "mode": "both",
                }
            ],
        },
    )

    raw = _mock_clip_detections_tensor(
        time=0.03,
        duration=0.1,
        frequency=60_000,
        bandwidth=1_000,
    )

    prediction = transform.to_clip_detections(raw, clip=clip)

    # duration filter keeps it, edge filter removes it.
    assert len(prediction.detections) == 0


def test_pipeline_keeps_detection_when_all_filters_pass(
    clip: data.Clip,
    sample_targets: TargetProtocol,
) -> None:
    clip = clip.model_copy(update={"start_time": 10.0, "end_time": 11.0})
    transform = build_output_transform(
        targets=sample_targets,
        config={
            "detection_transforms": [
                {
                    "name": "filter_by_duration",
                    "min_duration": 0.08,
                    "max_duration": 0.12,
                },
            ],
            "clip_transforms": [
                {
                    "name": "remove_at_edges",
                    "buffer": 0.05,
                    "mode": "both",
                }
            ],
        },
    )

    raw = _mock_clip_detections_tensor(
        time=0.3,
        duration=0.1,
        frequency=60_000,
        bandwidth=1_000,
    )

    prediction = transform.to_clip_detections(raw, clip=clip)

    assert len(prediction.detections) == 1
    start_time, _, _, _ = compute_bounds(prediction.detections[0].geometry)
    assert np.isclose(start_time, 10.3)


def test_remove_above_nyquist_uses_clip_recording_metadata(
    clip: data.Clip,
    sample_targets: TargetProtocol,
) -> None:
    clip = clip.model_copy(update={"start_time": 0.0, "end_time": 1.0})
    transform = build_output_transform(
        targets=sample_targets,
        config={
            "clip_transforms": [
                {
                    "name": "remove_above_nyquist",
                    "mode": "high_freq",
                    "buffer": 0,
                }
            ]
        },
    )

    raw = _mock_clip_detections_tensor(
        time=0.5,
        duration=0.05,
        frequency=127_500,
        bandwidth=2_000,
    )

    prediction = transform.to_clip_detections(raw, clip=clip)

    # clip fixture samplerate is 256_000, nyquist is 128_000, high bound
    # becomes 128_500 and must be removed.
    assert len(prediction.detections) == 0
