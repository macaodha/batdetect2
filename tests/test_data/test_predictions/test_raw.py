from pathlib import Path

import numpy as np
import pytest
from soundevent import data

from batdetect2.outputs.formats import RawOutputConfig, build_output_formatter
from batdetect2.typing import (
    ClipDetections,
    Detection,
    TargetProtocol,
)


@pytest.fixture
def sample_formatter(sample_targets: TargetProtocol):
    return build_output_formatter(
        config=RawOutputConfig(),
        targets=sample_targets,
    )


def test_roundtrip(
    sample_formatter,
    clip: data.Clip,
    sample_targets: TargetProtocol,
    tmp_path: Path,
):
    detections = [
        Detection(
            geometry=data.BoundingBox(
                coordinates=list(np.random.uniform(size=[4]))
            ),
            detection_score=0.5,
            class_scores=np.random.uniform(
                size=len(sample_targets.class_names)
            ),
            features=np.random.uniform(size=32),
        )
        for _ in range(10)
    ]

    prediction = ClipDetections(clip=clip, detections=detections)

    path = tmp_path / "predictions"

    sample_formatter.save(predictions=[prediction], path=path)

    recovered = sample_formatter.load(path=path)

    assert len(recovered) == 1
    assert recovered[0].clip == prediction.clip

    for recovered_prediction, detection in zip(
        recovered[0].detections,
        detections,
        strict=True,
    ):
        assert (
            recovered_prediction.detection_score == detection.detection_score
        )
        assert (
            recovered_prediction.class_scores == detection.class_scores
        ).all()
        assert (recovered_prediction.features == detection.features).all()
        assert recovered_prediction.geometry == detection.geometry
