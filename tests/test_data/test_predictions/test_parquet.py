from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
from soundevent import data

from batdetect2.data.predictions import (
    ParquetOutputConfig,
    build_output_formatter,
)
from batdetect2.typing import (
    ClipDetections,
    Detection,
    TargetProtocol,
)


@pytest.fixture
def sample_formatter(sample_targets: TargetProtocol):
    return build_output_formatter(
        config=ParquetOutputConfig(),
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

    path = tmp_path / "predictions.parquet"

    sample_formatter.save(predictions=[prediction], path=path)

    assert path.exists()

    recovered = sample_formatter.load(path=path)

    assert len(recovered) == 1
    assert recovered[0].clip == prediction.clip

    for recovered_prediction, detection in zip(
        recovered[0].detections, detections, strict=True
    ):
        assert (
            recovered_prediction.detection_score == detection.detection_score
        )
        # Note: floating point comparison might need tolerance, but parquet should preserve float64
        assert np.allclose(
            recovered_prediction.class_scores, detection.class_scores
        )
        assert np.allclose(recovered_prediction.features, detection.features)
        assert recovered_prediction.geometry == detection.geometry


def test_multiple_clips(
    sample_formatter,
    clip: data.Clip,
    sample_targets: TargetProtocol,
    tmp_path: Path,
):
    # Create a second clip
    clip2 = clip.model_copy(update={"uuid": uuid4()})

    detections1 = [
        Detection(
            geometry=data.BoundingBox(
                coordinates=list(np.random.uniform(size=[4]))
            ),
            detection_score=0.8,
            class_scores=np.random.uniform(
                size=len(sample_targets.class_names)
            ),
            features=np.random.uniform(size=32),
        )
    ]

    detections2 = [
        Detection(
            geometry=data.BoundingBox(
                coordinates=list(np.random.uniform(size=[4]))
            ),
            detection_score=0.9,
            class_scores=np.random.uniform(
                size=len(sample_targets.class_names)
            ),
            features=np.random.uniform(size=32),
        )
    ]

    predictions = [
        ClipDetections(clip=clip, detections=detections1),
        ClipDetections(clip=clip2, detections=detections2),
    ]

    path = tmp_path / "multi_predictions.parquet"
    sample_formatter.save(predictions=predictions, path=path)

    recovered = sample_formatter.load(path=path)

    assert len(recovered) == 2
    # Order might not be preserved if we don't sort, but implementation appends so it should be
    # However, let's sort by clip uuid to be safe if needed, or just check existence

    recovered_uuids = {p.clip.uuid for p in recovered}
    expected_uuids = {clip.uuid, clip2.uuid}
    assert recovered_uuids == expected_uuids


def test_complex_geometry(
    sample_formatter,
    clip: data.Clip,
    sample_targets: TargetProtocol,
    tmp_path: Path,
):
    # Create a polygon geometry
    polygon = data.Polygon(
        coordinates=[
            [
                [0.0, 10000.0],
                [0.1, 20000.0],
                [0.2, 10000.0],
                [0.0, 10000.0],
            ]
        ]
    )

    detections = [
        Detection(
            geometry=polygon,
            detection_score=0.95,
            class_scores=np.random.uniform(
                size=len(sample_targets.class_names)
            ),
            features=np.random.uniform(size=32),
        )
    ]

    prediction = ClipDetections(clip=clip, detections=detections)

    path = tmp_path / "complex_geometry.parquet"
    sample_formatter.save(predictions=[prediction], path=path)

    recovered = sample_formatter.load(path=path)

    assert len(recovered) == 1
    assert len(recovered[0].detections) == 1

    recovered_pred = recovered[0].detections[0]

    # Check if geometry is recovered correctly as a Polygon
    assert isinstance(recovered_pred.geometry, data.Polygon)
    assert recovered_pred.geometry == polygon
