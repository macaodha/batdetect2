import numpy as np
import pytest
from soundevent import data

from batdetect2.evaluate.tasks import build_task
from batdetect2.evaluate.tasks.classification import ClassificationTaskConfig
from batdetect2.typing import ClipDetections
from batdetect2.typing.targets import TargetProtocol


def test_classification(
    clip: data.Clip,
    sample_targets: TargetProtocol,
    create_detection,
    create_annotation,
):
    config = ClassificationTaskConfig.model_validate(
        {
            "name": "sound_event_classification",
            "metrics": [{"name": "average_precision"}],
        }
    )

    evaluator = build_task(config, targets=sample_targets)

    # Create a dummy prediction
    prediction = ClipDetections(
        clip=clip,
        detections=[
            create_detection(
                start_time=1 + 0.1 * index,
                pip_score=score,
            )
            for index, score in enumerate(np.linspace(0, 1, 100))
        ]
        + [
            create_detection(
                start_time=1.05 + 0.1 * index,
                myo_score=score,
            )
            for index, score in enumerate(np.linspace(1, 0, 100))
        ],
    )

    # Create a dummy annotation
    gt = data.ClipAnnotation(
        clip=clip,
        sound_events=[
            create_annotation(
                start_time=1 + 0.1 * index,
                is_target=index % 2 == 0,
                class_name="pippip",
            )
            for index in range(100)
        ]
        + [
            create_annotation(
                start_time=1.05 + 0.1 * index,
                is_target=index % 3 == 0,
                class_name="myomyo",
            )
            for index in range(100)
        ],
    )

    evals = evaluator.evaluate([gt], [prediction])
    metrics = evaluator.compute_metrics(evals)

    assert metrics["classification/average_precision/pippip"] == pytest.approx(
        0.5, abs=0.005
    )
    assert metrics["classification/average_precision/myomyo"] == pytest.approx(
        0.371, abs=0.005
    )
    assert metrics["classification/mean_average_precision"] == pytest.approx(
        (0.5 + 0.371) / 2, abs=0.005
    )
