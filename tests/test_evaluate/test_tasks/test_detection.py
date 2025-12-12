import numpy as np
import pytest
from soundevent import data

from batdetect2.evaluate.tasks import build_task
from batdetect2.evaluate.tasks.detection import DetectionTaskConfig
from batdetect2.typing import ClipDetections
from batdetect2.typing.targets import TargetProtocol


def test_detection(
    clip: data.Clip,
    sample_targets: TargetProtocol,
    create_detection,
    create_annotation,
):
    config = DetectionTaskConfig.model_validate(
        {
            "name": "sound_event_detection",
            "metrics": [{"name": "average_precision"}],
        }
    )
    evaluator = build_task(config, targets=sample_targets)

    # Create a dummy prediction
    prediction = ClipDetections(
        clip=clip,
        detections=[
            create_detection(start_time=1 + 0.1 * index, detection_score=score)
            for index, score in enumerate(np.linspace(0, 1, 100))
        ],
    )

    # Create a dummy annotation
    gt = data.ClipAnnotation(
        clip=clip,
        sound_events=[
            # Only half of the annotations are targets
            create_annotation(
                start_time=1 + 0.1 * index,
                is_target=index % 2 == 0,
            )
            for index in range(100)
        ],
    )

    # Run the task
    evals = evaluator.evaluate([gt], [prediction])
    metrics = evaluator.compute_metrics(evals)
    assert metrics["detection/average_precision"] == pytest.approx(0.5)
