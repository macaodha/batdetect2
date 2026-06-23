from pathlib import Path

import pytest
from soundevent import data

from batdetect2.inference.batch import run_batch_inference
from batdetect2.targets import build_roi_mapping, build_targets
from batdetect2.train import load_model_from_checkpoint
from tests.utils import assert_clip_detections_equal

pytestmark = pytest.mark.slow


def test_run_batch_inference_matches_single_clip_inference(
    contrib_dir: Path,
) -> None:
    recording = data.Recording.from_file(
        contrib_dir / "jeff37" / "0166_20240531_223911.wav"
    )
    clips = [
        data.Clip(recording=recording, start_time=start, end_time=start + 1.0)
        for start in (0.0, 1.0, 2.0)
    ]
    model, configs = load_model_from_checkpoint()
    targets = build_targets(configs.targets)
    roi_mapper = build_roi_mapping(configs.targets.roi)

    batched_predictions = run_batch_inference(
        model,
        clips,
        targets=targets,
        roi_mapper=roi_mapper,
        batch_size=3,
        num_workers=0,
    )
    single_predictions = [
        run_batch_inference(
            model,
            [clip],
            targets=targets,
            roi_mapper=roi_mapper,
            batch_size=1,
            num_workers=0,
        )[0]
        for clip in clips
    ]

    assert len(batched_predictions) == len(single_predictions)

    for batched, single in zip(
        batched_predictions,
        single_predictions,
        strict=True,
    ):
        assert_clip_detections_equal(batched, single)
