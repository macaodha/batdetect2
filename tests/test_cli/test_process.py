import json
import shutil
from collections import Counter
from pathlib import Path

from click.testing import CliRunner
from soundevent.geometry import compute_bounds

from batdetect2 import BatDetect2API
from batdetect2.cli import cli


def test_cli_process_directory_merges_clip_outputs_per_recording(
    tmp_path: Path,
    contrib_dir: Path,
) -> None:
    recording_path = contrib_dir / "jeff37" / "0166_20240531_223911.wav"

    source_folder = tmp_path / "audio"
    source_folder.mkdir()
    shutil.copy2(
        recording_path,
        source_folder / "example_audio.wav",
    )

    destination_folder = tmp_path / "results"
    destination_folder.mkdir()

    api = BatDetect2API.from_checkpoint()

    api_outputs = api.process_directory(
        source_folder,
        detection_threshold=0.3,
    )

    # Get all detections regardless of clip
    detections = [
        detection
        for clip_detections in api_outputs
        for detection in clip_detections.detections
    ]

    result = CliRunner().invoke(
        cli,
        args=[
            "process",
            "directory",
            str(source_folder),
            str(destination_folder),
            "--detection-threshold",
            "0.3",
        ],
    )

    assert result.exit_code == 0
    assert destination_folder.exists()

    output_json = destination_folder / "example_audio.wav.json"
    assert output_json.exists()

    saved_detections = json.loads(output_json.read_text())

    expected_annotations = Counter(
        (
            round(float(start_time), 4),
            round(float(end_time), 4),
            int(low_freq),
            int(high_freq),
            round(float(detection.class_scores.max()), 3),
            round(float(detection.detection_score), 3),
        )
        for detection in detections
        for start_time, low_freq, end_time, high_freq in [
            compute_bounds(detection.geometry)
        ]
    )

    actual_annotations = Counter(
        (
            annotation["start_time"],
            annotation["end_time"],
            annotation["low_freq"],
            annotation["high_freq"],
            annotation["class_prob"],
            annotation["det_prob"],
        )
        for annotation in saved_detections["annotation"]
    )

    assert actual_annotations == expected_annotations
