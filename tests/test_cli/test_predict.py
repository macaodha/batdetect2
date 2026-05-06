"""Behavior tests for process CLI workflows."""

import json
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner
from soundevent import data, io

from batdetect2.cli import cli
from batdetect2.outputs import OutputsConfig
from batdetect2.outputs.formats import BatDetect2OutputConfig


def test_cli_process_help() -> None:
    """User story: discover available process modes."""

    result = CliRunner().invoke(cli, ["process", "--help"])

    assert result.exit_code == 0
    assert "directory" in result.output
    assert "file_list" in result.output
    assert "dataset" in result.output


@pytest.mark.slow
def test_cli_process_directory_runs_on_real_audio(
    tmp_path: Path,
    tiny_checkpoint_path: Path,
    single_audio_dir: Path,
) -> None:
    """User story: process all files in a directory."""

    output_path = tmp_path / "predictions"

    result = CliRunner().invoke(
        cli,
        [
            "process",
            "directory",
            "--model",
            str(tiny_checkpoint_path),
            str(single_audio_dir),
            str(output_path),
            "--batch-size",
            "1",
            "--workers",
            "0",
            "--format",
            "batdetect2",
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()
    assert len(list(output_path.glob("*.json"))) == 1


@pytest.mark.slow
def test_cli_process_directory_runs_on_example_audio_data(
    tmp_path: Path,
    tiny_checkpoint_path: Path,
    example_audio_dir: Path,
    example_audio_files: list[Path],
) -> None:
    """User story: process the bundled example audio directory."""

    output_path = tmp_path / "predictions"

    result = CliRunner().invoke(
        cli,
        [
            "process",
            "directory",
            "--model",
            str(tiny_checkpoint_path),
            str(example_audio_dir),
            str(output_path),
            "--batch-size",
            "1",
            "--workers",
            "0",
            "--format",
            "batdetect2",
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()
    assert len(list(output_path.glob("*.json"))) == len(example_audio_files)


@pytest.mark.slow
def test_cli_process_directory_batdetect2_matches_legacy_artifacts(
    tmp_path: Path,
    tiny_checkpoint_path: Path,
    example_audio_dir: Path,
    example_audio_files: list[Path],
    example_anns_dir: Path,
) -> None:
    """User story: process batdetect2 output matches legacy-style files."""

    output_path = tmp_path / "predictions"

    result = CliRunner().invoke(
        cli,
        [
            "process",
            "directory",
            "--model",
            str(tiny_checkpoint_path),
            str(example_audio_dir),
            str(output_path),
            "--batch-size",
            "1",
            "--workers",
            "0",
            "--format",
            "batdetect2",
        ],
    )

    assert result.exit_code == 0

    json_files = sorted(output_path.rglob("*.json"))
    csv_files = sorted(output_path.rglob("*.csv"))

    assert len(json_files) == len(example_audio_files)
    assert len(csv_files) == len(example_audio_files)

    expected_names = sorted(
        audio_file.name for audio_file in example_audio_files
    )
    assert sorted(path.stem for path in json_files) == expected_names
    assert sorted(path.stem for path in csv_files) == expected_names

    first_output = json.loads(json_files[0].read_text())
    assert "file_path" not in first_output
    assert isinstance(first_output["class_name"], str)
    assert first_output["class_name"]

    first_annotation = first_output["annotation"][0]
    assert first_annotation["individual"] == "-1"
    assert isinstance(first_annotation["high_freq"], int)
    assert isinstance(first_annotation["low_freq"], int)

    expected_json = json.loads(
        (example_anns_dir / json_files[0].name).read_text()
    )
    assert first_output["id"] == expected_json["id"]
    assert first_output["time_exp"] == expected_json["time_exp"]

    first_csv = pd.read_csv(csv_files[0], index_col=0)
    assert list(first_csv.columns) == [
        "det_prob",
        "start_time",
        "end_time",
        "high_freq",
        "low_freq",
        "class",
        "class_prob",
    ]


@pytest.mark.slow
def test_cli_process_directory_batdetect2_writes_cnn_features_csv_when_enabled(
    tmp_path: Path,
    tiny_checkpoint_path: Path,
    example_audio_dir: Path,
) -> None:
    """User story: request legacy CNN feature CSV sidecars via config."""

    output_path = tmp_path / "predictions"
    outputs_config_path = tmp_path / "outputs.yaml"
    outputs_config_path.write_text(
        OutputsConfig(
            format=BatDetect2OutputConfig(write_cnn_features_csv=True)
        ).to_yaml_string()
    )

    result = CliRunner().invoke(
        cli,
        [
            "process",
            "directory",
            "--model",
            str(tiny_checkpoint_path),
            str(example_audio_dir),
            str(output_path),
            "--batch-size",
            "1",
            "--workers",
            "0",
            "--outputs-config",
            str(outputs_config_path),
        ],
    )

    assert result.exit_code == 0

    cnn_csvs = sorted(output_path.rglob("*_cnn_features.csv"))
    assert len(cnn_csvs) == 3

    first_df = pd.read_csv(cnn_csvs[0])
    assert not first_df.empty
    assert list(first_df.columns) == [
        str(ii) for ii in range(len(first_df.columns))
    ]


def test_cli_process_file_list_runs_on_real_audio(
    tmp_path: Path,
    tiny_checkpoint_path: Path,
    single_audio_dir: Path,
) -> None:
    """User story: process an explicit list of files."""

    audio_file = next(single_audio_dir.glob("*.wav"))
    file_list = tmp_path / "files.txt"
    file_list.write_text(f"{audio_file}\n")

    output_path = tmp_path / "predictions"

    result = CliRunner().invoke(
        cli,
        [
            "process",
            "file_list",
            "--model",
            str(tiny_checkpoint_path),
            str(file_list),
            str(output_path),
            "--batch-size",
            "1",
            "--workers",
            "0",
            "--format",
            "batdetect2",
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()
    assert len(list(output_path.glob("*.json"))) == 1


def test_cli_process_dataset_runs_on_aoef_metadata(
    tmp_path: Path,
    tiny_checkpoint_path: Path,
    single_audio_dir: Path,
) -> None:
    """User story: process from AOEF dataset metadata file."""

    audio_file = next(single_audio_dir.glob("*.wav"))
    recording = data.Recording.from_file(audio_file)
    clip = data.Clip(
        recording=recording,
        start_time=0,
        end_time=recording.duration,
    )
    annotation_set = data.AnnotationSet(
        name="test",
        description="process dataset test",
        clip_annotations=[data.ClipAnnotation(clip=clip, sound_events=[])],
    )

    dataset_path = tmp_path / "dataset.json"
    io.save(annotation_set, dataset_path)

    output_path = tmp_path / "predictions"

    result = CliRunner().invoke(
        cli,
        [
            "process",
            "dataset",
            "--model",
            str(tiny_checkpoint_path),
            str(dataset_path),
            str(output_path),
            "--batch-size",
            "1",
            "--workers",
            "0",
            "--format",
            "batdetect2",
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()
    assert len(list(output_path.glob("*.json"))) == 1


@pytest.mark.parametrize(
    ("format_name", "expected_pattern", "writes_single_file"),
    [
        ("batdetect2", "*.json", False),
        ("raw", "*.nc", False),
        ("soundevent", "*.json", True),
    ],
)
def test_cli_process_directory_supports_output_format_override(
    tmp_path: Path,
    tiny_checkpoint_path: Path,
    single_audio_dir: Path,
    format_name: str,
    expected_pattern: str,
    writes_single_file: bool,
) -> None:
    """User story: change output format via --format only."""

    output_path = tmp_path / f"predictions_{format_name}"

    result = CliRunner().invoke(
        cli,
        [
            "process",
            "directory",
            "--model",
            str(tiny_checkpoint_path),
            str(single_audio_dir),
            str(output_path),
            "--batch-size",
            "1",
            "--workers",
            "0",
            "--format",
            format_name,
        ],
    )

    assert result.exit_code == 0

    if writes_single_file:
        assert output_path.with_suffix(".json").exists()
    else:
        assert output_path.exists()
        assert len(list(output_path.glob(expected_pattern))) >= 1


def test_cli_process_dataset_deduplicates_recordings(
    tmp_path: Path,
    tiny_checkpoint_path: Path,
    single_audio_dir: Path,
) -> None:
    """User story: duplicated recording entries are processed once."""

    audio_file = next(single_audio_dir.glob("*.wav"))
    recording = data.Recording.from_file(audio_file)
    first_clip = data.Clip(
        recording=recording,
        start_time=0,
        end_time=recording.duration,
    )
    second_clip = data.Clip(
        recording=recording,
        start_time=0,
        end_time=recording.duration,
    )
    annotation_set = data.AnnotationSet(
        name="dupe-recording-dataset",
        description="contains same recording twice",
        clip_annotations=[
            data.ClipAnnotation(clip=first_clip, sound_events=[]),
            data.ClipAnnotation(clip=second_clip, sound_events=[]),
        ],
    )

    dataset_path = tmp_path / "dupes.json"
    io.save(annotation_set, dataset_path)

    output_path = tmp_path / "predictions"
    result = CliRunner().invoke(
        cli,
        [
            "process",
            "dataset",
            "--model",
            str(tiny_checkpoint_path),
            str(dataset_path),
            str(output_path),
            "--batch-size",
            "1",
            "--workers",
            "0",
            "--format",
            "raw",
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()
    assert len(list(output_path.glob("*.nc"))) == 1


def test_cli_process_rejects_unknown_output_format(
    tmp_path: Path,
    tiny_checkpoint_path: Path,
    single_audio_dir: Path,
) -> None:
    """User story: invalid output format fails with error."""

    output_path = tmp_path / "predictions"
    result = CliRunner().invoke(
        cli,
        [
            "process",
            "directory",
            "--model",
            str(tiny_checkpoint_path),
            str(single_audio_dir),
            str(output_path),
            "--format",
            "not_a_real_format",
        ],
    )

    assert result.exit_code != 0
