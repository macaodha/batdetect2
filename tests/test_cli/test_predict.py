"""Behavior tests for process CLI workflows."""

from pathlib import Path

import pytest
from click.testing import CliRunner
from soundevent import data, io

from batdetect2.cli import cli


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
            str(tiny_checkpoint_path),
            str(single_audio_dir),
            str(output_path),
            "--format",
            "not_a_real_format",
        ],
    )

    assert result.exit_code != 0
