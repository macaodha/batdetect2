"""Behavior tests for legacy detect command."""

from pathlib import Path

import pandas as pd
from click.testing import CliRunner

from batdetect2.cli import cli


def test_cli_detect_command_on_test_audio(tmp_path: Path) -> None:
    """User story: run legacy detect on example audio directory."""

    results_dir = tmp_path / "results"

    result = CliRunner().invoke(
        cli,
        [
            "detect",
            "example_data/audio",
            str(results_dir),
            "0.3",
        ],
    )

    assert result.exit_code == 0
    assert results_dir.exists()
    assert len(list(results_dir.glob("*.csv"))) == 3
    assert len(list(results_dir.glob("*.json"))) == 3


def test_cli_detect_command_with_non_trivial_time_expansion(
    tmp_path: Path,
) -> None:
    """User story: set time expansion in legacy detect command."""

    results_dir = tmp_path / "results"

    result = CliRunner().invoke(
        cli,
        [
            "detect",
            "example_data/audio",
            str(results_dir),
            "0.3",
            "--time_expansion_factor",
            "10",
        ],
    )

    assert result.exit_code == 0
    assert "Time Expansion Factor: 10" in result.stdout


def test_cli_detect_command_with_spec_feature_flag(tmp_path: Path) -> None:
    """User story: request extra spectral features in output CSV."""

    results_dir = tmp_path / "results"

    result = CliRunner().invoke(
        cli,
        [
            "detect",
            "example_data/audio",
            str(results_dir),
            "0.3",
            "--spec_features",
        ],
    )

    assert result.exit_code == 0
    assert results_dir.exists()

    csv_files = [path.name for path in results_dir.glob("*.csv")]

    expected_files = [
        "20170701_213954-MYOMYS-LR_0_0.5.wav_spec_features.csv",
        "20180530_213516-EPTSER-LR_0_0.5.wav_spec_features.csv",
        "20180627_215323-RHIFER-LR_0_0.5.wav_spec_features.csv",
    ]

    for expected_file in expected_files:
        assert expected_file in csv_files
        df = pd.read_csv(results_dir / expected_file)
        assert not (df.duration == -1).any()


def test_cli_detect_fails_gracefully_on_empty_file(tmp_path: Path) -> None:
    """User story: bad/empty input file reports error but command survives."""

    results_dir = tmp_path / "results"
    target = tmp_path / "audio"
    target.mkdir()

    empty_file = target / "empty.wav"
    empty_file.touch()

    result = CliRunner().invoke(
        cli,
        args=[
            "detect",
            str(target),
            str(results_dir),
            "0.3",
            "--spec_features",
        ],
    )

    assert result.exit_code == 0
    assert f"Error processing file {empty_file}" in result.output
