"""Test the command line interface."""
from pathlib import Path
from click.testing import CliRunner
import pandas as pd

from batdetect2.cli import cli


def test_cli_base_command():
    """Test the base command."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "BatDetect2 - Bat Call Detection and Classification" in result.output


def test_cli_detect_command_help():
    """Test the detect command help."""
    runner = CliRunner()
    result = runner.invoke(cli, ["detect", "--help"])
    assert result.exit_code == 0
    assert "Detect bat calls in files in AUDIO_DIR" in result.output


def test_cli_detect_command_on_test_audio(tmp_path):
    """Test the detect command on test audio."""
    results_dir = tmp_path / "results"

    # Remove results dir if it exists
    if results_dir.exists():
        results_dir.rmdir()

    runner = CliRunner()
    result = runner.invoke(
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


def test_cli_detect_command_with_non_trivial_time_expansion(tmp_path):
    """Test the detect command with a non-trivial time expansion factor."""
    results_dir = tmp_path / "results"

    # Remove results dir if it exists
    if results_dir.exists():
        results_dir.rmdir()

    runner = CliRunner()
    result = runner.invoke(
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
    assert 'Time Expansion Factor: 10' in result.stdout



def test_cli_detect_command_with_the_spec_feature_flag(tmp_path: Path):
    """Test the detect command with the spec feature flag."""
    results_dir = tmp_path / "results"

    # Remove results dir if it exists
    if results_dir.exists():
        results_dir.rmdir()

    runner = CliRunner()
    result = runner.invoke(
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
        "20180627_215323-RHIFER-LR_0_0.5.wav_spec_features.csv"
    ]

    for expected_file in expected_files:
        assert expected_file in csv_files

        df = pd.read_csv(results_dir / expected_file)
        assert not (df.duration == -1).any()
