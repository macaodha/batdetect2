"""Test the command line interface."""

import shutil
from pathlib import Path

import lightning as L
import pandas as pd
from click.testing import CliRunner

from batdetect2.cli import cli
from batdetect2.config import BatDetect2Config
from batdetect2.train.lightning import build_training_module

runner = CliRunner()


def test_cli_base_command():
    """Test the base command."""
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert (
        "BatDetect2 - Bat Call Detection and Classification" in result.output
    )


def test_cli_detect_command_help():
    """Test the detect command help."""
    result = runner.invoke(cli, ["detect", "--help"])
    assert result.exit_code == 0
    assert "Detect bat calls in files in AUDIO_DIR" in result.output


def test_cli_predict_command_help():
    """Test the predict command help."""
    result = runner.invoke(cli, ["predict", "--help"])
    assert result.exit_code == 0
    assert "directory" in result.output
    assert "file_list" in result.output
    assert "dataset" in result.output


def test_cli_predict_directory_runs_on_real_audio(tmp_path: Path):
    """User story: run prediction from CLI on a small directory."""

    source_audio = Path("example_data/audio")
    source_file = next(source_audio.glob("*.wav"))
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    target_file = audio_dir / source_file.name
    shutil.copy(source_file, target_file)

    module = build_training_module(model_config=BatDetect2Config().model)
    trainer = L.Trainer(enable_checkpointing=False, logger=False)
    model_path = tmp_path / "model.ckpt"
    trainer.strategy.connect(module)
    trainer.save_checkpoint(model_path)
    output_path = tmp_path / "predictions"

    result = runner.invoke(
        cli,
        [
            "predict",
            "directory",
            str(model_path),
            str(audio_dir),
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
    output_files = list(output_path.glob("*.json"))
    assert len(output_files) == 1


def test_cli_detect_command_on_test_audio(tmp_path):
    """Test the detect command on test audio."""
    results_dir = tmp_path / "results"

    # Remove results dir if it exists
    if results_dir.exists():
        results_dir.rmdir()

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
    assert "Time Expansion Factor: 10" in result.stdout


def test_cli_detect_command_with_the_spec_feature_flag(tmp_path: Path):
    """Test the detect command with the spec feature flag."""
    results_dir = tmp_path / "results"

    # Remove results dir if it exists
    if results_dir.exists():
        results_dir.rmdir()

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
        "20180627_215323-RHIFER-LR_0_0.5.wav_spec_features.csv",
    ]

    for expected_file in expected_files:
        assert expected_file in csv_files

        df = pd.read_csv(results_dir / expected_file)
        assert not (df.duration == -1).any()


def test_cli_detect_fails_gracefully_on_empty_file(tmp_path: Path):
    results_dir = tmp_path / "results"
    target = tmp_path / "audio"
    target.mkdir()

    # Create an empty file with the .wav extension
    empty_file = target / "empty.wav"
    empty_file.touch()

    result = runner.invoke(
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
