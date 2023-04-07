"""Test the command line interface."""
from click.testing import CliRunner

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
