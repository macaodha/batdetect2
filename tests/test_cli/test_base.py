"""Behavior-focused tests for top-level CLI command discovery."""

from pathlib import Path

from click.testing import CliRunner

from batdetect2.cli import cli


BASE_DIR = Path(__file__).parent.parent.parent


def test_cli_base_help_lists_main_commands() -> None:
    """User story: discover available workflows from top-level help."""

    result = CliRunner().invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "--log-file" in result.output
    assert "process" in result.output
    assert "train" in result.output
    assert "evaluate" in result.output
    assert "data" in result.output
    assert "detect" in result.output


def test_cli_writes_logs_to_file_and_terminal(
    tmp_path: Path,
    tiny_checkpoint_path: Path,
) -> None:
    """User story: save CLI logs to a file while keeping terminal output."""

    output_dir = tmp_path / "eval_out"
    log_file = tmp_path / "logs" / "cli.log"

    result = CliRunner().invoke(
        cli,
        [
            "--log-file",
            str(log_file),
            "-v",
            "evaluate",
            str(BASE_DIR / "example_data" / "dataset.yaml"),
            "--model",
            str(tiny_checkpoint_path),
            "--base-dir",
            str(BASE_DIR),
            "--workers",
            "0",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0
    assert "Initiating evaluation process..." in result.output
    assert log_file.exists()
    assert "Initiating evaluation process..." in log_file.read_text()
