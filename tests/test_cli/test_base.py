"""Behavior-focused tests for top-level CLI command discovery."""

from click.testing import CliRunner

from batdetect2.cli import cli


def test_cli_base_help_lists_main_commands() -> None:
    """User story: discover available workflows from top-level help."""

    result = CliRunner().invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "process" in result.output
    assert "train" in result.output
    assert "evaluate" in result.output
    assert "data" in result.output
    assert "detect" in result.output
