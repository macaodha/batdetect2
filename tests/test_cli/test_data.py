"""Behavior tests for data CLI command group."""

from pathlib import Path

from click.testing import CliRunner

from batdetect2.cli import cli


def test_cli_data_help() -> None:
    """User story: discover data subcommands."""

    result = CliRunner().invoke(cli, ["data", "--help"])

    assert result.exit_code == 0
    assert "summary" in result.output
    assert "convert" in result.output


def test_cli_data_convert_creates_annotation_set(tmp_path: Path) -> None:
    """User story: convert dataset config into a soundevent annotation set."""

    output = tmp_path / "annotations.json"

    result = CliRunner().invoke(
        cli,
        [
            "data",
            "convert",
            "example_data/dataset.yaml",
            "--base-dir",
            ".",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert output.exists()


def test_cli_data_convert_fails_with_invalid_field(tmp_path: Path) -> None:
    """User story: invalid nested field in dataset config fails clearly."""

    output = tmp_path / "annotations.json"

    result = CliRunner().invoke(
        cli,
        [
            "data",
            "convert",
            "example_data/dataset.yaml",
            "--field",
            "does.not.exist",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code != 0
