"""CLI tests for evaluate command."""

from pathlib import Path

from click.testing import CliRunner

from batdetect2.cli import cli

BASE_DIR = Path(__file__).parent.parent.parent


def test_cli_evaluate_help() -> None:
    """User story: inspect evaluate command interface and options."""

    result = CliRunner().invoke(cli, ["evaluate", "--help"])

    assert result.exit_code == 0
    assert "MODEL_PATH" in result.output
    assert "TEST_DATASET" in result.output
    assert "--evaluation-config" in result.output


def test_cli_evaluate_writes_metrics_for_small_dataset(
    tmp_path: Path,
    tiny_checkpoint_path: Path,
) -> None:
    """User story: evaluate a checkpoint and get metrics artifacts."""

    output_dir = tmp_path / "eval_out"

    result = CliRunner().invoke(
        cli,
        [
            "evaluate",
            str(tiny_checkpoint_path),
            str(BASE_DIR / "example_data" / "dataset.yaml"),
            "--base-dir",
            str(BASE_DIR),
            "--workers",
            "0",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0
    assert len(list(output_dir.rglob("metrics.csv"))) >= 1
