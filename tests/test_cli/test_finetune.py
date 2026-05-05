"""CLI tests for finetune command."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from batdetect2.cli import cli


def test_cli_finetune_help() -> None:
    """User story: inspect finetune command interface and options."""

    result = CliRunner().invoke(cli, ["finetune", "--help"])

    assert result.exit_code == 0
    assert "TRAIN_DATASET" in result.output
    assert "--model" in result.output
    assert "--targets" in result.output
    assert "--training-config" in result.output
    assert "--audio-config" in result.output
    assert "--logging-config" in result.output
    assert "--evaluation-config" not in result.output
    assert "--inference-config" not in result.output
    assert "--outputs-config" not in result.output


def test_cli_finetune_requires_model() -> None:
    """User story: finetune requires a checkpoint argument."""

    result = CliRunner().invoke(
        cli,
        [
            "finetune",
            "example_data/dataset.yaml",
            "--targets",
            "example_data/targets.yaml",
        ],
    )

    assert result.exit_code != 0
    assert "--model" in result.output


def test_cli_finetune_requires_targets(tiny_checkpoint_path: Path) -> None:
    """User story: finetune requires a new target definition."""

    result = CliRunner().invoke(
        cli,
        [
            "finetune",
            "example_data/dataset.yaml",
            "--model",
            str(tiny_checkpoint_path),
        ],
    )

    assert result.exit_code != 0
    assert "--targets" in result.output


@pytest.mark.slow
def test_cli_finetune_from_checkpoint_runs_on_small_dataset(
    tmp_path: Path,
    tiny_checkpoint_path: Path,
) -> None:
    """User story: fine-tune a checkpoint via CLI with new targets."""

    ckpt_dir = tmp_path / "checkpoints"
    log_dir = tmp_path / "logs"
    ckpt_dir.mkdir()
    log_dir.mkdir()

    result = CliRunner().invoke(
        cli,
        [
            "finetune",
            "example_data/dataset.yaml",
            "--val-dataset",
            "example_data/dataset.yaml",
            "--model",
            str(tiny_checkpoint_path),
            "--targets",
            "example_data/targets.yaml",
            "--num-epochs",
            "1",
            "--train-workers",
            "0",
            "--val-workers",
            "0",
            "--ckpt-dir",
            str(ckpt_dir),
            "--log-dir",
            str(log_dir),
        ],
    )

    assert result.exit_code == 0
    assert len(list(ckpt_dir.rglob("*.ckpt"))) >= 1
