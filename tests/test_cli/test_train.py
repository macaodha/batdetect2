"""CLI tests for train command."""

from pathlib import Path

from click.testing import CliRunner

from batdetect2.cli import cli
from batdetect2.models import ModelConfig


def test_cli_train_help() -> None:
    """User story: inspect train command interface and options."""

    result = CliRunner().invoke(cli, ["train", "--help"])

    assert result.exit_code == 0
    assert "TRAIN_DATASET" in result.output
    assert "--training-config" in result.output
    assert "--model" in result.output


def test_cli_train_from_checkpoint_runs_on_small_dataset(
    tmp_path: Path,
    tiny_checkpoint_path: Path,
) -> None:
    """User story: continue training from checkpoint via CLI."""

    ckpt_dir = tmp_path / "checkpoints"
    log_dir = tmp_path / "logs"
    ckpt_dir.mkdir()
    log_dir.mkdir()

    result = CliRunner().invoke(
        cli,
        [
            "train",
            "example_data/dataset.yaml",
            "--val-dataset",
            "example_data/dataset.yaml",
            "--model",
            str(tiny_checkpoint_path),
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


def test_cli_train_rejects_model_and_model_config_together(
    tmp_path: Path,
    tiny_checkpoint_path: Path,
) -> None:
    """User story: invalid train flags fail with clear usage error."""

    model_config_path = tmp_path / "model.yaml"
    model_config_path.write_text(ModelConfig().to_yaml_string())

    result = CliRunner().invoke(
        cli,
        [
            "train",
            "example_data/dataset.yaml",
            "--model",
            str(tiny_checkpoint_path),
            "--model-config",
            str(model_config_path),
        ],
    )

    assert result.exit_code != 0
    assert "--model-config cannot be used with --model" in result.output
