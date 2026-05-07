"""CLI tests for finetune command."""

from pathlib import Path
from types import SimpleNamespace

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


def test_cli_finetune_defaults_to_bundled_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """User story: finetune can use the bundled checkpoint by default."""

    called = {}

    class FakeAPI:
        def finetune(self, **kwargs):
            called["finetune"] = kwargs
            return None

    class FakeBatDetect2API:
        @classmethod
        def from_checkpoint(cls, path=None, **kwargs):
            called["path"] = path
            called["from_checkpoint_kwargs"] = kwargs
            return FakeAPI()

    monkeypatch.setattr(
        "batdetect2.api_v2.BatDetect2API",
        FakeBatDetect2API,
    )
    monkeypatch.setattr(
        "batdetect2.data.load_dataset_config",
        lambda path: SimpleNamespace(path=path),
    )
    monkeypatch.setattr(
        "batdetect2.data.load_dataset",
        lambda config, base_dir=None: [],
    )
    monkeypatch.setattr(
        "batdetect2.targets.TargetConfig.load",
        lambda path: SimpleNamespace(path=path),
    )

    result = CliRunner().invoke(
        cli,
        [
            "finetune",
            "example_data/dataset.yaml",
            "--targets",
            "example_data/targets.yaml",
        ],
    )

    assert result.exit_code == 0
    assert called["path"] is None
    assert "finetune" in called


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
