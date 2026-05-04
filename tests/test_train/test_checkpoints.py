from pathlib import Path

import pytest
from soundevent import data

from batdetect2.train import TrainingConfig, run_train

pytestmark = pytest.mark.slow


def _build_fast_train_config() -> TrainingConfig:
    config = TrainingConfig()
    config.trainer.limit_train_batches = 1
    config.trainer.limit_val_batches = 1
    config.trainer.log_every_n_steps = 1
    config.trainer.check_val_every_n_epoch = 1
    config.train_loader.batch_size = 1
    config.train_loader.augmentations.enabled = False
    return config


def test_train_saves_checkpoint_in_requested_experiment_run_dir(
    tmp_path: Path,
    example_annotations: list[data.ClipAnnotation],
) -> None:
    config = _build_fast_train_config()

    run_train(
        train_annotations=example_annotations[:1],
        val_annotations=example_annotations[:1],
        train_config=config,
        num_epochs=1,
        train_workers=0,
        val_workers=0,
        checkpoint_dir=tmp_path,
        experiment_name="exp_a",
        run_name="run_b",
        seed=0,
    )

    checkpoints = list((tmp_path / "exp_a" / "run_b").rglob("*.ckpt"))
    assert checkpoints


def test_train_without_validation_can_still_save_last_checkpoint(
    tmp_path: Path,
    example_annotations: list[data.ClipAnnotation],
) -> None:
    config = _build_fast_train_config()
    config.checkpoints.save_last = True

    run_train(
        train_annotations=example_annotations[:1],
        val_annotations=None,
        train_config=config,
        num_epochs=1,
        train_workers=0,
        val_workers=0,
        checkpoint_dir=tmp_path,
        seed=0,
    )

    assert list(tmp_path.rglob("last*.ckpt"))


def test_train_controls_which_checkpoints_are_kept(
    tmp_path: Path,
    example_annotations: list[data.ClipAnnotation],
) -> None:
    config = _build_fast_train_config()
    config.checkpoints.save_top_k = 1
    config.checkpoints.save_last = True
    config.checkpoints.filename = "epoch{epoch}"

    run_train(
        train_annotations=example_annotations[:1],
        val_annotations=example_annotations[:1],
        train_config=config,
        num_epochs=3,
        train_workers=0,
        val_workers=0,
        checkpoint_dir=tmp_path,
        seed=0,
    )

    all_checkpoints = list(tmp_path.rglob("*.ckpt"))
    last_checkpoints = list(tmp_path.rglob("last*.ckpt"))
    best_checkpoints = [
        path for path in all_checkpoints if not path.name.startswith("last")
    ]

    assert last_checkpoints
    assert len(best_checkpoints) == 1
    assert "epoch" in best_checkpoints[0].name
