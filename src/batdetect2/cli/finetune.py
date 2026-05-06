from pathlib import Path
from typing import Literal, cast

import click
from loguru import logger

from batdetect2.cli.base import cli
from batdetect2.train.checkpoints import DEFAULT_BUNDLED_CHECKPOINT

__all__ = ["finetune_command"]


@cli.command(
    name="finetune", short_help="Fine-tune a checkpoint on new targets."
)
@click.argument("train_dataset", type=click.Path(exists=True))
@click.option(
    "--model",
    "model_path",
    type=str,
    help=(
        "Path to a checkpoint, bundled checkpoint alias, or a Hugging Face "
        "URI to fine-tune from. Defaults to "
        f"'{DEFAULT_BUNDLED_CHECKPOINT}'."
    ),
)
@click.option(
    "--targets",
    "targets_config",
    required=True,
    type=click.Path(exists=True),
    help="Path to the new targets config file.",
)
@click.option(
    "--val-dataset",
    type=click.Path(exists=True),
    help="Path to validation dataset config file.",
)
@click.option(
    "--base-dir",
    type=click.Path(exists=True),
    help=(
        "Base directory used to resolve relative paths inside the training "
        "and validation dataset configs."
    ),
)
@click.option(
    "--training-config",
    type=click.Path(exists=True),
    help="Path to training config file.",
)
@click.option(
    "--audio-config",
    type=click.Path(exists=True),
    help="Path to audio config file.",
)
@click.option(
    "--logging-config",
    type=click.Path(exists=True),
    help="Path to logging config file.",
)
@click.option(
    "--trainable",
    type=click.Choice(["all", "heads", "classifier_head", "bbox_head"]),
    default="heads",
    show_default=True,
    help="Which model parameters remain trainable during fine-tuning.",
)
@click.option(
    "--ckpt-dir",
    type=click.Path(exists=True),
    help="Directory where checkpoints are saved.",
)
@click.option(
    "--log-dir",
    type=click.Path(exists=True),
    help="Directory where logs are written.",
)
@click.option(
    "--train-workers",
    type=int,
    default=0,
    help="Number of worker processes for training data loading.",
)
@click.option(
    "--val-workers",
    type=int,
    default=0,
    help="Number of worker processes for validation data loading.",
)
@click.option(
    "--num-epochs",
    type=int,
    help="Maximum number of training epochs.",
)
@click.option(
    "--experiment-name",
    type=str,
    help="Experiment name used for logging backends.",
)
@click.option(
    "--run-name",
    type=str,
    help="Run name used for logging backends.",
)
@click.option(
    "--seed",
    type=int,
    help="Random seed used for reproducibility.",
)
def finetune_command(
    train_dataset: Path,
    model_path: str | None,
    targets_config: Path,
    val_dataset: Path | None = None,
    ckpt_dir: Path | None = None,
    log_dir: Path | None = None,
    base_dir: Path | None = None,
    training_config: Path | None = None,
    audio_config: Path | None = None,
    logging_config: Path | None = None,
    trainable: str = "heads",
    seed: int | None = None,
    num_epochs: int | None = None,
    train_workers: int = 0,
    val_workers: int = 0,
    experiment_name: str | None = None,
    run_name: str | None = None,
):
    """Fine-tune a BatDetect2 checkpoint on a new target definition."""
    from batdetect2.api_v2 import BatDetect2API
    from batdetect2.audio import AudioConfig
    from batdetect2.data import load_dataset, load_dataset_config
    from batdetect2.logging import AppLoggingConfig
    from batdetect2.targets import TargetConfig
    from batdetect2.train import TrainingConfig
    from batdetect2.train.logging import (
        DatasetConfigArtifact,
        DatasetConfigArtifactLogging,
    )

    logger.info("Initiating fine-tuning process...")

    target_conf = TargetConfig.load(targets_config)
    train_conf = (
        TrainingConfig.load(training_config)
        if training_config is not None
        else None
    )
    audio_conf = (
        AudioConfig.load(audio_config) if audio_config is not None else None
    )
    logging_conf = (
        AppLoggingConfig.load(logging_config)
        if logging_config is not None
        else None
    )

    train_dataset_conf = load_dataset_config(train_dataset)
    train_annotations = load_dataset(train_dataset_conf, base_dir=base_dir)

    val_dataset_conf = (
        load_dataset_config(val_dataset) if val_dataset else None
    )
    val_annotations = (
        load_dataset(val_dataset_conf, base_dir=base_dir)
        if val_dataset_conf
        else None
    )

    logging_callbacks = [
        DatasetConfigArtifactLogging(
            train_dataset_config=DatasetConfigArtifact(
                filename="train_dataset.yaml",
                config=train_dataset_conf,
            ),
            val_dataset_config=(
                DatasetConfigArtifact(
                    filename="val_dataset.yaml",
                    config=val_dataset_conf,
                )
                if val_dataset_conf
                else None
            ),
        )
    ]

    api = BatDetect2API.from_checkpoint(
        model_path,
        train_config=train_conf,
        audio_config=audio_conf,
        logging_config=logging_conf,
    )

    return api.finetune(
        train_annotations=train_annotations,
        val_annotations=val_annotations,
        targets_config=target_conf,
        trainable=cast(
            Literal["all", "heads", "classifier_head", "bbox_head"],
            trainable,
        ),
        train_workers=train_workers,
        val_workers=val_workers,
        checkpoint_dir=ckpt_dir,
        log_dir=log_dir,
        experiment_name=experiment_name,
        num_epochs=num_epochs,
        run_name=run_name,
        seed=seed,
        train_config=train_conf,
        audio_config=audio_conf,
        logger_config=logging_conf.train if logging_conf is not None else None,
        logging_callbacks=logging_callbacks,
    )
