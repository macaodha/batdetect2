from pathlib import Path
from typing import Optional

import click
from loguru import logger

from batdetect2.cli.base import cli
from batdetect2.train import (
    FullTrainingConfig,
    load_full_training_config,
    train,
)
from batdetect2.train.dataset import list_preprocessed_files

__all__ = [
    "train_command",
]


@cli.command(name="train")
@click.argument("train_dir", type=click.Path(exists=True))
@click.option("--val-dir", type=click.Path(exists=True))
@click.option("--model-path", type=click.Path(exists=True))
@click.option("--config", type=click.Path(exists=True))
@click.option("--config-field", type=str)
@click.option("--train-workers", type=int, default=0)
@click.option("--val-workers", type=int, default=0)
def train_command(
    train_dir: Path,
    val_dir: Optional[Path] = None,
    model_path: Optional[Path] = None,
    config: Optional[Path] = None,
    config_field: Optional[str] = None,
    train_workers: int = 0,
    val_workers: int = 0,
):
    logger.info("Starting training!")

    conf = (
        load_full_training_config(config, field=config_field)
        if config is not None
        else FullTrainingConfig()
    )

    train_examples = list_preprocessed_files(train_dir)
    val_examples = (
        list_preprocessed_files(val_dir) if val_dir is not None else None
    )

    train(
        train_examples=train_examples,
        val_examples=val_examples,
        config=conf,
        model_path=model_path,
        train_workers=train_workers,
        val_workers=val_workers,
    )
