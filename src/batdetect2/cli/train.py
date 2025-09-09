import sys
from pathlib import Path
from typing import Optional

import click
from loguru import logger

from batdetect2.cli.base import cli
from batdetect2.data import load_dataset_from_config
from batdetect2.train import (
    FullTrainingConfig,
    load_full_training_config,
    train,
)

__all__ = ["train_command"]


@cli.command(name="train")
@click.argument("train_dataset", type=click.Path(exists=True))
@click.option("--val-dataset", type=click.Path(exists=True))
@click.option("--model-path", type=click.Path(exists=True))
@click.option("--ckpt-dir", type=click.Path(exists=True))
@click.option("--log-dir", type=click.Path(exists=True))
@click.option("--config", type=click.Path(exists=True))
@click.option("--config-field", type=str)
@click.option("--train-workers", type=int)
@click.option("--val-workers", type=int)
@click.option("--experiment-name", type=str)
@click.option("--run-name", type=str)
@click.option("--seed", type=int)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Increase verbosity. -v for INFO, -vv for DEBUG.",
)
def train_command(
    train_dataset: Path,
    val_dataset: Optional[Path] = None,
    model_path: Optional[Path] = None,
    ckpt_dir: Optional[Path] = None,
    log_dir: Optional[Path] = None,
    config: Optional[Path] = None,
    config_field: Optional[str] = None,
    seed: Optional[int] = None,
    train_workers: int = 0,
    val_workers: int = 0,
    experiment_name: Optional[str] = None,
    run_name: Optional[str] = None,
    verbose: int = 0,
):
    logger.remove()
    if verbose == 0:
        log_level = "WARNING"
    elif verbose == 1:
        log_level = "INFO"
    else:
        log_level = "DEBUG"
    logger.add(sys.stderr, level=log_level)

    logger.info("Initiating training process...")

    logger.info("Loading training configuration...")
    conf = (
        load_full_training_config(config, field=config_field)
        if config is not None
        else FullTrainingConfig()
    )

    logger.info("Loading training dataset...")
    train_annotations = load_dataset_from_config(train_dataset)
    logger.debug(
        "Loaded {num_annotations} training examples",
        num_annotations=len(train_annotations),
    )

    val_annotations = None
    if val_dataset is not None:
        val_annotations = load_dataset_from_config(val_dataset)
        logger.debug(
            "Loaded {num_annotations} validation examples",
            num_annotations=len(val_annotations),
        )
    else:
        logger.debug("No validation directory provided.")

    logger.info("Configuration and data loaded. Starting training...")
    train(
        train_annotations=train_annotations,
        val_annotations=val_annotations,
        config=conf,
        model_path=model_path,
        train_workers=train_workers,
        val_workers=val_workers,
        experiment_name=experiment_name,
        log_dir=log_dir,
        checkpoint_dir=ckpt_dir,
        seed=seed,
        run_name=run_name,
    )
