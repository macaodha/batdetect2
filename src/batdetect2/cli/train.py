from pathlib import Path
from typing import Optional

import click
from loguru import logger

from batdetect2.cli.base import cli

__all__ = ["train_command"]


@cli.command(name="train")
@click.argument("train_dataset", type=click.Path(exists=True))
@click.option("--val-dataset", type=click.Path(exists=True))
@click.option("--model", "model_path", type=click.Path(exists=True))
@click.option("--targets", "targets_config", type=click.Path(exists=True))
@click.option("--ckpt-dir", type=click.Path(exists=True))
@click.option("--log-dir", type=click.Path(exists=True))
@click.option("--config", type=click.Path(exists=True))
@click.option("--config-field", type=str)
@click.option("--train-workers", type=int)
@click.option("--val-workers", type=int)
@click.option("--num-epochs", type=int)
@click.option("--experiment-name", type=str)
@click.option("--run-name", type=str)
@click.option("--seed", type=int)
def train_command(
    train_dataset: Path,
    val_dataset: Optional[Path] = None,
    model_path: Optional[Path] = None,
    ckpt_dir: Optional[Path] = None,
    log_dir: Optional[Path] = None,
    config: Optional[Path] = None,
    targets_config: Optional[Path] = None,
    config_field: Optional[str] = None,
    seed: Optional[int] = None,
    num_epochs: Optional[int] = None,
    train_workers: int = 0,
    val_workers: int = 0,
    experiment_name: Optional[str] = None,
    run_name: Optional[str] = None,
):
    from batdetect2.api_v2 import BatDetect2API
    from batdetect2.config import (
        BatDetect2Config,
        load_full_config,
    )
    from batdetect2.data import load_dataset_from_config
    from batdetect2.targets import load_target_config

    logger.info("Initiating training process...")

    logger.info("Loading configuration...")
    conf = (
        load_full_config(config, field=config_field)
        if config is not None
        else BatDetect2Config()
    )

    if targets_config is not None:
        logger.info("Loading targets configuration...")
        conf = conf.model_copy(
            update=dict(targets=load_target_config(targets_config))
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

    if model_path is None:
        api = BatDetect2API.from_config(conf)
    else:
        api = BatDetect2API.from_checkpoint(
            model_path,
            config=conf if config is not None else None,
        )

    return api.train(
        train_annotations=train_annotations,
        val_annotations=val_annotations,
        train_workers=train_workers,
        val_workers=val_workers,
        checkpoint_dir=ckpt_dir,
        log_dir=log_dir,
        num_epochs=num_epochs,
        experiment_name=experiment_name,
        run_name=run_name,
        seed=seed,
    )
