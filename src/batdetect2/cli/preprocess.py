import sys
from pathlib import Path
from typing import Optional

import click
import yaml
from loguru import logger

from batdetect2.cli.base import cli
from batdetect2.data import load_dataset_from_config
from batdetect2.train.preprocess import (
    TrainPreprocessConfig,
    load_train_preprocessing_config,
    preprocess_dataset,
)

__all__ = ["preprocess"]


@cli.command()
@click.argument(
    "dataset_config",
    type=click.Path(exists=True),
)
@click.argument(
    "output",
    type=click.Path(),
)
@click.option(
    "--dataset-field",
    type=str,
    help=(
        "Specifies the key to access the dataset information within the "
        "dataset configuration file, if the information is nested inside a "
        "dictionary. If the dataset information is at the top level of the "
        "config file, you don't need to specify this."
    ),
)
@click.option(
    "--base-dir",
    type=click.Path(exists=True),
    help=(
        "The main directory where your audio recordings and annotation "
        "files are stored. This helps the program find your data, "
        "especially if the paths in your dataset configuration file "
        "are relative."
    ),
)
@click.option(
    "--config",
    type=click.Path(exists=True),
    help=(
        "Path to the configuration file. This file tells "
        "the program how to prepare your audio data before training, such "
        "as resampling or applying filters."
    ),
)
@click.option(
    "--config-field",
    type=str,
    help=(
        "If the preprocessing settings are inside a nested dictionary "
        "within the preprocessing configuration file, specify the key "
        "here to access them. If the preprocessing settings are at the "
        "top level, you don't need to specify this."
    ),
)
@click.option(
    "--num-workers",
    type=int,
    help=(
        "The maximum number of computer cores to use when processing "
        "your audio data. Using more cores can speed up the preprocessing, "
        "but don't use more than your computer has available. By default, "
        "the program will use all available cores."
    ),
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Increase verbosity. -v for INFO, -vv for DEBUG.",
)
def preprocess(
    dataset_config: Path,
    output: Path,
    base_dir: Optional[Path] = None,
    config: Optional[Path] = None,
    config_field: Optional[str] = None,
    num_workers: Optional[int] = None,
    dataset_field: Optional[str] = None,
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

    logger.info("Starting preprocessing.")

    output = Path(output)
    logger.info("Will save outputs to {output}", output=output)

    base_dir = base_dir or Path.cwd()
    logger.debug("Current working directory: {base_dir}", base_dir=base_dir)

    if config:
        logger.info(
            "Loading preprocessing config from: {config}", config=config
        )

    conf = (
        load_train_preprocessing_config(config, field=config_field)
        if config is not None
        else TrainPreprocessConfig()
    )
    logger.debug(
        "Preprocessing config:\n{conf}",
        conf=yaml.dump(conf.model_dump()),
    )

    dataset = load_dataset_from_config(
        dataset_config,
        field=dataset_field,
        base_dir=base_dir,
    )

    logger.info(
        "Loaded {num_examples} annotated clips from the configured dataset",
        num_examples=len(dataset),
    )

    preprocess_dataset(
        dataset,
        conf,
        output=output,
        max_workers=num_workers,
    )
