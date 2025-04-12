from pathlib import Path
from typing import Optional

import click

from batdetect2.cli.base import cli
from batdetect2.data import load_dataset_from_config
from batdetect2.preprocess import (
    load_preprocessing_config,
)
from batdetect2.targets import (
    load_label_config,
    load_target_config,
)
from batdetect2.train import (
    preprocess_annotations,
)

__all__ = ["train"]


@cli.group()
def train(): ...


@train.command()
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
    "--preprocess-config",
    type=click.Path(exists=True),
    help=(
        "Path to the preprocessing configuration file. This file tells "
        "the program how to prepare your audio data before training, such "
        "as resampling or applying filters."
    ),
)
@click.option(
    "--preprocess-config-field",
    type=str,
    help=(
        "If the preprocessing settings are inside a nested dictionary "
        "within the preprocessing configuration file, specify the key "
        "here to access them. If the preprocessing settings are at the "
        "top level, you don't need to specify this."
    ),
)
@click.option(
    "--label-config",
    type=click.Path(exists=True),
    help=(
        "Path to the label generation configuration file. This file "
        "contains settings for how to create labels from your "
        "annotations, which the model uses to learn."
    ),
)
@click.option(
    "--label-config-field",
    type=str,
    help=(
        "If the label generation settings are inside a nested dictionary "
        "within the label configuration file, specify the key here. If "
        "the settings are at the top level, leave this blank."
    ),
)
@click.option(
    "--target-config",
    type=click.Path(exists=True),
    help=(
        "Path to the training target configuration file. This file "
        "specifies what sounds the model should learn to predict."
    ),
)
@click.option(
    "--target-config-field",
    type=str,
    help=(
        "If the target settings are inside a nested dictionary "
        "within the target configuration file, specify the key here. "
        "If the settings are at the top level, you don't need to specify this."
    ),
)
@click.option(
    "--force",
    is_flag=True,
    help=(
        "If a preprocessed file already exists, this option tells the "
        "program to overwrite it with the new preprocessed data. Use "
        "this if you want to re-do the preprocessing even if the files "
        "already exist."
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
def preprocess(
    dataset_config: Path,
    output: Path,
    base_dir: Optional[Path] = None,
    preprocess_config: Optional[Path] = None,
    target_config: Optional[Path] = None,
    label_config: Optional[Path] = None,
    force: bool = False,
    num_workers: Optional[int] = None,
    target_config_field: Optional[str] = None,
    preprocess_config_field: Optional[str] = None,
    label_config_field: Optional[str] = None,
    dataset_field: Optional[str] = None,
):
    output = Path(output)
    base_dir = base_dir or Path.cwd()

    preprocess = (
        load_preprocessing_config(
            preprocess_config,
            field=preprocess_config_field,
        )
        if preprocess_config
        else None
    )

    target = (
        load_target_config(
            target_config,
            field=target_config_field,
        )
        if target_config
        else None
    )

    label = (
        load_label_config(
            label_config,
            field=label_config_field,
        )
        if label_config
        else None
    )

    dataset = load_dataset_from_config(
        dataset_config,
        field=dataset_field,
        base_dir=base_dir,
    )

    if not output.exists():
        output.mkdir(parents=True)

    preprocess_annotations(
        dataset.clip_annotations,
        output_dir=output,
        replace=force,
        preprocessing_config=preprocess,
        label_config=label,
        target_config=target,
        max_workers=num_workers,
    )
