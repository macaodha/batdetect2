from pathlib import Path
from typing import Optional

import click
from loguru import logger

from batdetect2.cli.base import cli
from batdetect2.evaluate.metrics import (
    ClassificationAccuracy,
    ClassificationMeanAveragePrecision,
    DetectionAveragePrecision,
)
from batdetect2.models import build_model
from batdetect2.models.backbones import load_backbone_config
from batdetect2.postprocess import build_postprocessor, load_postprocess_config
from batdetect2.preprocess import build_preprocessor, load_preprocessing_config
from batdetect2.targets import build_targets, load_target_config
from batdetect2.train import train
from batdetect2.train.callbacks import ValidationMetrics
from batdetect2.train.config import TrainingConfig, load_train_config
from batdetect2.train.dataset import list_preprocessed_files

__all__ = [
    "train_command",
]

DEFAULT_CONFIG_FILE = Path("config.yaml")


@cli.command(name="train")
@click.option(
    "--train-examples",
    type=click.Path(exists=True),
    required=True,
)
@click.option("--val-examples", type=click.Path(exists=True))
@click.option(
    "--model-path",
    type=click.Path(exists=True),
)
@click.option(
    "--train-config",
    type=click.Path(exists=True),
    default=DEFAULT_CONFIG_FILE,
)
@click.option(
    "--train-field",
    type=str,
    default="train",
)
@click.option(
    "--preprocess-config",
    type=click.Path(exists=True),
    help=(
        "Path to the preprocessing configuration file. This file tells "
        "the program how to prepare your audio data before training, such "
        "as resampling or applying filters."
    ),
    default=DEFAULT_CONFIG_FILE,
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
    default="preprocess",
)
@click.option(
    "--target-config",
    type=click.Path(exists=True),
    help=(
        "Path to the training target configuration file. This file "
        "specifies what sounds the model should learn to predict."
    ),
    default=DEFAULT_CONFIG_FILE,
)
@click.option(
    "--target-config-field",
    type=str,
    help=(
        "If the target settings are inside a nested dictionary "
        "within the target configuration file, specify the key here. "
        "If the settings are at the top level, you don't need to specify this."
    ),
    default="targets",
)
@click.option(
    "--postprocess-config",
    type=click.Path(exists=True),
    default=DEFAULT_CONFIG_FILE,
)
@click.option(
    "--postprocess-config-field",
    type=str,
    default="postprocess",
)
@click.option(
    "--model-config",
    type=click.Path(exists=True),
    default=DEFAULT_CONFIG_FILE,
)
@click.option(
    "--model-config-field",
    type=str,
    default="model",
)
def train_command(
    train_examples: Path,
    val_examples: Optional[Path] = None,
    model_path: Optional[Path] = None,
    train_config: Path = DEFAULT_CONFIG_FILE,
    train_config_field: str = "train",
    preprocess_config: Path = DEFAULT_CONFIG_FILE,
    preprocess_config_field: str = "preprocess",
    target_config: Path = DEFAULT_CONFIG_FILE,
    target_config_field: str = "targets",
    postprocess_config: Path = DEFAULT_CONFIG_FILE,
    postprocess_config_field: str = "postprocess",
    model_config: Path = DEFAULT_CONFIG_FILE,
    model_config_field: str = "model",
):
    logger.info("Starting training!")

    try:
        target_config_loaded = load_target_config(
            path=target_config,
            field=target_config_field,
        )
        targets = build_targets(config=target_config_loaded)
        logger.debug(
            "Loaded targets info from config file {path}", path=target_config
        )
    except IOError:
        logger.debug(
            "Could not load target info from config file, using default"
        )
        targets = build_targets()

    try:
        preprocess_config_loaded = load_preprocessing_config(
            path=preprocess_config,
            field=preprocess_config_field,
        )
        preprocessor = build_preprocessor(preprocess_config_loaded)
        logger.debug(
            "Loaded preprocessor from config file {path}", path=target_config
        )

    except IOError:
        logger.debug(
            "Could not load preprocessor from config file, using default"
        )
        preprocessor = build_preprocessor()

    try:
        model_config_loaded = load_backbone_config(
            path=model_config, field=model_config_field
        )
        model = build_model(
            num_classes=len(targets.class_names),
            config=model_config_loaded,
        )
    except IOError:
        model = build_model(num_classes=len(targets.class_names))

    try:
        postprocess_config_loaded = load_postprocess_config(
            path=postprocess_config,
            field=postprocess_config_field,
        )
        postprocessor = build_postprocessor(
            targets=targets,
            config=postprocess_config_loaded,
        )
    except IOError:
        postprocessor = build_postprocessor(targets=targets)

    try:
        train_config_loaded = load_train_config(
            path=train_config, field=train_config_field
        )
    except IOError:
        train_config_loaded = TrainingConfig()

    train_files = list_preprocessed_files(train_examples)

    val_files = (
        None if val_examples is None else list_preprocessed_files(val_examples)
    )

    return train(
        detector=model,
        train_examples=train_files,  # type: ignore
        val_examples=val_files,  # type: ignore
        model_path=model_path,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        targets=targets,
        config=train_config_loaded,
        callbacks=[
            ValidationMetrics(
                metrics=[
                    DetectionAveragePrecision(),
                    ClassificationMeanAveragePrecision(
                        class_names=targets.class_names,
                    ),
                    ClassificationAccuracy(class_names=targets.class_names),
                ]
            )
        ],
    )
