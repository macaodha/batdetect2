from pathlib import Path

import click
from loguru import logger

from batdetect2.cli.base import cli

__all__ = ["train_command"]


@cli.command(name="train")
@click.argument("train_dataset", type=click.Path(exists=True))
@click.option("--val-dataset", type=click.Path(exists=True))
@click.option("--model", "model_path", type=click.Path(exists=True))
@click.option("--targets", "targets_config", type=click.Path(exists=True))
@click.option("--model-config", type=click.Path(exists=True))
@click.option("--training-config", type=click.Path(exists=True))
@click.option("--audio-config", type=click.Path(exists=True))
@click.option("--evaluation-config", type=click.Path(exists=True))
@click.option("--inference-config", type=click.Path(exists=True))
@click.option("--outputs-config", type=click.Path(exists=True))
@click.option("--logging-config", type=click.Path(exists=True))
@click.option("--ckpt-dir", type=click.Path(exists=True))
@click.option("--log-dir", type=click.Path(exists=True))
@click.option("--train-workers", type=int)
@click.option("--val-workers", type=int)
@click.option("--num-epochs", type=int)
@click.option("--experiment-name", type=str)
@click.option("--run-name", type=str)
@click.option("--seed", type=int)
def train_command(
    train_dataset: Path,
    val_dataset: Path | None = None,
    model_path: Path | None = None,
    ckpt_dir: Path | None = None,
    log_dir: Path | None = None,
    targets_config: Path | None = None,
    model_config: Path | None = None,
    training_config: Path | None = None,
    audio_config: Path | None = None,
    evaluation_config: Path | None = None,
    inference_config: Path | None = None,
    outputs_config: Path | None = None,
    logging_config: Path | None = None,
    seed: int | None = None,
    num_epochs: int | None = None,
    train_workers: int = 0,
    val_workers: int = 0,
    experiment_name: str | None = None,
    run_name: str | None = None,
):
    from batdetect2.api_v2 import BatDetect2API
    from batdetect2.audio import AudioConfig
    from batdetect2.config import BatDetect2Config
    from batdetect2.data import load_dataset_from_config
    from batdetect2.evaluate import EvaluationConfig
    from batdetect2.inference import InferenceConfig
    from batdetect2.logging import AppLoggingConfig
    from batdetect2.models import ModelConfig
    from batdetect2.outputs import OutputsConfig
    from batdetect2.targets import TargetConfig
    from batdetect2.train import TrainingConfig

    logger.info("Initiating training process...")

    logger.info("Loading configuration...")
    target_conf = (
        TargetConfig.load(targets_config)
        if targets_config is not None
        else None
    )
    model_conf = (
        ModelConfig.load(model_config) if model_config is not None else None
    )
    train_conf = (
        TrainingConfig.load(training_config)
        if training_config is not None
        else None
    )
    audio_conf = (
        AudioConfig.load(audio_config) if audio_config is not None else None
    )
    eval_conf = (
        EvaluationConfig.load(evaluation_config)
        if evaluation_config is not None
        else None
    )
    inference_conf = (
        InferenceConfig.load(inference_config)
        if inference_config is not None
        else None
    )
    outputs_conf = (
        OutputsConfig.load(outputs_config)
        if outputs_config is not None
        else None
    )
    logging_conf = (
        AppLoggingConfig.load(logging_config)
        if logging_config is not None
        else None
    )

    if target_conf is not None:
        logger.info("Loaded targets configuration.")

    if model_conf is not None and target_conf is not None:
        model_conf = model_conf.model_copy(update={"targets": target_conf})

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

    if model_path is not None and model_conf is not None:
        raise click.UsageError(
            "--model-config cannot be used with --model. "
            "Checkpoint model configuration is loaded from the checkpoint."
        )

    if model_path is None:
        conf = BatDetect2Config()
        if model_conf is not None:
            conf.model = model_conf
        elif target_conf is not None:
            conf.model = conf.model.model_copy(update={"targets": target_conf})

        if train_conf is not None:
            conf.train = train_conf
        if audio_conf is not None:
            conf.audio = audio_conf
        if eval_conf is not None:
            conf.evaluation = eval_conf
        if inference_conf is not None:
            conf.inference = inference_conf
        if outputs_conf is not None:
            conf.outputs = outputs_conf
        if logging_conf is not None:
            conf.logging = logging_conf

        api = BatDetect2API.from_config(conf)
    else:
        api = BatDetect2API.from_checkpoint(
            model_path,
            targets_config=target_conf,
            train_config=train_conf,
            audio_config=audio_conf,
            evaluation_config=eval_conf,
            inference_config=inference_conf,
            outputs_config=outputs_conf,
            logging_config=logging_conf,
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
