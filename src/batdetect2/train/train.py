from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from lightning import Trainer, seed_everything
from loguru import logger
from soundevent import data

from batdetect2.audio import AudioConfig, build_audio_loader
from batdetect2.evaluate import build_evaluator
from batdetect2.logging import build_logger
from batdetect2.models import ModelConfig
from batdetect2.preprocess import build_preprocessor
from batdetect2.targets import build_targets
from batdetect2.train import TrainingConfig
from batdetect2.train.callbacks import ValidationMetrics
from batdetect2.train.checkpoints import build_checkpoint_callback
from batdetect2.train.dataset import build_train_loader, build_val_loader
from batdetect2.train.labels import build_clip_labeler
from batdetect2.train.lightning import build_training_module

if TYPE_CHECKING:
    from batdetect2.typing import (
        AudioLoader,
        ClipLabeller,
        EvaluatorProtocol,
        PreprocessorProtocol,
        TargetProtocol,
    )

__all__ = [
    "build_trainer",
    "run_train",
]


def run_train(
    train_annotations: Sequence[data.ClipAnnotation],
    val_annotations: Sequence[data.ClipAnnotation] | None = None,
    targets: Optional["TargetProtocol"] = None,
    preprocessor: Optional["PreprocessorProtocol"] = None,
    audio_loader: Optional["AudioLoader"] = None,
    labeller: Optional["ClipLabeller"] = None,
    audio_config: Optional[AudioConfig] = None,
    model_config: Optional[ModelConfig] = None,
    train_config: Optional[TrainingConfig] = None,
    trainer: Trainer | None = None,
    train_workers: int | None = None,
    val_workers: int | None = None,
    checkpoint_dir: Path | None = None,
    log_dir: Path | None = None,
    experiment_name: str | None = None,
    num_epochs: int | None = None,
    run_name: str | None = None,
    seed: int | None = None,
):
    if seed is not None:
        seed_everything(seed)

    model_config = model_config or ModelConfig()
    audio_config = audio_config or AudioConfig()
    train_config = train_config or TrainingConfig()

    targets = targets or build_targets(config=model_config.targets)

    audio_loader = audio_loader or build_audio_loader(config=audio_config)

    preprocessor = preprocessor or build_preprocessor(
        input_samplerate=audio_loader.samplerate,
        config=model_config.preprocess,
    )

    labeller = labeller or build_clip_labeler(
        targets,
        min_freq=preprocessor.min_freq,
        max_freq=preprocessor.max_freq,
        config=train_config.labels,
    )

    train_dataloader = build_train_loader(
        train_annotations,
        audio_loader=audio_loader,
        labeller=labeller,
        preprocessor=preprocessor,
        config=train_config.train_loader,
        num_workers=train_workers,
    )

    val_dataloader = (
        build_val_loader(
            val_annotations,
            audio_loader=audio_loader,
            labeller=labeller,
            preprocessor=preprocessor,
            config=train_config.val_loader,
            num_workers=val_workers,
        )
        if val_annotations is not None
        else None
    )

    module = build_training_module(
        model_config=model_config,
        train_config=train_config,
    )

    trainer = trainer or build_trainer(
        train_config,
        evaluator=build_evaluator(
            train_config.validation,
            targets=targets,
        ),
        checkpoint_dir=checkpoint_dir,
        num_epochs=num_epochs,
        log_dir=log_dir,
        experiment_name=experiment_name,
        run_name=run_name,
    )

    logger.info("Starting main training loop...")
    trainer.fit(
        module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    logger.info("Training complete.")

    return module


def build_trainer(
    config: TrainingConfig,
    evaluator: "EvaluatorProtocol",
    checkpoint_dir: Path | None = None,
    log_dir: Path | None = None,
    experiment_name: str | None = None,
    run_name: str | None = None,
    num_epochs: int | None = None,
) -> Trainer:
    trainer_conf = config.trainer
    logger.opt(lazy=True).debug(
        "Building trainer with config: \n{config}",
        config=lambda: trainer_conf.to_yaml_string(exclude_none=True),
    )

    train_logger = build_logger(
        config.logger,
        log_dir=log_dir,
        experiment_name=experiment_name,
        run_name=run_name,
    )

    train_logger.log_hyperparams(
        config.model_dump(
            mode="json",
            exclude_none=True,
        )
    )

    train_config = trainer_conf.model_dump(exclude_none=True)

    if num_epochs is not None:
        train_config["max_epochs"] = num_epochs

    return Trainer(
        **train_config,
        logger=train_logger,
        callbacks=[
            build_checkpoint_callback(
                config=config.checkpoints,
                checkpoint_dir=checkpoint_dir,
                experiment_name=experiment_name,
                run_name=run_name,
            ),
            ValidationMetrics(evaluator),
        ],
    )
