from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from lightning import Trainer, seed_everything
from loguru import logger
from soundevent import data

from batdetect2.audio import build_audio_loader
from batdetect2.evaluate import build_evaluator
from batdetect2.logging import build_logger
from batdetect2.preprocess import build_preprocessor
from batdetect2.targets import build_targets
from batdetect2.train.callbacks import ValidationMetrics
from batdetect2.train.checkpoints import build_checkpoint_callback
from batdetect2.train.dataset import build_train_loader, build_val_loader
from batdetect2.train.labels import build_clip_labeler
from batdetect2.train.lightning import build_training_module

if TYPE_CHECKING:
    from batdetect2.config import BatDetect2Config
    from batdetect2.typing import (
        AudioLoader,
        ClipLabeller,
        EvaluatorProtocol,
        PreprocessorProtocol,
        TargetProtocol,
    )

__all__ = [
    "build_trainer",
    "train",
]


def train(
    train_annotations: Sequence[data.ClipAnnotation],
    val_annotations: Sequence[data.ClipAnnotation] | None = None,
    targets: Optional["TargetProtocol"] = None,
    preprocessor: Optional["PreprocessorProtocol"] = None,
    audio_loader: Optional["AudioLoader"] = None,
    labeller: Optional["ClipLabeller"] = None,
    config: Optional["BatDetect2Config"] = None,
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
    from batdetect2.config import BatDetect2Config

    if seed is not None:
        seed_everything(seed)

    config = config or BatDetect2Config()

    targets = targets or build_targets(config=config.model.targets)

    audio_loader = audio_loader or build_audio_loader(config=config.audio)

    preprocessor = preprocessor or build_preprocessor(
        input_samplerate=audio_loader.samplerate,
        config=config.model.preprocess,
    )

    labeller = labeller or build_clip_labeler(
        targets,
        min_freq=preprocessor.min_freq,
        max_freq=preprocessor.max_freq,
        config=config.train.labels,
    )

    train_dataloader = build_train_loader(
        train_annotations,
        audio_loader=audio_loader,
        labeller=labeller,
        preprocessor=preprocessor,
        config=config.train.train_loader,
        num_workers=train_workers,
    )

    val_dataloader = (
        build_val_loader(
            val_annotations,
            audio_loader=audio_loader,
            labeller=labeller,
            preprocessor=preprocessor,
            config=config.train.val_loader,
            num_workers=val_workers,
        )
        if val_annotations is not None
        else None
    )

    train_config_dict = config.train.model_dump(mode="json")
    if "optimizer" in train_config_dict:
        train_config_dict["optimizer"]["t_max"] *= len(train_dataloader)

    module = build_training_module(
        model_config=config.model.model_dump(mode="json"),
        train_config=train_config_dict,
    )

    trainer = trainer or build_trainer(
        config,
        evaluator=build_evaluator(
            config.train.validation,
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


def build_trainer(
    config: "BatDetect2Config",
    evaluator: "EvaluatorProtocol",
    checkpoint_dir: Path | None = None,
    log_dir: Path | None = None,
    experiment_name: str | None = None,
    run_name: str | None = None,
    num_epochs: int | None = None,
) -> Trainer:
    trainer_conf = config.train.trainer
    logger.opt(lazy=True).debug(
        "Building trainer with config: \n{config}",
        config=lambda: trainer_conf.to_yaml_string(exclude_none=True),
    )

    train_logger = build_logger(
        config.train.logger,
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
                config=config.train.checkpoints,
                checkpoint_dir=checkpoint_dir,
                experiment_name=experiment_name,
                run_name=run_name,
            ),
            ValidationMetrics(evaluator),
        ],
    )
