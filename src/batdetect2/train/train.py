from collections.abc import Sequence
from pathlib import Path
from typing import Optional

from lightning import Trainer, seed_everything
from loguru import logger
from soundevent import data

from batdetect2.audio import AudioConfig, build_audio_loader
from batdetect2.audio.types import AudioLoader
from batdetect2.evaluate import build_evaluator
from batdetect2.evaluate.types import EvaluatorProtocol
from batdetect2.logging import (
    LoggerConfig,
    TensorBoardLoggerConfig,
    build_logger,
)
from batdetect2.models import Model, ModelConfig, build_model
from batdetect2.preprocess import build_preprocessor
from batdetect2.preprocess.types import PreprocessorProtocol
from batdetect2.targets import build_targets
from batdetect2.targets.types import TargetProtocol
from batdetect2.train import TrainingConfig
from batdetect2.train.callbacks import ValidationMetrics
from batdetect2.train.checkpoints import build_checkpoint_callback
from batdetect2.train.dataset import build_train_loader, build_val_loader
from batdetect2.train.labels import build_clip_labeler
from batdetect2.train.lightning import build_training_module
from batdetect2.train.types import ClipLabeller

__all__ = [
    "build_trainer",
    "run_train",
]


def run_train(
    train_annotations: Sequence[data.ClipAnnotation],
    val_annotations: Sequence[data.ClipAnnotation] | None = None,
    model: Model | None = None,
    targets: Optional["TargetProtocol"] = None,
    preprocessor: Optional["PreprocessorProtocol"] = None,
    audio_loader: Optional["AudioLoader"] = None,
    labeller: Optional["ClipLabeller"] = None,
    audio_config: Optional[AudioConfig] = None,
    model_config: Optional[ModelConfig] = None,
    train_config: Optional[TrainingConfig] = None,
    logger_config: LoggerConfig | None = None,
    trainer: Trainer | None = None,
    train_workers: int = 0,
    val_workers: int = 0,
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

    if model is not None:
        _validate_model_compatibility(model=model, model_config=model_config)

    if model is not None:
        targets = targets or model.targets

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
        model=model,
    )

    trainer = trainer or build_trainer(
        train_config,
        logger_config=logger_config,
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


def _validate_model_compatibility(
    model: Model,
    model_config: ModelConfig,
) -> None:
    reference_model = build_model(config=model_config)

    expected_shapes = {
        key: tuple(value.shape)
        for key, value in reference_model.state_dict().items()
    }
    actual_shapes = {
        key: tuple(value.shape) for key, value in model.state_dict().items()
    }

    expected_keys = set(expected_shapes)
    actual_keys = set(actual_shapes)

    missing_keys = sorted(expected_keys - actual_keys)
    if missing_keys:
        key = missing_keys[0]
        raise ValueError(
            "Provided model is incompatible with model_config: "
            f"missing state key '{key}'."
        )

    extra_keys = sorted(actual_keys - expected_keys)
    if extra_keys:
        key = extra_keys[0]
        raise ValueError(
            "Provided model is incompatible with model_config: "
            f"unexpected state key '{key}'."
        )

    for key, expected_shape in expected_shapes.items():
        actual_shape = actual_shapes[key]
        if actual_shape != expected_shape:
            raise ValueError(
                "Provided model is incompatible with model_config: "
                f"shape mismatch for '{key}' (expected {expected_shape}, "
                f"got {actual_shape})."
            )


def build_trainer(
    config: TrainingConfig,
    logger_config: LoggerConfig | None,
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
        logger_config or TensorBoardLoggerConfig(),
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
