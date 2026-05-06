from collections.abc import Sequence
from pathlib import Path
from typing import Optional

from lightning import Trainer, seed_everything
from lightning.pytorch.loggers import Logger
from loguru import logger
from soundevent import data

from batdetect2.audio import AudioConfig, AudioLoader, build_audio_loader
from batdetect2.evaluate import EvaluatorProtocol, build_evaluator
from batdetect2.logging import (
    LoggerConfig,
    LoggingCallback,
    TensorBoardLoggerConfig,
    build_logger,
)
from batdetect2.models import Model, ModelConfig, build_model
from batdetect2.preprocess import PreprocessorProtocol, build_preprocessor
from batdetect2.targets import (
    ROIMapperProtocol,
    TargetConfig,
    TargetProtocol,
    build_roi_mapping,
    build_targets,
)
from batdetect2.train.callbacks import ValidationMetrics
from batdetect2.train.checkpoints import build_checkpoint_callback
from batdetect2.train.config import TrainingConfig
from batdetect2.train.dataset import build_train_loader, build_val_loader
from batdetect2.train.labels import build_clip_labeler
from batdetect2.train.lightning import build_training_module
from batdetect2.train.logging import (
    ConfigHyperparameterLogging,
    DataSummaryArtifactLogging,
    TargetConfigArtifactLogging,
    TrainLoggingContext,
)
from batdetect2.train.types import ClipLabeller

__all__ = [
    "build_trainer",
    "run_train",
]


DEFAULT_LOG_DIR = Path("outputs") / "logs"


def run_train(
    train_annotations: Sequence[data.ClipAnnotation],
    val_annotations: Sequence[data.ClipAnnotation] | None = None,
    model: Model | None = None,
    targets: Optional["TargetProtocol"] = None,
    roi_mapper: Optional["ROIMapperProtocol"] = None,
    preprocessor: Optional["PreprocessorProtocol"] = None,
    audio_loader: Optional["AudioLoader"] = None,
    labeller: Optional["ClipLabeller"] = None,
    audio_config: Optional[AudioConfig] = None,
    targets_config: TargetConfig | None = None,
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
    logging_callbacks: Sequence[LoggingCallback[TrainLoggingContext]] = (),
):
    if seed is not None:
        seed_everything(seed)

    model_config = (
        ModelConfig()
        if model is None
        else ModelConfig.model_validate(model.get_config())
    )
    targets_config = targets_config or TargetConfig()
    audio_config = audio_config or AudioConfig()
    train_config = train_config or TrainingConfig()

    if model is not None:
        if targets is None:
            raise ValueError(
                "targets must be provided when training with an existing "
                "model."
            )

        if roi_mapper is None:
            raise ValueError(
                "roi_mapper must be provided when training with an existing "
                "model."
            )

    if targets is None:
        targets = build_targets(config=targets_config)
    else:
        targets_config = TargetConfig.model_validate(targets.get_config())

    roi_mapper = roi_mapper or build_roi_mapping(config=targets_config.roi)

    if model is not None:
        _validate_model_compatibility(
            model=model,
            model_config=model_config,
            class_names=targets.class_names,
            dimension_names=roi_mapper.dimension_names,
        )

    audio_loader = audio_loader or build_audio_loader(config=audio_config)

    preprocessor = preprocessor or build_preprocessor(
        input_samplerate=audio_loader.samplerate,
        config=model_config.preprocess,
    )

    labeller = labeller or build_clip_labeler(
        targets,
        roi_mapper,
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
        targets_config=targets_config,
        class_names=targets.class_names,
        dimension_names=roi_mapper.dimension_names,
        train_config=train_config,
        model=model,
    )

    evaluator = build_evaluator(
        train_config.validation,
        targets=targets,
        roi_mapper=roi_mapper,
    )

    train_logger = build_logger(
        logger_config or TensorBoardLoggerConfig(),
        log_dir=log_dir,
        experiment_name=experiment_name,
        run_name=run_name,
    )
    root_artifact_path = (
        Path(log_dir) if log_dir is not None else DEFAULT_LOG_DIR
    )
    root_artifact_path.mkdir(parents=True, exist_ok=True)

    logging_context = TrainLoggingContext(
        model_config=model_config.model_dump(mode="json"),
        train_config=train_config,
        audio_config=audio_config,
        targets=targets,
        train_dataset=train_annotations,
        val_dataset=val_annotations,
    )

    resolved_logging_callbacks = (
        ConfigHyperparameterLogging(),
        TargetConfigArtifactLogging(),
        DataSummaryArtifactLogging(),
        *logging_callbacks,
    )

    for callback in resolved_logging_callbacks:
        callback.run(train_logger, root_artifact_path, logging_context)

    trainer = trainer or build_trainer(
        train_config,
        train_logger=train_logger,
        evaluator=evaluator,
        targets=targets,
        roi_mapper=roi_mapper,
        checkpoint_dir=checkpoint_dir,
        num_epochs=num_epochs,
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
    class_names: list[str],
    dimension_names: list[str],
) -> None:
    reference_model = build_model(
        config=model_config,
        class_names=class_names,
        dimension_names=dimension_names,
    )

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
    train_logger: Logger,
    evaluator: "EvaluatorProtocol",
    targets: "TargetProtocol",
    roi_mapper: "ROIMapperProtocol",
    checkpoint_dir: Path | None = None,
    experiment_name: str | None = None,
    run_name: str | None = None,
    num_epochs: int | None = None,
) -> Trainer:
    trainer_conf = config.trainer
    logger.opt(lazy=True).debug(
        "Building trainer with config: \n{config}",
        config=lambda: trainer_conf.to_yaml_string(exclude_none=True),
    )

    if num_epochs is not None:
        trainer_conf.max_epochs = num_epochs

    train_config = trainer_conf.model_dump(exclude_none=True)

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
            ValidationMetrics(evaluator, targets, roi_mapper),
        ],
    )
