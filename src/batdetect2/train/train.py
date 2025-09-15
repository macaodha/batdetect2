from collections.abc import Sequence
from pathlib import Path
from typing import List, Optional

from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from loguru import logger
from soundevent import data

from batdetect2.evaluate.evaluator import build_evaluator
from batdetect2.plotting.clips import PreprocessorProtocol, build_audio_loader
from batdetect2.preprocess import build_preprocessor
from batdetect2.targets import build_targets
from batdetect2.train.callbacks import ValidationMetrics
from batdetect2.train.config import (
    FullTrainingConfig,
)
from batdetect2.train.dataset import build_train_loader, build_val_loader
from batdetect2.train.labels import build_clip_labeler
from batdetect2.train.lightning import TrainingModule, build_training_module
from batdetect2.train.logging import build_logger
from batdetect2.typing import (
    TargetProtocol,
)
from batdetect2.typing.preprocess import AudioLoader
from batdetect2.typing.train import ClipLabeller

__all__ = [
    "build_trainer",
    "train",
]

DEFAULT_CHECKPOINT_DIR: Path = Path("outputs") / "checkpoints"


def train(
    train_annotations: Sequence[data.ClipAnnotation],
    val_annotations: Optional[Sequence[data.ClipAnnotation]] = None,
    trainer: Optional[Trainer] = None,
    targets: Optional[TargetProtocol] = None,
    preprocessor: Optional[PreprocessorProtocol] = None,
    audio_loader: Optional[AudioLoader] = None,
    labeller: Optional[ClipLabeller] = None,
    config: Optional[FullTrainingConfig] = None,
    model_path: Optional[data.PathLike] = None,
    train_workers: Optional[int] = None,
    val_workers: Optional[int] = None,
    checkpoint_dir: Optional[Path] = None,
    log_dir: Optional[Path] = None,
    experiment_name: Optional[str] = None,
    run_name: Optional[str] = None,
    seed: Optional[int] = None,
):
    if seed is not None:
        seed_everything(seed)

    config = config or FullTrainingConfig()

    targets = targets or build_targets(config.targets)

    preprocessor = preprocessor or build_preprocessor(config.preprocess)

    audio_loader = audio_loader or build_audio_loader(
        config=config.preprocess.audio
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

    if model_path is not None:
        logger.debug("Loading model from: {path}", path=model_path)
        module = TrainingModule.load_from_checkpoint(model_path)  # type: ignore
    else:
        module = build_training_module(
            config,
            t_max=config.train.optimizer.t_max * len(train_dataloader),
        )

    trainer = trainer or build_trainer(
        config,
        targets=targets,
        checkpoint_dir=checkpoint_dir,
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


def build_trainer_callbacks(
    targets: TargetProtocol,
    config: FullTrainingConfig,
    checkpoint_dir: Optional[Path] = None,
    experiment_name: Optional[str] = None,
    run_name: Optional[str] = None,
) -> List[Callback]:
    if checkpoint_dir is None:
        checkpoint_dir = DEFAULT_CHECKPOINT_DIR

    if experiment_name is not None:
        checkpoint_dir = checkpoint_dir / experiment_name

    if run_name is not None:
        checkpoint_dir = checkpoint_dir / run_name

    evaluator = build_evaluator(config=config.evaluation, targets=targets)

    return [
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            save_top_k=1,
            filename="best-{epoch:02d}-{val_loss:.0f}",
            monitor="total_loss/val",
        ),
        ValidationMetrics(evaluator),
    ]


def build_trainer(
    conf: FullTrainingConfig,
    targets: TargetProtocol,
    checkpoint_dir: Optional[Path] = None,
    log_dir: Optional[Path] = None,
    experiment_name: Optional[str] = None,
    run_name: Optional[str] = None,
) -> Trainer:
    trainer_conf = conf.train.trainer
    logger.opt(lazy=True).debug(
        "Building trainer with config: \n{config}",
        config=lambda: trainer_conf.to_yaml_string(exclude_none=True),
    )
    train_logger = build_logger(
        conf.train.logger,
        log_dir=log_dir,
        experiment_name=experiment_name,
        run_name=run_name,
    )

    train_logger.log_hyperparams(
        conf.model_dump(
            mode="json",
            exclude_none=True,
        )
    )

    return Trainer(
        **trainer_conf.model_dump(exclude_none=True),
        logger=train_logger,
        callbacks=build_trainer_callbacks(
            targets,
            config=conf,
            checkpoint_dir=checkpoint_dir,
            experiment_name=experiment_name,
            run_name=run_name,
        ),
    )
