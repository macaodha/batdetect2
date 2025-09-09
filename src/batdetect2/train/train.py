from collections.abc import Sequence
from typing import List, Optional

import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from loguru import logger
from soundevent import data
from torch.utils.data import DataLoader

from batdetect2.evaluate.config import EvaluationConfig
from batdetect2.evaluate.metrics import (
    ClassificationAccuracy,
    ClassificationMeanAveragePrecision,
    DetectionAveragePrecision,
)
from batdetect2.plotting.clips import AudioLoader, build_audio_loader
from batdetect2.preprocess import build_preprocessor
from batdetect2.targets import build_targets
from batdetect2.train.augmentations import (
    RandomAudioSource,
    build_augmentations,
)
from batdetect2.train.callbacks import ValidationMetrics
from batdetect2.train.clips import build_clipper
from batdetect2.train.config import FullTrainingConfig, TrainingConfig
from batdetect2.train.dataset import TrainingDataset, ValidationDataset
from batdetect2.train.labels import build_clip_labeler
from batdetect2.train.lightning import TrainingModule
from batdetect2.train.logging import build_logger
from batdetect2.typing import (
    PreprocessorProtocol,
    TargetProtocol,
    TrainExample,
)
from batdetect2.typing.train import ClipLabeller
from batdetect2.utils.arrays import adjust_width

__all__ = [
    "build_train_dataset",
    "build_train_loader",
    "build_trainer",
    "build_val_dataset",
    "build_val_loader",
    "train",
]


def train(
    train_annotations: Sequence[data.ClipAnnotation],
    val_annotations: Optional[Sequence[data.ClipAnnotation]] = None,
    config: Optional[FullTrainingConfig] = None,
    model_path: Optional[data.PathLike] = None,
    train_workers: Optional[int] = None,
    val_workers: Optional[int] = None,
    checkpoint_dir: Optional[data.PathLike] = None,
    log_dir: Optional[data.PathLike] = None,
    experiment_name: Optional[str] = None,
    seed: Optional[int] = None,
):
    if seed is not None:
        seed_everything(seed)

    config = config or FullTrainingConfig()

    targets = build_targets(config.targets)

    preprocessor = build_preprocessor(config.preprocess)

    audio_loader = build_audio_loader(config=config.preprocess.audio)

    labeller = build_clip_labeler(
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
        config=config.train,
        num_workers=train_workers,
    )

    val_dataloader = (
        build_val_loader(
            val_annotations,
            audio_loader=audio_loader,
            labeller=labeller,
            preprocessor=preprocessor,
            config=config.train,
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
            t_max=config.train.t_max * len(train_dataloader),
        )

    trainer = build_trainer(
        config,
        targets=targets,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        experiment_name=experiment_name,
    )

    logger.info("Starting main training loop...")
    trainer.fit(
        module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    logger.info("Training complete.")


def build_training_module(
    config: Optional[FullTrainingConfig] = None,
    t_max: int = 200,
) -> TrainingModule:
    config = config or FullTrainingConfig()
    return TrainingModule(
        config=config,
        learning_rate=config.train.learning_rate,
        t_max=t_max,
    )


def build_trainer_callbacks(
    targets: TargetProtocol,
    preprocessor: PreprocessorProtocol,
    config: EvaluationConfig,
    checkpoint_dir: Optional[data.PathLike] = None,
    experiment_name: Optional[str] = None,
) -> List[Callback]:
    if checkpoint_dir is None:
        checkpoint_dir = "outputs/checkpoints"

    if experiment_name is not None:
        checkpoint_dir = f"{checkpoint_dir}/{experiment_name}"

    return [
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            save_top_k=1,
            monitor="total_loss/val",
        ),
        ValidationMetrics(
            metrics=[
                DetectionAveragePrecision(),
                ClassificationMeanAveragePrecision(
                    class_names=targets.class_names
                ),
                ClassificationAccuracy(class_names=targets.class_names),
            ],
            preprocessor=preprocessor,
            match_config=config.match,
        ),
    ]


def build_trainer(
    conf: FullTrainingConfig,
    targets: TargetProtocol,
    checkpoint_dir: Optional[data.PathLike] = None,
    log_dir: Optional[data.PathLike] = None,
    experiment_name: Optional[str] = None,
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
            config=conf.evaluation,
            preprocessor=build_preprocessor(conf.preprocess),
            checkpoint_dir=checkpoint_dir,
            experiment_name=train_logger.name,
        ),
    )


def build_train_loader(
    clip_annotations: Sequence[data.ClipAnnotation],
    audio_loader: AudioLoader,
    labeller: ClipLabeller,
    preprocessor: PreprocessorProtocol,
    config: Optional[TrainingConfig] = None,
    num_workers: Optional[int] = None,
) -> DataLoader:
    config = config or TrainingConfig()
    train_dataset = build_train_dataset(
        clip_annotations,
        audio_loader=audio_loader,
        labeller=labeller,
        preprocessor=preprocessor,
        config=config,
    )

    logger.info("Building training data loader...")
    loader_conf = config.dataloaders.train
    logger.opt(lazy=True).debug(
        "Training data loader config: \n{config}",
        config=lambda: loader_conf.to_yaml_string(exclude_none=True),
    )
    num_workers = num_workers or loader_conf.num_workers
    return DataLoader(
        train_dataset,
        batch_size=loader_conf.batch_size,
        shuffle=loader_conf.shuffle,
        num_workers=num_workers,
        collate_fn=_collate_fn,
    )


def build_val_loader(
    clip_annotations: Sequence[data.ClipAnnotation],
    audio_loader: AudioLoader,
    labeller: ClipLabeller,
    preprocessor: PreprocessorProtocol,
    config: Optional[TrainingConfig] = None,
    num_workers: Optional[int] = None,
):
    logger.info("Building validation data loader...")
    config = config or TrainingConfig()

    val_dataset = build_val_dataset(
        clip_annotations,
        audio_loader=audio_loader,
        labeller=labeller,
        preprocessor=preprocessor,
        config=config,
    )
    loader_conf = config.dataloaders.val
    logger.opt(lazy=True).debug(
        "Validation data loader config: \n{config}",
        config=lambda: loader_conf.to_yaml_string(exclude_none=True),
    )
    num_workers = num_workers or loader_conf.num_workers
    return DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=loader_conf.shuffle,
        num_workers=num_workers,
        collate_fn=_collate_fn,
    )


def _collate_fn(batch: List[TrainExample]) -> TrainExample:
    max_width = max(item.spec.shape[-1] for item in batch)
    return TrainExample(
        spec=torch.stack(
            [adjust_width(item.spec, max_width) for item in batch]
        ),
        detection_heatmap=torch.stack(
            [adjust_width(item.detection_heatmap, max_width) for item in batch]
        ),
        size_heatmap=torch.stack(
            [adjust_width(item.size_heatmap, max_width) for item in batch]
        ),
        class_heatmap=torch.stack(
            [adjust_width(item.class_heatmap, max_width) for item in batch]
        ),
        idx=torch.stack([item.idx for item in batch]),
        start_time=torch.stack([item.start_time for item in batch]),
        end_time=torch.stack([item.end_time for item in batch]),
    )


def build_train_dataset(
    clip_annotations: Sequence[data.ClipAnnotation],
    audio_loader: AudioLoader,
    labeller: ClipLabeller,
    preprocessor: PreprocessorProtocol,
    config: Optional[TrainingConfig] = None,
) -> TrainingDataset:
    logger.info("Building training dataset...")
    config = config or TrainingConfig()

    clipper = build_clipper(
        config=config.cliping,
        random=True,
    )

    random_example_source = RandomAudioSource(
        clip_annotations,
        audio_loader=audio_loader,
    )

    if config.augmentations.enabled:
        audio_augmentation, spectrogram_augmentation = build_augmentations(
            samplerate=preprocessor.input_samplerate,
            config=config.augmentations,
            audio_source=random_example_source,
        )
    else:
        logger.debug("No augmentations configured for training dataset.")
        audio_augmentation = None
        spectrogram_augmentation = None

    return TrainingDataset(
        clip_annotations,
        audio_loader=audio_loader,
        labeller=labeller,
        clipper=clipper,
        preprocessor=preprocessor,
        audio_augmentation=audio_augmentation,
        spectrogram_augmentation=spectrogram_augmentation,
    )


def build_val_dataset(
    clip_annotations: Sequence[data.ClipAnnotation],
    audio_loader: AudioLoader,
    labeller: ClipLabeller,
    preprocessor: PreprocessorProtocol,
    config: Optional[TrainingConfig] = None,
) -> ValidationDataset:
    logger.info("Building validation dataset...")
    config = config or TrainingConfig()

    return ValidationDataset(
        clip_annotations,
        audio_loader=audio_loader,
        labeller=labeller,
        preprocessor=preprocessor,
    )
