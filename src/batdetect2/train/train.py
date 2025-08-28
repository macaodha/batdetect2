from collections.abc import Sequence
from typing import List, Optional

import torch
from lightning import Trainer
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
from batdetect2.models import Model, build_model
from batdetect2.train.augmentations import (
    RandomExampleSource,
    build_augmentations,
)
from batdetect2.train.callbacks import ValidationMetrics
from batdetect2.train.clips import build_clipper
from batdetect2.train.config import FullTrainingConfig, TrainingConfig
from batdetect2.train.dataset import (
    LabeledDataset,
)
from batdetect2.train.lightning import TrainingModule
from batdetect2.train.logging import build_logger
from batdetect2.train.losses import build_loss
from batdetect2.typing import (
    PreprocessorProtocol,
    TargetProtocol,
    TrainExample,
)
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
    train_examples: Sequence[data.PathLike],
    val_examples: Optional[Sequence[data.PathLike]] = None,
    config: Optional[FullTrainingConfig] = None,
    model_path: Optional[data.PathLike] = None,
    train_workers: Optional[int] = None,
    val_workers: Optional[int] = None,
):
    config = config or FullTrainingConfig()

    model = build_model(config=config)

    trainer = build_trainer(config, targets=model.targets)

    train_dataloader = build_train_loader(
        train_examples,
        preprocessor=model.preprocessor,
        config=config.train,
        num_workers=train_workers,
    )

    val_dataloader = (
        build_val_loader(
            val_examples,
            preprocessor=model.preprocessor,
            config=config.train,
            num_workers=val_workers,
        )
        if val_examples is not None
        else None
    )

    if model_path is not None:
        logger.debug("Loading model from: {path}", path=model_path)
        module = TrainingModule.load_from_checkpoint(model_path)  # type: ignore
    else:
        module = build_training_module(
            model,
            config,
            batches_per_epoch=len(train_dataloader),
        )

    logger.info("Starting main training loop...")
    trainer.fit(
        module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    logger.info("Training complete.")


def build_training_module(
    model: Model,
    config: FullTrainingConfig,
    batches_per_epoch: int,
) -> TrainingModule:
    loss = build_loss(config=config.train.loss)
    return TrainingModule(
        model=model,
        loss=loss,
        learning_rate=config.train.learning_rate,
        t_max=config.train.t_max * batches_per_epoch,
    )


def build_trainer_callbacks(
    targets: TargetProtocol, config: EvaluationConfig
) -> List[Callback]:
    return [
        ModelCheckpoint(
            dirpath="outputs/checkpoints",
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
            match_config=config.match,
        ),
    ]


def build_trainer(
    conf: FullTrainingConfig,
    targets: TargetProtocol,
) -> Trainer:
    trainer_conf = conf.train.trainer
    logger.opt(lazy=True).debug(
        "Building trainer with config: \n{config}",
        config=lambda: trainer_conf.to_yaml_string(exclude_none=True),
    )
    train_logger = build_logger(conf.train.logger)

    train_logger.log_hyperparams(conf.model_dump(mode="json"))

    return Trainer(
        **trainer_conf.model_dump(exclude_none=True),
        logger=train_logger,
        callbacks=build_trainer_callbacks(targets, config=conf.evaluation),
    )


def build_train_loader(
    train_examples: Sequence[data.PathLike],
    preprocessor: PreprocessorProtocol,
    config: Optional[TrainingConfig] = None,
    num_workers: Optional[int] = None,
) -> DataLoader:
    config = config or TrainingConfig()

    logger.info("Building training data loader...")
    train_dataset = build_train_dataset(
        train_examples,
        preprocessor=preprocessor,
        config=config,
    )
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
    val_examples: Sequence[data.PathLike],
    preprocessor: PreprocessorProtocol,
    config: Optional[TrainingConfig] = None,
    num_workers: Optional[int] = None,
):
    config = config or TrainingConfig()

    logger.info("Building validation data loader...")
    val_dataset = build_val_dataset(
        val_examples,
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
        batch_size=loader_conf.batch_size,
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
    examples: Sequence[data.PathLike],
    preprocessor: PreprocessorProtocol,
    config: Optional[TrainingConfig] = None,
) -> LabeledDataset:
    logger.info("Building training dataset...")
    config = config or TrainingConfig()

    clipper = build_clipper(
        preprocessor=preprocessor,
        config=config.cliping,
        random=True,
    )

    random_example_source = RandomExampleSource(
        list(examples),
        clipper=clipper,
    )

    if config.augmentations.enabled and config.augmentations.steps:
        augmentations = build_augmentations(
            preprocessor,
            config=config.augmentations,
            example_source=random_example_source,
        )
    else:
        logger.debug("No augmentations configured for training dataset.")
        augmentations = None

    return LabeledDataset(
        examples,
        clipper=clipper,
        augmentation=augmentations,
    )


def build_val_dataset(
    examples: Sequence[data.PathLike],
    preprocessor: PreprocessorProtocol,
    config: Optional[TrainingConfig] = None,
    train: bool = True,
) -> LabeledDataset:
    logger.info("Building validation dataset...")
    config = config or TrainingConfig()
    clipper = build_clipper(
        preprocessor=preprocessor,
        config=config.cliping,
        random=train,
    )
    return LabeledDataset(examples, clipper=clipper)
