from collections.abc import Sequence
from typing import List, Optional

from lightning import Trainer
from lightning.pytorch.callbacks import Callback
from soundevent import data
from torch.utils.data import DataLoader

from batdetect2.evaluate.metrics import (
    ClassificationAccuracy,
    ClassificationMeanAveragePrecision,
    DetectionAveragePrecision,
)
from batdetect2.preprocess import (
    PreprocessorProtocol,
)
from batdetect2.targets import TargetProtocol
from batdetect2.train.augmentations import build_augmentations
from batdetect2.train.callbacks import ValidationMetrics
from batdetect2.train.clips import build_clipper
from batdetect2.train.config import FullTrainingConfig, TrainingConfig
from batdetect2.train.dataset import (
    LabeledDataset,
    RandomExampleSource,
    collate_fn,
)
from batdetect2.train.lightning import TrainingModule
from batdetect2.train.logging import build_logger

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
    train_workers: int = 0,
    val_workers: int = 0,
):
    conf = config or FullTrainingConfig()

    if model_path is not None:
        module = TrainingModule.load_from_checkpoint(model_path)  # type: ignore
    else:
        module = TrainingModule(conf)

    trainer = build_trainer(conf, targets=module.targets)

    train_dataloader = build_train_loader(
        train_examples,
        preprocessor=module.preprocessor,
        config=conf.train,
        num_workers=train_workers,
    )

    val_dataloader = (
        build_val_loader(
            val_examples,
            config=conf.train,
            num_workers=val_workers,
        )
        if val_examples is not None
        else None
    )

    trainer.fit(
        module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )


def build_trainer_callbacks(targets: TargetProtocol) -> List[Callback]:
    return [
        ValidationMetrics(
            metrics=[
                DetectionAveragePrecision(),
                ClassificationMeanAveragePrecision(
                    class_names=targets.class_names
                ),
                ClassificationAccuracy(class_names=targets.class_names),
            ]
        )
    ]


def build_trainer(
    conf: FullTrainingConfig,
    targets: TargetProtocol,
) -> Trainer:
    logger = build_logger(conf.train.logger)

    if logger and hasattr(logger, "log_hyperparams"):
        logger.log_hyperparams(conf.model_dump(exclude_none=True))

    return Trainer(
        accelerator=conf.train.accelerator,
        logger=logger,
        callbacks=build_trainer_callbacks(targets),
    )


def build_train_loader(
    train_examples: Sequence[data.PathLike],
    preprocessor: PreprocessorProtocol,
    config: TrainingConfig,
    num_workers: Optional[int] = None,
) -> DataLoader:
    train_dataset = build_train_dataset(
        train_examples,
        preprocessor=preprocessor,
        config=config,
    )

    return DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers or 0,
        collate_fn=collate_fn,
    )


def build_val_loader(
    val_examples: Sequence[data.PathLike],
    config: TrainingConfig,
    num_workers: Optional[int] = None,
):
    val_dataset = build_val_dataset(
        val_examples,
        config=config,
    )
    return DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=num_workers or 0,
        collate_fn=collate_fn,
    )


def build_train_dataset(
    examples: Sequence[data.PathLike],
    preprocessor: PreprocessorProtocol,
    config: Optional[TrainingConfig] = None,
) -> LabeledDataset:
    config = config or TrainingConfig()

    clipper = build_clipper(config.cliping, random=True)

    random_example_source = RandomExampleSource(
        list(examples),
        clipper=clipper,
    )

    augmentations = build_augmentations(
        preprocessor,
        config=config.augmentations,
        example_source=random_example_source,
    )

    return LabeledDataset(
        examples,
        clipper=clipper,
        augmentation=augmentations,
    )


def build_val_dataset(
    examples: Sequence[data.PathLike],
    config: Optional[TrainingConfig] = None,
    train: bool = True,
) -> LabeledDataset:
    config = config or TrainingConfig()
    clipper = build_clipper(config.cliping, random=train)
    return LabeledDataset(examples, clipper=clipper)
