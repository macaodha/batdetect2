from typing import List, Optional

from lightning import Trainer
from soundevent import data
from torch.utils.data import DataLoader

from batdetect2.models.types import DetectionModel
from batdetect2.postprocess.types import PostprocessorProtocol
from batdetect2.preprocess.types import PreprocessorProtocol
from batdetect2.targets.types import TargetProtocol
from batdetect2.train.augmentations import (
    build_augmentations,
)
from batdetect2.train.clips import build_clipper
from batdetect2.train.config import TrainingConfig
from batdetect2.train.dataset import LabeledDataset, RandomExampleSource
from batdetect2.train.lightning import TrainingModule
from batdetect2.train.losses import build_loss

__all__ = [
    "train",
]


def train(
    detector: DetectionModel,
    targets: TargetProtocol,
    preprocessor: PreprocessorProtocol,
    postprocessor: PostprocessorProtocol,
    train_examples: List[data.PathLike],
    val_examples: Optional[List[data.PathLike]] = None,
    config: Optional[TrainingConfig] = None,
) -> None:
    config = config or TrainingConfig()

    train_dataset = build_dataset(
        train_examples,
        preprocessor,
        config=config,
        train=True,
    )

    loss = build_loss(config.loss)

    module = TrainingModule(
        detector=detector,
        loss=loss,
        targets=targets,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        learning_rate=config.optimizer.learning_rate,
        t_max=config.optimizer.t_max,
    )

    trainer = Trainer(**config.trainer.model_dump())

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )

    val_dataloader = None
    if val_examples:
        val_dataset = build_dataset(
            val_examples,
            preprocessor,
            config=config,
            train=False,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
        )

    trainer.fit(
        module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )


def build_dataset(
    examples: List[data.PathLike],
    preprocessor: PreprocessorProtocol,
    config: Optional[TrainingConfig] = None,
    train: bool = True,
):
    config = config or TrainingConfig()

    clipper = build_clipper(config.cliping, random=train)

    augmentations = None

    if train:
        random_example_source = RandomExampleSource(
            examples,
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
