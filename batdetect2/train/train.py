from typing import List, Optional

from lightning import Trainer
from lightning.pytorch.callbacks import Callback
from soundevent import data
from torch.utils.data import DataLoader

from batdetect2.models.types import DetectionModel
from batdetect2.postprocess import build_postprocessor
from batdetect2.postprocess.types import PostprocessorProtocol
from batdetect2.preprocess import build_preprocessor
from batdetect2.preprocess.types import PreprocessorProtocol
from batdetect2.targets import build_targets
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
    "build_val_dataset",
    "build_train_dataset",
]


def train(
    detector: DetectionModel,
    train_examples: List[data.PathLike],
    targets: Optional[TargetProtocol] = None,
    preprocessor: Optional[PreprocessorProtocol] = None,
    postprocessor: Optional[PostprocessorProtocol] = None,
    val_examples: Optional[List[data.PathLike]] = None,
    config: Optional[TrainingConfig] = None,
    callbacks: Optional[List[Callback]] = None,
    model_path: Optional[data.PathLike] = None,
    **trainer_kwargs,
) -> None:
    config = config or TrainingConfig()
    if model_path is None:
        if preprocessor is None:
            preprocessor = build_preprocessor()

        if targets is None:
            targets = build_targets()

        if postprocessor is None:
            postprocessor = build_postprocessor(
                targets,
                min_freq=preprocessor.min_freq,
                max_freq=preprocessor.max_freq,
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
    else:
        module = TrainingModule.load_from_checkpoint(model_path)  # type: ignore

    train_dataset = build_train_dataset(
        train_examples,
        preprocessor=module.preprocessor,
        config=config,
    )

    trainer = Trainer(
        **config.trainer.model_dump(exclude_none=True),
        callbacks=callbacks,
        **trainer_kwargs,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )

    val_dataloader = None
    if val_examples:
        val_dataset = build_val_dataset(
            val_examples,
            config=config,
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


def build_train_dataset(
    examples: List[data.PathLike],
    preprocessor: PreprocessorProtocol,
    config: Optional[TrainingConfig] = None,
) -> LabeledDataset:
    config = config or TrainingConfig()

    clipper = build_clipper(config.cliping, random=True)

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


def build_val_dataset(
    examples: List[data.PathLike],
    config: Optional[TrainingConfig] = None,
    train: bool = True,
) -> LabeledDataset:
    config = config or TrainingConfig()
    clipper = build_clipper(config.cliping, random=train)
    return LabeledDataset(examples, clipper=clipper)
