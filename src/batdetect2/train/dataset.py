from typing import List, Sequence

import torch
from loguru import logger
from pydantic import Field
from soundevent import data
from torch.utils.data import DataLoader, Dataset

from batdetect2.audio import ClipConfig, build_audio_loader, build_clipper
from batdetect2.audio.clips import PaddedClipConfig
from batdetect2.audio.types import AudioLoader, ClipperProtocol
from batdetect2.core import BaseConfig
from batdetect2.core.arrays import adjust_width
from batdetect2.preprocess import build_preprocessor
from batdetect2.preprocess.types import PreprocessorProtocol
from batdetect2.train.augmentations import (
    DEFAULT_AUGMENTATION_CONFIG,
    AugmentationsConfig,
    RandomAudioSource,
    build_augmentations,
)
from batdetect2.train.labels import build_clip_labeler
from batdetect2.train.types import Augmentation, ClipLabeller, TrainExample

__all__ = [
    "TrainingDataset",
    "ValidationDataset",
    "build_val_loader",
    "build_train_loader",
    "build_train_dataset",
    "build_val_dataset",
]


class TrainingDataset(Dataset):
    def __init__(
        self,
        clip_annotations: Sequence[data.ClipAnnotation],
        audio_loader: AudioLoader,
        preprocessor: PreprocessorProtocol,
        labeller: ClipLabeller,
        clipper: ClipperProtocol | None = None,
        audio_augmentation: Augmentation | None = None,
        spectrogram_augmentation: Augmentation | None = None,
        audio_dir: data.PathLike | None = None,
    ):
        self.clip_annotations = clip_annotations
        self.clipper = clipper
        self.labeller = labeller
        self.preprocessor = preprocessor
        self.audio_loader = audio_loader
        self.audio_augmentation = audio_augmentation
        self.spectrogram_augmentation = spectrogram_augmentation
        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.clip_annotations)

    def __getitem__(self, index) -> TrainExample:
        clip_annotation = self.clip_annotations[index]

        if self.clipper is not None:
            clip_annotation = self.clipper(clip_annotation)

        clip = clip_annotation.clip

        wav = self.audio_loader.load_clip(clip, audio_dir=self.audio_dir)

        # Add channel dim
        wav_tensor = torch.tensor(wav).unsqueeze(0)

        if self.audio_augmentation is not None:
            wav_tensor, clip_annotation = self.audio_augmentation(
                wav_tensor,
                clip_annotation,
            )

        spectrogram = self.preprocessor(wav_tensor)

        if self.spectrogram_augmentation is not None:
            spectrogram, clip_annotation = self.spectrogram_augmentation(
                spectrogram,
                clip_annotation,
            )

        heatmaps = self.labeller(clip_annotation, spectrogram)

        return TrainExample(
            spec=spectrogram,
            detection_heatmap=heatmaps.detection,
            class_heatmap=heatmaps.classes,
            size_heatmap=heatmaps.size,
            idx=torch.tensor(index),
            start_time=torch.tensor(clip.start_time),
            end_time=torch.tensor(clip.end_time),
        )


class ValidationDataset(Dataset):
    def __init__(
        self,
        clip_annotations: Sequence[data.ClipAnnotation],
        audio_loader: AudioLoader,
        preprocessor: PreprocessorProtocol,
        labeller: ClipLabeller,
        clipper: ClipperProtocol | None = None,
        audio_dir: data.PathLike | None = None,
    ):
        self.clip_annotations = clip_annotations
        self.labeller = labeller
        self.preprocessor = preprocessor
        self.audio_loader = audio_loader
        self.audio_dir = audio_dir
        self.clipper = clipper

    def __len__(self):
        return len(self.clip_annotations)

    def __getitem__(self, index) -> TrainExample:
        clip_annotation = self.clip_annotations[index]

        if self.clipper is not None:
            clip_annotation = self.clipper(clip_annotation)

        clip = clip_annotation.clip
        wav = torch.tensor(
            self.audio_loader.load_clip(clip, audio_dir=self.audio_dir)
        ).unsqueeze(0)

        spectrogram = self.preprocessor(wav)

        heatmaps = self.labeller(clip_annotation, spectrogram)

        return TrainExample(
            spec=spectrogram,
            detection_heatmap=heatmaps.detection,
            class_heatmap=heatmaps.classes,
            size_heatmap=heatmaps.size,
            idx=torch.tensor(index),
            start_time=torch.tensor(clip.start_time),
            end_time=torch.tensor(clip.end_time),
        )


class TrainLoaderConfig(BaseConfig):
    num_workers: int = 0

    batch_size: int = 8

    shuffle: bool = False

    augmentations: AugmentationsConfig = Field(
        default_factory=lambda: DEFAULT_AUGMENTATION_CONFIG.model_copy()
    )

    clipping_strategy: ClipConfig = Field(
        default_factory=lambda: PaddedClipConfig()
    )


def build_train_loader(
    clip_annotations: Sequence[data.ClipAnnotation],
    audio_loader: AudioLoader | None = None,
    labeller: ClipLabeller | None = None,
    preprocessor: PreprocessorProtocol | None = None,
    config: TrainLoaderConfig | None = None,
    num_workers: int | None = None,
) -> DataLoader:
    config = config or TrainLoaderConfig()

    logger.info("Building training data loader...")
    logger.opt(lazy=True).debug(
        "Training data loader config: \n{config}",
        config=lambda: config.to_yaml_string(exclude_none=True),
    )

    train_dataset = build_train_dataset(
        clip_annotations,
        audio_loader=audio_loader,
        labeller=labeller,
        preprocessor=preprocessor,
        config=config,
    )

    num_workers = num_workers or config.num_workers
    return DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=num_workers,
        collate_fn=_collate_fn,
    )


class ValLoaderConfig(BaseConfig):
    num_workers: int = 0

    clipping_strategy: ClipConfig = Field(
        default_factory=lambda: PaddedClipConfig()
    )


def build_val_loader(
    clip_annotations: Sequence[data.ClipAnnotation],
    audio_loader: AudioLoader | None = None,
    labeller: ClipLabeller | None = None,
    preprocessor: PreprocessorProtocol | None = None,
    config: ValLoaderConfig | None = None,
    num_workers: int | None = None,
):
    logger.info("Building validation data loader...")
    config = config or ValLoaderConfig()
    logger.opt(lazy=True).debug(
        "Validation data loader config: \n{config}",
        config=lambda: config.to_yaml_string(exclude_none=True),
    )

    val_dataset = build_val_dataset(
        clip_annotations,
        audio_loader=audio_loader,
        labeller=labeller,
        preprocessor=preprocessor,
        config=config,
    )

    num_workers = num_workers or config.num_workers
    return DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_fn,
    )


def build_train_dataset(
    clip_annotations: Sequence[data.ClipAnnotation],
    audio_loader: AudioLoader | None = None,
    labeller: ClipLabeller | None = None,
    preprocessor: PreprocessorProtocol | None = None,
    config: TrainLoaderConfig | None = None,
) -> TrainingDataset:
    logger.info("Building training dataset...")
    config = config or TrainLoaderConfig()

    clipper = build_clipper(config=config.clipping_strategy)

    if audio_loader is None:
        audio_loader = build_audio_loader()

    if preprocessor is None:
        preprocessor = build_preprocessor()

    if labeller is None:
        labeller = build_clip_labeler(
            min_freq=preprocessor.min_freq,
            max_freq=preprocessor.max_freq,
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
    audio_loader: AudioLoader | None = None,
    labeller: ClipLabeller | None = None,
    preprocessor: PreprocessorProtocol | None = None,
    config: ValLoaderConfig | None = None,
) -> ValidationDataset:
    logger.info("Building validation dataset...")
    config = config or ValLoaderConfig()

    if audio_loader is None:
        audio_loader = build_audio_loader()

    if preprocessor is None:
        preprocessor = build_preprocessor()

    if labeller is None:
        labeller = build_clip_labeler(
            min_freq=preprocessor.min_freq,
            max_freq=preprocessor.max_freq,
        )

    clipper = build_clipper(config.clipping_strategy)
    return ValidationDataset(
        clip_annotations,
        audio_loader=audio_loader,
        labeller=labeller,
        preprocessor=preprocessor,
        clipper=clipper,
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
