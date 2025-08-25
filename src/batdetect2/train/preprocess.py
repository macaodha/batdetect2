"""Preprocesses datasets for BatDetect2 model training.

This module provides functions to take a collection of annotated audio clips
(`soundevent.data.ClipAnnotation`) and process them into the final format
required for training a BatDetect2 model. This typically involves:

1.  Loading the relevant audio segment for each annotation using a configured
    `PreprocessorProtocol`.
2.  Generating the corresponding input spectrogram using the
    `PreprocessorProtocol`.
3.  Generating the target heatmaps (detection, classification, size) using a
    configured `ClipLabeller` (which encapsulates the `TargetProtocol` logic).
4.  Packaging the input spectrogram, target heatmaps, and potentially the
    processed audio waveform into an `xarray.Dataset`.
5.  Saving each processed `xarray.Dataset` to a separate file (typically NetCDF)
    in an output directory.

This offline preprocessing is often preferred for large datasets as it avoids
computationally intensive steps during the actual training loop. The module
includes utilities for parallel processing using `multiprocessing`.
"""

import os
from pathlib import Path
from typing import Callable, Optional, Sequence, TypedDict

import numpy as np
import torch
import torch.utils.data
from loguru import logger
from pydantic import Field
from soundevent import data
from tqdm import tqdm

from batdetect2.configs import BaseConfig, load_config
from batdetect2.data.datasets import Dataset
from batdetect2.preprocess import PreprocessingConfig, build_preprocessor
from batdetect2.preprocess.audio import build_audio_loader
from batdetect2.targets import TargetConfig, build_targets
from batdetect2.train.labels import LabelConfig, build_clip_labeler
from batdetect2.typing import ClipLabeller, PreprocessorProtocol
from batdetect2.typing.preprocess import AudioLoader
from batdetect2.typing.train import PreprocessedExample

__all__ = [
    "preprocess_annotations",
    "generate_train_example",
    "preprocess_dataset",
    "TrainPreprocessConfig",
    "load_train_preprocessing_config",
]

FilenameFn = Callable[[data.ClipAnnotation], str]
"""Type alias for a function that generates an output filename."""


class TrainPreprocessConfig(BaseConfig):
    preprocess: PreprocessingConfig = Field(
        default_factory=PreprocessingConfig
    )
    targets: TargetConfig = Field(default_factory=TargetConfig)
    labels: LabelConfig = Field(default_factory=LabelConfig)


def load_train_preprocessing_config(
    path: data.PathLike,
    field: Optional[str] = None,
) -> TrainPreprocessConfig:
    return load_config(path=path, schema=TrainPreprocessConfig, field=field)


def preprocess_dataset(
    dataset: Dataset,
    config: TrainPreprocessConfig,
    output: Path,
    max_workers: Optional[int] = None,
) -> None:
    targets = build_targets(config=config.targets)
    preprocessor = build_preprocessor(config=config.preprocess)
    labeller = build_clip_labeler(
        targets,
        min_freq=preprocessor.min_freq,
        max_freq=preprocessor.max_freq,
        config=config.labels,
    )
    audio_loader = build_audio_loader(config=config.preprocess.audio)

    if not output.exists():
        logger.debug("Creating directory {directory}", directory=output)
        output.mkdir(parents=True)

    preprocess_annotations(
        dataset,
        output_dir=output,
        audio_loader=audio_loader,
        preprocessor=preprocessor,
        labeller=labeller,
        max_workers=max_workers,
    )


class Example(TypedDict):
    audio: torch.Tensor
    spectrogram: torch.Tensor
    detection_heatmap: torch.Tensor
    class_heatmap: torch.Tensor
    size_heatmap: torch.Tensor


def generate_train_example(
    clip_annotation: data.ClipAnnotation,
    audio_loader: AudioLoader,
    preprocessor: PreprocessorProtocol,
    labeller: ClipLabeller,
) -> PreprocessedExample:
    """Generate a complete training example for one annotation."""
    wave = torch.tensor(audio_loader.load_clip(clip_annotation.clip))
    spectrogram = preprocessor(wave)
    heatmaps = labeller(clip_annotation, spectrogram)
    return PreprocessedExample(
        audio=wave,
        spectrogram=spectrogram,
        detection_heatmap=heatmaps.detection,
        class_heatmap=heatmaps.classes,
        size_heatmap=heatmaps.size,
    )


class PreprocessingDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        clips: Dataset,
        audio_loader: AudioLoader,
        preprocessor: PreprocessorProtocol,
        labeller: ClipLabeller,
        filename_fn: FilenameFn,
        output_dir: Path,
        force: bool = False,
    ):
        self.clips = clips
        self.audio_loader = audio_loader
        self.preprocessor = preprocessor
        self.labeller = labeller
        self.filename_fn = filename_fn
        self.output_dir = output_dir
        self.force = force

    def __getitem__(self, idx) -> int:
        clip_annotation = self.clips[idx]

        filename = self.filename_fn(clip_annotation)

        path = self.output_dir / filename

        if path.exists() and not self.force:
            return idx

        if not path.parent.exists():
            path.parent.mkdir()

        example = generate_train_example(
            clip_annotation,
            audio_loader=self.audio_loader,
            preprocessor=self.preprocessor,
            labeller=self.labeller,
        )

        save_example_to_file(example, clip_annotation, path)

        return idx

    def __len__(self) -> int:
        return len(self.clips)


def save_example_to_file(
    example: PreprocessedExample,
    clip_annotation: data.ClipAnnotation,
    path: data.PathLike,
) -> None:
    np.savez_compressed(
        path,
        audio=example.audio.numpy(),
        spectrogram=example.spectrogram.numpy(),
        detection_heatmap=example.detection_heatmap.numpy(),
        class_heatmap=example.class_heatmap.numpy(),
        size_heatmap=example.size_heatmap.numpy(),
        clip_annotation=clip_annotation,
    )


def _get_filename(clip_annotation: data.ClipAnnotation) -> str:
    """Generate a default output filename based on the annotation UUID."""
    return f"{clip_annotation.uuid}"


def preprocess_annotations(
    clip_annotations: Sequence[data.ClipAnnotation],
    output_dir: data.PathLike,
    preprocessor: PreprocessorProtocol,
    audio_loader: AudioLoader,
    labeller: ClipLabeller,
    filename_fn: FilenameFn = _get_filename,
    max_workers: Optional[int] = None,
) -> None:
    """Preprocess a sequence of ClipAnnotations and save results to disk."""
    output_dir = Path(output_dir)

    if not output_dir.is_dir():
        logger.info(
            "Creating output directory: {output_dir}", output_dir=output_dir
        )
        output_dir.mkdir(parents=True)

    logger.info(
        "Starting preprocessing of {num_annotations} annotations with {max_workers} workers.",
        num_annotations=len(clip_annotations),
        max_workers=max_workers or "all available",
    )

    if max_workers is None:
        max_workers = os.cpu_count() or 0

    dataset = PreprocessingDataset(
        clips=list(clip_annotations),
        audio_loader=audio_loader,
        preprocessor=preprocessor,
        labeller=labeller,
        output_dir=Path(output_dir),
        filename_fn=filename_fn,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=max_workers,
        prefetch_factor=16,
    )

    for _ in tqdm(loader, total=len(dataset)):
        pass
