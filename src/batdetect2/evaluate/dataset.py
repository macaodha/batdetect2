from typing import List, NamedTuple, Sequence

import torch
from loguru import logger
from pydantic import Field
from soundevent import data
from torch.utils.data import DataLoader, Dataset

from batdetect2.audio import ClipConfig, build_audio_loader, build_clipper
from batdetect2.audio.clips import PaddedClipConfig
from batdetect2.core import BaseConfig
from batdetect2.core.arrays import adjust_width
from batdetect2.preprocess import build_preprocessor
from batdetect2.typing import (
    AudioLoader,
    ClipperProtocol,
    PreprocessorProtocol,
)

__all__ = [
    "TestDataset",
    "build_test_dataset",
    "build_test_loader",
]


class TestExample(NamedTuple):
    spec: torch.Tensor
    idx: torch.Tensor
    start_time: torch.Tensor
    end_time: torch.Tensor


class TestDataset(Dataset[TestExample]):
    clip_annotations: List[data.ClipAnnotation]

    def __init__(
        self,
        clip_annotations: Sequence[data.ClipAnnotation],
        audio_loader: AudioLoader,
        preprocessor: PreprocessorProtocol,
        clipper: ClipperProtocol | None = None,
        audio_dir: data.PathLike | None = None,
    ):
        self.clip_annotations = list(clip_annotations)
        self.clipper = clipper
        self.preprocessor = preprocessor
        self.audio_loader = audio_loader
        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.clip_annotations)

    def __getitem__(self, idx: int) -> TestExample:
        clip_annotation = self.clip_annotations[idx]

        if self.clipper is not None:
            clip_annotation = self.clipper(clip_annotation)

        clip = clip_annotation.clip
        wav = self.audio_loader.load_clip(clip, audio_dir=self.audio_dir)
        wav_tensor = torch.tensor(wav).unsqueeze(0)
        spectrogram = self.preprocessor(wav_tensor)
        return TestExample(
            spec=spectrogram,
            idx=torch.tensor(idx),
            start_time=torch.tensor(clip.start_time),
            end_time=torch.tensor(clip.end_time),
        )


class TestLoaderConfig(BaseConfig):
    num_workers: int = 0
    clipping_strategy: ClipConfig = Field(
        default_factory=lambda: PaddedClipConfig()
    )


def build_test_loader(
    clip_annotations: Sequence[data.ClipAnnotation],
    audio_loader: AudioLoader | None = None,
    preprocessor: PreprocessorProtocol | None = None,
    config: TestLoaderConfig | None = None,
    num_workers: int | None = None,
) -> DataLoader[TestExample]:
    logger.info("Building test data loader...")
    config = config or TestLoaderConfig()
    logger.opt(lazy=True).debug(
        "Test data loader config: \n{config}",
        config=lambda: config.to_yaml_string(exclude_none=True),
    )

    test_dataset = build_test_dataset(
        clip_annotations,
        audio_loader=audio_loader,
        preprocessor=preprocessor,
        config=config,
    )

    num_workers = num_workers or config.num_workers
    return DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_fn,
    )


def build_test_dataset(
    clip_annotations: Sequence[data.ClipAnnotation],
    audio_loader: AudioLoader | None = None,
    preprocessor: PreprocessorProtocol | None = None,
    config: TestLoaderConfig | None = None,
) -> TestDataset:
    logger.info("Building training dataset...")
    config = config or TestLoaderConfig()

    clipper = build_clipper(config=config.clipping_strategy)

    if audio_loader is None:
        audio_loader = build_audio_loader()

    if preprocessor is None:
        preprocessor = build_preprocessor()

    return TestDataset(
        clip_annotations,
        audio_loader=audio_loader,
        clipper=clipper,
        preprocessor=preprocessor,
    )


def _collate_fn(batch: List[TestExample]) -> TestExample:
    max_width = max(item.spec.shape[-1] for item in batch)
    return TestExample(
        spec=torch.stack(
            [adjust_width(item.spec, max_width) for item in batch]
        ),
        idx=torch.stack([item.idx for item in batch]),
        start_time=torch.stack([item.start_time for item in batch]),
        end_time=torch.stack([item.end_time for item in batch]),
    )
