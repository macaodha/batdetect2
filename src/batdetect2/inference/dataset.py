from typing import List, NamedTuple, Sequence

import torch
from loguru import logger
from soundevent import data
from torch.utils.data import DataLoader, Dataset

from batdetect2.audio import build_audio_loader
from batdetect2.core import BaseConfig
from batdetect2.core.arrays import adjust_width
from batdetect2.preprocess import build_preprocessor
from batdetect2.typing import AudioLoader, PreprocessorProtocol

__all__ = [
    "InferenceDataset",
    "build_inference_dataset",
    "build_inference_loader",
]


DEFAULT_INFERENCE_CLIP_DURATION = 0.512


class DatasetItem(NamedTuple):
    spec: torch.Tensor
    idx: torch.Tensor
    start_time: torch.Tensor
    end_time: torch.Tensor


class InferenceDataset(Dataset[DatasetItem]):
    clips: List[data.Clip]

    def __init__(
        self,
        clips: Sequence[data.Clip],
        audio_loader: AudioLoader,
        preprocessor: PreprocessorProtocol,
        audio_dir: data.PathLike | None = None,
    ):
        self.clips = list(clips)
        self.preprocessor = preprocessor
        self.audio_loader = audio_loader
        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx: int) -> DatasetItem:
        clip = self.clips[idx]
        wav = self.audio_loader.load_clip(clip, audio_dir=self.audio_dir)
        wav_tensor = torch.tensor(wav).unsqueeze(0)
        spectrogram = self.preprocessor(wav_tensor)
        return DatasetItem(
            spec=spectrogram,
            idx=torch.tensor(idx),
            start_time=torch.tensor(clip.start_time),
            end_time=torch.tensor(clip.end_time),
        )


class InferenceLoaderConfig(BaseConfig):
    num_workers: int = 0
    batch_size: int = 8


def build_inference_loader(
    clips: Sequence[data.Clip],
    audio_loader: AudioLoader | None = None,
    preprocessor: PreprocessorProtocol | None = None,
    config: InferenceLoaderConfig | None = None,
    num_workers: int | None = None,
    batch_size: int | None = None,
) -> DataLoader[DatasetItem]:
    logger.info("Building inference data loader...")
    config = config or InferenceLoaderConfig()

    inference_dataset = build_inference_dataset(
        clips,
        audio_loader=audio_loader,
        preprocessor=preprocessor,
    )

    batch_size = batch_size or config.batch_size

    num_workers = num_workers or config.num_workers
    return DataLoader(
        inference_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=_collate_fn,
    )


def build_inference_dataset(
    clips: Sequence[data.Clip],
    audio_loader: AudioLoader | None = None,
    preprocessor: PreprocessorProtocol | None = None,
) -> InferenceDataset:
    if audio_loader is None:
        audio_loader = build_audio_loader()

    if preprocessor is None:
        preprocessor = build_preprocessor()

    return InferenceDataset(
        clips,
        audio_loader=audio_loader,
        preprocessor=preprocessor,
    )


def _collate_fn(batch: List[DatasetItem]) -> DatasetItem:
    max_width = max(item.spec.shape[-1] for item in batch)
    return DatasetItem(
        spec=torch.stack(
            [adjust_width(item.spec, max_width) for item in batch]
        ),
        idx=torch.stack([item.idx for item in batch]),
        start_time=torch.stack([item.start_time for item in batch]),
        end_time=torch.stack([item.end_time for item in batch]),
    )
