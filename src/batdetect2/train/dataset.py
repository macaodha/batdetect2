from typing import Optional, Sequence, Tuple

import torch
from soundevent import data
from torch.utils.data import Dataset

from batdetect2.typing import ClipperProtocol, TrainExample
from batdetect2.typing.preprocess import AudioLoader, PreprocessorProtocol
from batdetect2.typing.train import (
    Augmentation,
    ClipLabeller,
)

__all__ = [
    "TrainingDataset",
]


class TrainingDataset(Dataset):
    def __init__(
        self,
        clip_annotations: Sequence[data.ClipAnnotation],
        audio_loader: AudioLoader,
        preprocessor: PreprocessorProtocol,
        labeller: ClipLabeller,
        clipper: Optional[ClipperProtocol] = None,
        audio_augmentation: Optional[Augmentation] = None,
        spectrogram_augmentation: Optional[Augmentation] = None,
        audio_dir: Optional[data.PathLike] = None,
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

    def __getitem__(self, idx) -> TrainExample:
        clip_annotation = self.clip_annotations[idx]

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
            idx=torch.tensor(idx),
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
        clipper: Optional[ClipperProtocol] = None,
        audio_dir: Optional[data.PathLike] = None,
    ):
        self.clip_annotations = clip_annotations
        self.labeller = labeller
        self.preprocessor = preprocessor
        self.audio_loader = audio_loader
        self.audio_dir = audio_dir
        self.clipper = clipper

    def __len__(self):
        return len(self.clip_annotations)

    def __getitem__(self, idx) -> TrainExample:
        clip_annotation = self.clip_annotations[idx]

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
            idx=torch.tensor(idx),
            start_time=torch.tensor(clip.start_time),
            end_time=torch.tensor(clip.end_time),
        )
