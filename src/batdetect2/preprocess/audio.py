from typing import Annotated, Literal, Union

import torch
from pydantic import Field

from batdetect2.audio import TARGET_SAMPLERATE_HZ
from batdetect2.core import BaseConfig, Registry
from batdetect2.preprocess.common import center_tensor, peak_normalize

__all__ = [
    "CenterAudioConfig",
    "ScaleAudioConfig",
    "FixDurationConfig",
    "build_audio_transform",
]


audio_transforms: Registry[torch.nn.Module, [int]] = Registry(
    "audio_transform"
)


class CenterAudioConfig(BaseConfig):
    name: Literal["center_audio"] = "center_audio"


class CenterAudio(torch.nn.Module):
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        return center_tensor(wav)

    @classmethod
    def from_config(cls, config: CenterAudioConfig, samplerate: int):
        return cls()


audio_transforms.register(CenterAudioConfig, CenterAudio)


class ScaleAudioConfig(BaseConfig):
    name: Literal["scale_audio"] = "scale_audio"


class ScaleAudio(torch.nn.Module):
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        return peak_normalize(wav)

    @classmethod
    def from_config(cls, config: ScaleAudioConfig, samplerate: int):
        return cls()


audio_transforms.register(ScaleAudioConfig, ScaleAudio)


class FixDurationConfig(BaseConfig):
    name: Literal["fix_duration"] = "fix_duration"
    duration: float = 0.5


class FixDuration(torch.nn.Module):
    def __init__(self, samplerate: int, duration: float):
        super().__init__()
        self.samplerate = samplerate
        self.duration = duration
        self.length = int(samplerate * duration)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        length = wav.shape[-1]

        if length == self.length:
            return wav

        if length > self.length:
            return wav[: self.length]

        return torch.nn.functional.pad(wav, (0, self.length - length))

    @classmethod
    def from_config(cls, config: FixDurationConfig, samplerate: int):
        return cls(samplerate=samplerate, duration=config.duration)


audio_transforms.register(FixDurationConfig, FixDuration)

AudioTransform = Annotated[
    Union[
        FixDurationConfig,
        ScaleAudioConfig,
        CenterAudioConfig,
    ],
    Field(discriminator="name"),
]


def build_audio_transform(
    config: AudioTransform,
    samplerate: int = TARGET_SAMPLERATE_HZ,
) -> torch.nn.Module:
    return audio_transforms.build(config, samplerate)
