"""Applies data augmentation techniques to BatDetect2 training examples."""

import warnings
from collections.abc import Sequence
from typing import Annotated, Callable, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from loguru import logger
from pydantic import Field
from soundevent import data
from soundevent.geometry import scale_geometry, shift_geometry

from batdetect2.configs import BaseConfig, load_config
from batdetect2.train.clips import get_subclip_annotation
from batdetect2.typing import Augmentation
from batdetect2.typing.preprocess import AudioLoader
from batdetect2.utils.arrays import adjust_width

__all__ = [
    "AugmentationConfig",
    "AugmentationsConfig",
    "DEFAULT_AUGMENTATION_CONFIG",
    "EchoAugmentationConfig",
    "AudioSource",
    "FrequencyMaskAugmentationConfig",
    "MixAugmentationConfig",
    "TimeMaskAugmentationConfig",
    "VolumeAugmentationConfig",
    "WarpAugmentationConfig",
    "add_echo",
    "build_augmentations",
    "load_augmentation_config",
    "mask_frequency",
    "mask_time",
    "mix_audio",
    "scale_volume",
    "warp_spectrogram",
]

AudioSource = Callable[[float], tuple[torch.Tensor, data.ClipAnnotation]]


class MixAugmentationConfig(BaseConfig):
    """Configuration for MixUp augmentation (mixing two examples)."""

    augmentation_type: Literal["mix_audio"] = "mix_audio"

    probability: float = 0.2
    """Probability of applying this augmentation to an example."""

    min_weight: float = 0.3
    """Minimum mixing weight (lambda) applied to the primary example."""

    max_weight: float = 0.7
    """Maximum mixing weight (lambda) applied to the primary example."""


class MixAudio(torch.nn.Module):
    """Callable class for MixUp augmentation, handling example fetching."""

    def __init__(
        self,
        example_source: AudioSource,
        min_weight: float = 0.3,
        max_weight: float = 0.7,
    ):
        """Initialize the AudioMixer."""
        super().__init__()
        self.min_weight = min_weight
        self.example_source = example_source
        self.max_weight = max_weight

    def __call__(
        self,
        wav: torch.Tensor,
        clip_annotation: data.ClipAnnotation,
    ) -> Tuple[torch.Tensor, data.ClipAnnotation]:
        """Fetch another example and perform mixup."""
        other_wav, other_clip_annotation = self.example_source(
            clip_annotation.clip.duration
        )
        weight = np.random.uniform(self.min_weight, self.max_weight)
        mixed_audio = mix_audio(wav, other_wav, weight=weight)
        mixed_annotations = combine_clip_annotations(
            clip_annotation,
            other_clip_annotation,
        )
        return mixed_audio, mixed_annotations


def mix_audio(
    wav1: torch.Tensor,
    wav2: torch.Tensor,
    weight: float,
) -> torch.Tensor:
    """Combine two training examples."""
    wav2 = adjust_width(wav2, wav1.shape[-1])
    return weight * wav1 + (1 - weight) * wav2


def shift_sound_event_annotation(
    sound_event_annotation: data.SoundEventAnnotation,
    time: float,
) -> data.SoundEventAnnotation:
    sound_event = sound_event_annotation.sound_event
    geometry = sound_event.geometry

    if geometry is None:
        return sound_event_annotation

    sound_event = sound_event.model_copy(
        update=dict(geometry=shift_geometry(geometry, time=time))
    )
    return sound_event_annotation.model_copy(
        update=dict(sound_event=sound_event)
    )


def combine_clip_annotations(
    clip_annotation1: data.ClipAnnotation,
    clip_annotation2: data.ClipAnnotation,
) -> data.ClipAnnotation:
    time_shift = (
        clip_annotation1.clip.start_time - clip_annotation2.clip.start_time
    )
    return clip_annotation1.model_copy(
        update=dict(
            sound_events=[
                *clip_annotation1.sound_events,
                *[
                    shift_sound_event_annotation(sound_event, time=time_shift)
                    for sound_event in clip_annotation2.sound_events
                ],
            ]
        )
    )


class EchoAugmentationConfig(BaseConfig):
    """Configuration for adding synthetic echo/reverb."""

    augmentation_type: Literal["add_echo"] = "add_echo"
    probability: float = 0.2
    max_delay: float = 0.005
    min_weight: float = 0.0
    max_weight: float = 1.0


class AddEcho(torch.nn.Module):
    def __init__(
        self,
        min_weight: float = 0.1,
        max_weight: float = 1.0,
        max_delay: int = 2560,
    ):
        super().__init__()
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.max_delay = max_delay

    def forward(
        self,
        wav: torch.Tensor,
        clip_annotation: data.ClipAnnotation,
    ) -> Tuple[torch.Tensor, data.ClipAnnotation]:
        delay = np.random.randint(0, self.max_delay)
        weight = np.random.uniform(self.min_weight, self.max_weight)
        return add_echo(wav, delay=delay, weight=weight), clip_annotation


def add_echo(
    wav: torch.Tensor,
    delay: int,
    weight: float,
) -> torch.Tensor:
    """Add a synthetic echo to the audio waveform."""

    slices = [slice(None)] * wav.ndim
    slices[-1] = slice(None, -delay)
    audio_delay = adjust_width(wav[tuple(slices)], wav.shape[-1]).roll(
        delay, dims=-1
    )
    return mix_audio(wav, audio_delay, weight)


class VolumeAugmentationConfig(BaseConfig):
    """Configuration for random volume scaling of the spectrogram."""

    augmentation_type: Literal["scale_volume"] = "scale_volume"
    probability: float = 0.2
    min_scaling: float = 0.0
    max_scaling: float = 2.0


class ScaleVolume(torch.nn.Module):
    def __init__(self, min_scaling: float = 0.0, max_scaling: float = 2.0):
        super().__init__()
        self.min_scaling = min_scaling
        self.max_scaling = max_scaling

    def forward(
        self,
        spec: torch.Tensor,
        clip_annotation: data.ClipAnnotation,
    ) -> Tuple[torch.Tensor, data.ClipAnnotation]:
        factor = np.random.uniform(self.min_scaling, self.max_scaling)
        return scale_volume(spec, factor=factor), clip_annotation


def scale_volume(spec: torch.Tensor, factor: float) -> torch.Tensor:
    """Scale the amplitude of the spectrogram by a factor."""
    return spec * factor


class WarpAugmentationConfig(BaseConfig):
    augmentation_type: Literal["warp"] = "warp"
    probability: float = 0.2
    delta: float = 0.04


class WarpSpectrogram(torch.nn.Module):
    def __init__(self, delta: float = 0.04) -> None:
        super().__init__()
        self.delta = delta

    def forward(
        self,
        spec: torch.Tensor,
        clip_annotation: data.ClipAnnotation,
    ) -> Tuple[torch.Tensor, data.ClipAnnotation]:
        factor = np.random.uniform(1 - self.delta, 1 + self.delta)
        return (
            warp_spectrogram(spec, factor=factor),
            warp_clip_annotation(clip_annotation, factor=factor),
        )


def warp_sound_event_annotation(
    sound_event_annotation: data.SoundEventAnnotation,
    factor: float,
    anchor: float,
) -> data.SoundEventAnnotation:
    sound_event = sound_event_annotation.sound_event
    geometry = sound_event.geometry

    if geometry is None:
        return sound_event_annotation

    sound_event = sound_event.model_copy(
        update=dict(
            geometry=scale_geometry(
                geometry,
                time=1 / factor,
                time_anchor=anchor,
            )
        ),
    )
    return sound_event_annotation.model_copy(
        update=dict(sound_event=sound_event)
    )


def warp_clip_annotation(
    clip_annotation: data.ClipAnnotation,
    factor: float,
) -> data.ClipAnnotation:
    return clip_annotation.model_copy(
        update=dict(
            sound_events=[
                warp_sound_event_annotation(
                    sound_event,
                    factor=factor,
                    anchor=clip_annotation.clip.start_time,
                )
                for sound_event in clip_annotation.sound_events
            ]
        )
    )


def warp_spectrogram(
    spec: torch.Tensor,
    factor: float,
) -> torch.Tensor:
    """Apply time warping by resampling the time axis."""
    width = spec.shape[-1]
    height = spec.shape[-2]
    target_shape = [height, width]
    new_width = int(target_shape[-1] * factor)
    return torch.nn.functional.interpolate(
        adjust_width(spec, new_width).unsqueeze(0),
        size=target_shape,
        mode="bilinear",
    ).squeeze(0)


class TimeMaskAugmentationConfig(BaseConfig):
    augmentation_type: Literal["mask_time"] = "mask_time"
    probability: float = 0.2
    max_perc: float = 0.05
    max_masks: int = 3


class MaskTime(torch.nn.Module):
    def __init__(
        self,
        max_perc: float = 0.05,
        max_masks: int = 3,
        mask_heatmaps: bool = False,
    ) -> None:
        super().__init__()
        self.max_perc = max_perc
        self.max_masks = max_masks
        self.mask_heatmaps = mask_heatmaps

    def forward(
        self,
        spec: torch.Tensor,
        clip_annotation: data.ClipAnnotation,
    ) -> Tuple[torch.Tensor, data.ClipAnnotation]:
        num_masks = np.random.randint(1, self.max_masks + 1)
        width = spec.shape[-1]

        mask_size = np.random.randint(
            low=0,
            high=int(self.max_perc * width),
            size=num_masks,
        )
        mask_start = np.random.randint(
            low=0,
            high=width - mask_size,
            size=num_masks,
        )
        masks = [
            (start, start + size) for start, size in zip(mask_start, mask_size)
        ]
        return mask_time(spec, masks), clip_annotation


def mask_time(
    spec: torch.Tensor,
    masks: List[Tuple[int, int]],
    value: float = 0,
) -> torch.Tensor:
    """Apply time masking to the spectrogram."""
    for start, end in masks:
        slices = [slice(None)] * spec.ndim
        slices[-1] = slice(start, end)
        spec[tuple(slices)] = value

    return spec


class FrequencyMaskAugmentationConfig(BaseConfig):
    augmentation_type: Literal["mask_freq"] = "mask_freq"
    probability: float = 0.2
    max_perc: float = 0.10
    max_masks: int = 3
    mask_heatmaps: bool = False


class MaskFrequency(torch.nn.Module):
    def __init__(
        self,
        max_perc: float = 0.10,
        max_masks: int = 3,
        mask_heatmaps: bool = False,
    ) -> None:
        super().__init__()
        self.max_perc = max_perc
        self.max_masks = max_masks
        self.mask_heatmaps = mask_heatmaps

    def forward(
        self,
        spec: torch.Tensor,
        clip_annotation: data.ClipAnnotation,
    ) -> Tuple[torch.Tensor, data.ClipAnnotation]:
        num_masks = np.random.randint(1, self.max_masks + 1)
        height = spec.shape[-2]

        mask_size = np.random.randint(
            low=0,
            high=int(self.max_perc * height),
            size=num_masks,
        )
        mask_start = np.random.randint(
            low=0,
            high=height - mask_size,
            size=num_masks,
        )
        masks = [
            (start, start + size) for start, size in zip(mask_start, mask_size)
        ]
        return mask_frequency(spec, masks), clip_annotation


def mask_frequency(
    spec: torch.Tensor,
    masks: List[Tuple[int, int]],
) -> torch.Tensor:
    """Apply frequency masking to the spectrogram."""
    for start, end in masks:
        slices = [slice(None)] * spec.ndim
        slices[-2] = slice(start, end)
        spec[tuple(slices)] = 0

    return spec


AudioAugmentationConfig = Annotated[
    Union[
        MixAugmentationConfig,
        EchoAugmentationConfig,
    ],
    Field(discriminator="augmentation_type"),
]


SpectrogramAugmentationConfig = Annotated[
    Union[
        VolumeAugmentationConfig,
        WarpAugmentationConfig,
        FrequencyMaskAugmentationConfig,
        TimeMaskAugmentationConfig,
    ],
    Field(discriminator="augmentation_type"),
]

AugmentationConfig = Annotated[
    Union[
        MixAugmentationConfig,
        EchoAugmentationConfig,
        VolumeAugmentationConfig,
        WarpAugmentationConfig,
        FrequencyMaskAugmentationConfig,
        TimeMaskAugmentationConfig,
    ],
    Field(discriminator="augmentation_type"),
]
"""Type alias for the discriminated union of individual augmentation config."""


class AugmentationsConfig(BaseConfig):
    """Configuration for a sequence of data augmentations."""

    enabled: bool = True

    audio: List[AudioAugmentationConfig] = Field(default_factory=list)

    spectrogram: List[SpectrogramAugmentationConfig] = Field(
        default_factory=list
    )


class MaybeApply(torch.nn.Module):
    """Applies an augmentation function probabilistically."""

    def __init__(
        self,
        augmentation: Augmentation,
        probability: float = 0.2,
    ):
        """Initialize the wrapper."""
        super().__init__()
        self.augmentation = augmentation
        self.probability = probability

    def __call__(
        self,
        tensor: torch.Tensor,
        clip_annotation: data.ClipAnnotation,
    ) -> Tuple[torch.Tensor, data.ClipAnnotation]:
        """Apply the wrapped augmentation with configured probability."""
        if np.random.random() > self.probability:
            return tensor, clip_annotation

        return self.augmentation(tensor, clip_annotation)


def build_augmentation_from_config(
    config: AugmentationConfig,
    samplerate: int,
    audio_source: Optional[AudioSource] = None,
) -> Optional[Augmentation]:
    """Factory function to build a single augmentation from its config."""
    if config.augmentation_type == "mix_audio":
        if audio_source is None:
            warnings.warn(
                "Mix audio augmentation ('mix_audio') requires an "
                "'example_source' callable to be provided.",
                stacklevel=2,
            )
            return None

        return MixAudio(
            example_source=audio_source,
            min_weight=config.min_weight,
            max_weight=config.max_weight,
        )

    if config.augmentation_type == "add_echo":
        return AddEcho(
            max_delay=int(config.max_delay * samplerate),
            min_weight=config.min_weight,
            max_weight=config.max_weight,
        )

    if config.augmentation_type == "scale_volume":
        return ScaleVolume(
            max_scaling=config.max_scaling,
            min_scaling=config.min_scaling,
        )

    if config.augmentation_type == "warp":
        return WarpSpectrogram(
            delta=config.delta,
        )

    if config.augmentation_type == "mask_time":
        return MaskTime(
            max_perc=config.max_perc,
            max_masks=config.max_masks,
        )

    if config.augmentation_type == "mask_freq":
        return MaskFrequency(
            max_perc=config.max_perc,
            max_masks=config.max_masks,
        )

    raise NotImplementedError(
        "Invalid or unimplemented augmentation type: "
        f"{config.augmentation_type}"
    )


DEFAULT_AUGMENTATION_CONFIG: AugmentationsConfig = AugmentationsConfig(
    enabled=True,
    audio=[
        MixAugmentationConfig(),
        EchoAugmentationConfig(),
    ],
    spectrogram=[
        VolumeAugmentationConfig(),
        WarpAugmentationConfig(),
        TimeMaskAugmentationConfig(),
        FrequencyMaskAugmentationConfig(),
    ],
)


def build_augmentation_sequence(
    samplerate: int,
    steps: Optional[Sequence[AugmentationConfig]] = None,
    audio_source: Optional[AudioSource] = None,
) -> Optional[Augmentation]:
    if not steps:
        return None

    augmentations = []

    for step_config in steps:
        augmentation = build_augmentation_from_config(
            step_config,
            samplerate=samplerate,
            audio_source=audio_source,
        )

        if augmentation is None:
            continue

        augmentations.append(
            MaybeApply(
                augmentation=augmentation,
                probability=step_config.probability,
            )
        )

    return torch.nn.Sequential(*augmentations)


def build_augmentations(
    samplerate: int,
    config: Optional[AugmentationsConfig] = None,
    audio_source: Optional[AudioSource] = None,
) -> Tuple[Optional[Augmentation], Optional[Augmentation]]:
    """Build a composite augmentation pipeline function from configuration."""
    config = config or DEFAULT_AUGMENTATION_CONFIG

    logger.opt(lazy=True).debug(
        "Building augmentations with config: \n{}",
        lambda: config.to_yaml_string(),
    )

    audio_augmentation = build_augmentation_sequence(
        samplerate,
        steps=config.audio,
        audio_source=audio_source,
    )
    spectrogram_augmentation = build_augmentation_sequence(
        samplerate,
        steps=config.audio,
        audio_source=audio_source,
    )

    return audio_augmentation, spectrogram_augmentation


def load_augmentation_config(
    path: data.PathLike, field: Optional[str] = None
) -> AugmentationsConfig:
    """Load the augmentations configuration from a file."""
    return load_config(path, schema=AugmentationsConfig, field=field)


class RandomAudioSource:
    def __init__(
        self,
        clip_annotations: Sequence[data.ClipAnnotation],
        audio_loader: AudioLoader,
    ):
        self.audio_loader = audio_loader
        self.clip_annotations = clip_annotations

    def __call__(
        self,
        duration: float,
    ) -> Tuple[torch.Tensor, data.ClipAnnotation]:
        index = int(np.random.randint(len(self.clip_annotations)))
        clip_annotation = get_subclip_annotation(
            self.clip_annotations[index],
            duration=duration,
            max_empty=0,
        )
        wav = self.audio_loader.load_clip(clip_annotation.clip)
        return torch.from_numpy(wav).unsqueeze(0), clip_annotation
