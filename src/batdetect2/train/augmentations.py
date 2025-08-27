"""Applies data augmentation techniques to BatDetect2 training examples."""

import warnings
from typing import Annotated, Callable, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from loguru import logger
from pydantic import Field
from soundevent import data

from batdetect2.configs import BaseConfig, load_config
from batdetect2.typing import Augmentation, PreprocessorProtocol
from batdetect2.typing.train import PreprocessedExample
from batdetect2.utils.arrays import adjust_width

__all__ = [
    "AugmentationConfig",
    "AugmentationsConfig",
    "DEFAULT_AUGMENTATION_CONFIG",
    "EchoAugmentationConfig",
    "ExampleSource",
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
    "mix_examples",
    "scale_volume",
    "warp_spectrogram",
]

ExampleSource = Callable[[], PreprocessedExample]
"""Type alias for a function that returns a training example"""


class MixAugmentationConfig(BaseConfig):
    """Configuration for MixUp augmentation (mixing two examples)."""

    augmentation_type: Literal["mix_audio"] = "mix_audio"

    probability: float = 0.2
    """Probability of applying this augmentation to an example."""

    min_weight: float = 0.3
    """Minimum mixing weight (lambda) applied to the primary example."""

    max_weight: float = 0.7
    """Maximum mixing weight (lambda) applied to the primary example."""


def mix_examples(
    example: PreprocessedExample,
    other: PreprocessedExample,
    preprocessor: PreprocessorProtocol,
    weight: float,
) -> PreprocessedExample:
    """Combine two training examples."""
    audio1 = example.audio
    audio2 = adjust_width(other.audio, audio1.shape[-1])

    combined = weight * audio1 + (1 - weight) * audio2

    spectrogram = preprocessor(combined)

    # NOTE: The subclip's spectrogram might be slightly longer than the
    # spectrogram computed from the subclip's audio. This is due to a
    # simplification in the subclip process: It doesn't account for the
    # spectrogram parameters to precisely determine the corresponding audio
    # samples. To work around this, we pad the computed spectrogram with zeros
    # as needed.
    previous_width = example.spectrogram.shape[-1]
    spectrogram = adjust_width(spectrogram, previous_width)

    detection_heatmap = torch.maximum(
        example.detection_heatmap,
        adjust_width(other.detection_heatmap, previous_width),
    )

    class_heatmap = torch.maximum(
        example.class_heatmap,
        adjust_width(other.class_heatmap, previous_width),
    )

    size_heatmap = torch.maximum(
        example.size_heatmap,
        adjust_width(other.size_heatmap, previous_width),
    )

    return PreprocessedExample(
        audio=combined,
        spectrogram=spectrogram,
        detection_heatmap=detection_heatmap,
        class_heatmap=class_heatmap,
        size_heatmap=size_heatmap,
    )


class EchoAugmentationConfig(BaseConfig):
    """Configuration for adding synthetic echo/reverb."""

    augmentation_type: Literal["add_echo"] = "add_echo"

    probability: float = 0.2
    """Probability of applying this augmentation."""

    max_delay: float = 0.005
    min_weight: float = 0.0
    max_weight: float = 1.0


class AddEcho(torch.nn.Module):
    def __init__(
        self,
        preprocessor: PreprocessorProtocol,
        min_weight: float = 0.1,
        max_weight: float = 1.0,
        max_delay: float = 0.005,
    ):
        super().__init__()
        self.preprocessor = preprocessor
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.max_delay = max_delay

    def forward(self, example: PreprocessedExample) -> PreprocessedExample:
        delay = np.random.uniform(0, self.max_delay)
        weight = np.random.uniform(self.min_weight, self.max_weight)
        return add_echo(
            example,
            preprocessor=self.preprocessor,
            delay=delay,
            weight=weight,
        )


def add_echo(
    example: PreprocessedExample,
    preprocessor: PreprocessorProtocol,
    delay: float,
    weight: float,
) -> PreprocessedExample:
    """Add a synthetic echo to the audio waveform."""

    audio = example.audio
    delay_steps = int(preprocessor.input_samplerate * delay)
    audio_delay = adjust_width(audio[delay_steps:], audio.shape[-1])

    audio = audio + weight * audio_delay
    spectrogram = preprocessor(audio)

    # NOTE: The subclip's spectrogram might be slightly longer than the
    # spectrogram computed from the subclip's audio. This is due to a
    # simplification in the subclip process: It doesn't account for the
    # spectrogram parameters to precisely determine the corresponding audio
    # samples. To work around this, we pad the computed spectrogram with zeros
    # as needed.
    spectrogram = adjust_width(
        spectrogram,
        example.spectrogram.shape[-1],
    )

    return PreprocessedExample(
        audio=audio,
        spectrogram=spectrogram,
        detection_heatmap=example.detection_heatmap,
        class_heatmap=example.class_heatmap,
        size_heatmap=example.size_heatmap,
    )


class VolumeAugmentationConfig(BaseConfig):
    """Configuration for random volume scaling of the spectrogram."""

    augmentation_type: Literal["scale_volume"] = "scale_volume"
    probability: float = 0.2
    min_scaling: float = 0.0
    max_scaling: float = 2.0


class ScaleVolume(torch.nn.Module):
    def __init__(self, min_scaling: float, max_scaling: float):
        super().__init__()
        self.min_scaling = min_scaling
        self.max_scaling = max_scaling

    def forward(self, example: PreprocessedExample) -> PreprocessedExample:
        factor = np.random.uniform(self.min_scaling, self.max_scaling)
        return scale_volume(example, factor=factor)


def scale_volume(
    example: PreprocessedExample,
    factor: Optional[float] = None,
) -> PreprocessedExample:
    """Scale the amplitude of the spectrogram by a random factor."""
    return PreprocessedExample(
        audio=example.audio,
        size_heatmap=example.size_heatmap,
        class_heatmap=example.class_heatmap,
        detection_heatmap=example.detection_heatmap,
        spectrogram=example.spectrogram * factor,
    )


class WarpAugmentationConfig(BaseConfig):
    augmentation_type: Literal["warp"] = "warp"
    probability: float = 0.2
    delta: float = 0.04


class WarpSpectrogram(torch.nn.Module):
    def __init__(self, delta: float = 0.04) -> None:
        super().__init__()
        self.delta = delta

    def forward(self, example: PreprocessedExample) -> PreprocessedExample:
        factor = np.random.uniform(1 - self.delta, 1 + self.delta)
        return warp_spectrogram(example, factor=factor)


def warp_spectrogram(
    example: PreprocessedExample, factor: float
) -> PreprocessedExample:
    """Apply time warping by resampling the time axis."""
    target_shape = example.spectrogram.shape
    new_width = int(target_shape[-1] * factor)

    spectrogram = (
        torch.nn.functional.interpolate(
            adjust_width(example.spectrogram, new_width)
            .unsqueeze(0)
            .unsqueeze(0),
            size=target_shape,
            mode="bilinear",
        )
        .squeeze(0)
        .squeeze(0)
    )

    detection = (
        torch.nn.functional.interpolate(
            adjust_width(example.detection_heatmap, new_width)
            .unsqueeze(0)
            .unsqueeze(0),
            size=target_shape,
            mode="nearest",
        )
        .squeeze(0)
        .squeeze(0)
    )

    classification = torch.nn.functional.interpolate(
        adjust_width(example.class_heatmap, new_width).unsqueeze(1),
        size=target_shape,
        mode="nearest",
    ).squeeze(1)

    size = torch.nn.functional.interpolate(
        adjust_width(example.size_heatmap, new_width).unsqueeze(1),
        size=target_shape,
        mode="nearest",
    ).squeeze(1)

    return PreprocessedExample(
        audio=example.audio,
        size_heatmap=size,
        class_heatmap=classification,
        detection_heatmap=detection,
        spectrogram=spectrogram,
    )


class TimeMaskAugmentationConfig(BaseConfig):
    augmentation_type: Literal["mask_time"] = "mask_time"
    probability: float = 0.2
    max_perc: float = 0.05
    max_masks: int = 3


class MaskTime(torch.nn.Module):
    def __init__(self, max_perc: float = 0.05, max_masks: int = 3) -> None:
        super().__init__()
        self.max_perc = max_perc
        self.max_masks = max_masks

    def forward(self, example: PreprocessedExample) -> PreprocessedExample:
        num_masks = np.random.randint(1, self.max_masks + 1)
        width = example.spectrogram.shape[-1]

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
        return mask_time(example, masks)


def mask_time(
    example: PreprocessedExample,
    masks: List[Tuple[int, int]],
) -> PreprocessedExample:
    """Apply time masking to the spectrogram."""

    for start, end in masks:
        example.spectrogram[:, start:end] = example.spectrogram.mean()
        example.class_heatmap[:, :, start:end] = 0
        example.size_heatmap[:, :, start:end] = 0
        example.detection_heatmap[:, start:end] = 0

    return PreprocessedExample(
        audio=example.audio,
        size_heatmap=example.size_heatmap,
        class_heatmap=example.class_heatmap,
        detection_heatmap=example.detection_heatmap,
        spectrogram=example.spectrogram,
    )


class FrequencyMaskAugmentationConfig(BaseConfig):
    augmentation_type: Literal["mask_freq"] = "mask_freq"
    probability: float = 0.2
    max_perc: float = 0.10
    max_masks: int = 3


class MaskFrequency(torch.nn.Module):
    def __init__(self, max_perc: float = 0.10, max_masks: int = 3) -> None:
        super().__init__()
        self.max_perc = max_perc
        self.max_masks = max_masks

    def forward(self, example: PreprocessedExample) -> PreprocessedExample:
        num_masks = np.random.randint(1, self.max_masks + 1)
        height = example.spectrogram.shape[-2]

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
        return mask_frequency(example, masks)


def mask_frequency(
    example: PreprocessedExample,
    masks: List[Tuple[int, int]],
) -> PreprocessedExample:
    """Apply frequency masking to the spectrogram."""
    for start, end in masks:
        example.spectrogram[start:end, :] = example.spectrogram.mean()
        example.class_heatmap[:, start:end, :] = 0
        example.size_heatmap[:, start:end, :] = 0
        example.detection_heatmap[start:end, :] = 0

    return PreprocessedExample(
        audio=example.audio,
        size_heatmap=example.size_heatmap,
        class_heatmap=example.class_heatmap,
        detection_heatmap=example.detection_heatmap,
        spectrogram=example.spectrogram,
    )


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

    steps: List[AugmentationConfig] = Field(default_factory=list)


class MaybeApply(torch.nn.Module):
    """Applies an augmentation function probabilistically."""

    def __init__(
        self,
        augmentation: Augmentation,
        probability: float = 0.2,
    ):
        """Initialize the wrapper.

        Parameters
        ----------
        augmentation : Augmentation (Callable[[xr.Dataset], xr.Dataset])
            The augmentation function to potentially apply.
        probability : float, default=0.5
            The probability (0.0 to 1.0) of applying the augmentation.
        """
        super().__init__()
        self.augmentation = augmentation
        self.probability = probability

    def __call__(self, example: PreprocessedExample) -> PreprocessedExample:
        """Apply the wrapped augmentation with configured probability.

        Parameters
        ----------
        example : xr.Dataset
            The input training example.

        Returns
        -------
        xr.Dataset
            The potentially augmented training example.
        """
        if np.random.random() > self.probability:
            return example

        return self.augmentation(example)


class AudioMixer(torch.nn.Module):
    """Callable class for MixUp augmentation, handling example fetching."""

    def __init__(
        self,
        min_weight: float,
        max_weight: float,
        example_source: ExampleSource,
        preprocessor: PreprocessorProtocol,
    ):
        """Initialize the AudioMixer."""
        super().__init__()
        self.min_weight = min_weight
        self.example_source = example_source
        self.max_weight = max_weight
        self.preprocessor = preprocessor

    def __call__(self, example: PreprocessedExample) -> PreprocessedExample:
        """Fetch another example and perform mixup."""
        other = self.example_source()
        weight = np.random.uniform(self.min_weight, self.max_weight)
        return mix_examples(
            example,
            other,
            self.preprocessor,
            weight=weight,
        )


def build_augmentation_from_config(
    config: AugmentationConfig,
    preprocessor: PreprocessorProtocol,
    example_source: Optional[ExampleSource] = None,
) -> Optional[Augmentation]:
    """Factory function to build a single augmentation from its config."""
    if config.augmentation_type == "mix_audio":
        if example_source is None:
            warnings.warn(
                "Mix audio augmentation ('mix_audio') requires an "
                "'example_source' callable to be provided.",
                stacklevel=2,
            )
            return None

        return AudioMixer(
            example_source=example_source,
            preprocessor=preprocessor,
            min_weight=config.min_weight,
            max_weight=config.max_weight,
        )

    if config.augmentation_type == "add_echo":
        return AddEcho(
            preprocessor=preprocessor,
            max_delay=config.max_delay,
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
    steps=[
        MixAugmentationConfig(),
        EchoAugmentationConfig(),
        VolumeAugmentationConfig(),
        WarpAugmentationConfig(),
        TimeMaskAugmentationConfig(),
        FrequencyMaskAugmentationConfig(),
    ]
)


def build_augmentations(
    preprocessor: PreprocessorProtocol,
    config: Optional[AugmentationsConfig] = None,
    example_source: Optional[ExampleSource] = None,
) -> Augmentation:
    """Build a composite augmentation pipeline function from configuration."""
    config = config or DEFAULT_AUGMENTATION_CONFIG

    logger.opt(lazy=True).debug(
        "Building augmentations with config: \n{}",
        lambda: config.to_yaml_string(),
    )

    augmentations = []

    for step_config in config.steps:
        augmentation = build_augmentation_from_config(
            step_config,
            preprocessor=preprocessor,
            example_source=example_source,
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


def load_augmentation_config(
    path: data.PathLike, field: Optional[str] = None
) -> AugmentationsConfig:
    """Load the augmentations configuration from a file."""
    return load_config(path, schema=AugmentationsConfig, field=field)
