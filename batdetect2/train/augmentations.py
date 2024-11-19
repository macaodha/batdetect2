from typing import Callable, Optional, Union

import numpy as np
import xarray as xr
from pydantic import Field
from soundevent import arrays
from soundevent.arrays import operations as ops

from batdetect2.configs import BaseConfig
from batdetect2.preprocess import PreprocessingConfig, compute_spectrogram

Augmentation = Callable[[xr.Dataset], xr.Dataset]


class AugmentationConfig(BaseConfig):
    enable: bool = True
    probability: float = 0.2


class SubclipConfig(BaseConfig):
    enable: bool = True
    duration: Optional[float] = None


def select_random_subclip(
    example: xr.Dataset,
    start_time: Optional[float] = None,
    duration: Optional[float] = None,
    width: Optional[int] = None,
) -> xr.Dataset:
    """Select a random subclip from a clip."""
    step = arrays.get_dim_step(example, "time")  # type: ignore

    if width is None:
        if duration is None:
            raise ValueError("Either duration or width must be provided")

        width = int(np.floor(duration / step))

    if duration is None:
        duration = width * step

    if start_time is None:
        start, stop = arrays.get_dim_range(example, "time")  # type: ignore
        start_time = np.random.uniform(start, stop - duration)

    start_index = arrays.get_coord_index(
        example,  # type: ignore
        "time",
        start_time,
    )
    end_index = start_index + width - 1
    start_time = example.time.values[start_index]
    end_time = example.time.values[end_index]

    return example.sel(
        time=slice(start_time, end_time),
        audio_time=slice(start_time, end_time),
    )


class MixAugmentationConfig(AugmentationConfig):
    min_weight: float = 0.3
    max_weight: float = 0.7


def mix_examples(
    example: xr.Dataset,
    other: xr.Dataset,
    weight: Optional[float] = None,
    min_weight: float = 0.3,
    max_weight: float = 0.7,
    config: Optional[PreprocessingConfig] = None,
) -> xr.Dataset:
    """Combine two audio clips."""
    config = config or PreprocessingConfig()

    if weight is None:
        weight = np.random.uniform(min_weight, max_weight)

    audio2 = other["audio"].values
    audio1 = ops.adjust_dim_width(example["audio"], "audio_time", len(audio2))
    combined = weight * audio1 + (1 - weight) * audio2

    spec = compute_spectrogram(
        combined.rename({"audio_time": "time"}),
        config=config.spectrogram,
    )

    detection_heatmap = xr.apply_ufunc(
        np.maximum,
        example["detection"],
        other["detection"].values,
    )

    class_heatmap = xr.apply_ufunc(
        np.maximum,
        example["class"],
        other["class"].values,
    )

    size_heatmap = example["size"] + other["size"].values

    return xr.Dataset(
        {
            "audio": combined,
            "spectrogram": spec,
            "detection": detection_heatmap,
            "class": class_heatmap,
            "size": size_heatmap,
        }
    )


class EchoAugmentationConfig(AugmentationConfig):
    max_delay: float = 0.005
    min_weight: float = 0.0
    max_weight: float = 1.0


def add_echo(
    example: xr.Dataset,
    delay: Optional[float] = None,
    weight: Optional[float] = None,
    min_weight: float = 0.1,
    max_weight: float = 1.0,
    max_delay: float = 0.005,
    config: Optional[PreprocessingConfig] = None,
) -> xr.Dataset:
    """Add a delay to the audio."""
    config = config or PreprocessingConfig()

    if delay is None:
        delay = np.random.uniform(0, max_delay)

    if weight is None:
        weight = np.random.uniform(min_weight, max_weight)

    audio = example["audio"]
    step = arrays.get_dim_step(audio, "audio_time")
    audio_delay = audio.shift(audio_time=int(delay / step), fill_value=0)
    audio = audio + weight * audio_delay

    spectrogram = compute_spectrogram(
        audio.rename({"audio_time": "time"}),
        config=config.spectrogram,
    )

    return example.assign(audio=audio, spectrogram=spectrogram)


class VolumeAugmentationConfig(AugmentationConfig):
    min_scaling: float = 0.0
    max_scaling: float = 2.0


def scale_volume(
    example: xr.Dataset,
    factor: Optional[float] = None,
    max_scaling: float = 2,
    min_scaling: float = 0,
) -> xr.Dataset:
    """Scale the volume of a spectrogram."""
    if factor is None:
        factor = np.random.uniform(min_scaling, max_scaling)

    return example.assign(spectrogram=example["spectrogram"] * factor)


class WarpAugmentationConfig(AugmentationConfig):
    delta: float = 0.04


def warp_spectrogram(
    example: xr.Dataset,
    factor: Optional[float] = None,
    delta: float = 0.04,
) -> xr.Dataset:
    """Warp a spectrogram."""
    if factor is None:
        factor = np.random.uniform(1 - delta, 1 + delta)

    start_time, end_time = arrays.get_dim_range(example, "time")  # type: ignore
    duration = end_time - start_time

    new_time = np.linspace(
        start_time,
        start_time + duration * factor,
        example.time.size,
    )

    spectrogram = (
        example["spectrogram"]
        .interp(
            coords={"time": new_time},
            method="linear",
            kwargs=dict(
                fill_value=0,
            ),
        )
        .clip(min=0)
    )

    detection = example["detection"].interp(
        time=new_time,
        method="nearest",
        kwargs=dict(
            fill_value=0,
        ),
    )

    classification = example["class"].interp(
        time=new_time,
        method="nearest",
        kwargs=dict(
            fill_value=0,
        ),
    )

    size = example["size"].interp(
        time=new_time,
        method="nearest",
        kwargs=dict(
            fill_value=0,
        ),
    )

    return example.assign(
        {
            "time": new_time,
            "spectrogram": spectrogram,
            "detection": detection,
            "class": classification,
            "size": size,
        }
    )


def mask_axis(
    array: xr.DataArray,
    dim: str,
    start: float,
    end: float,
    mask_value: Union[float, Callable[[xr.DataArray], float]] = np.mean,
) -> xr.DataArray:
    if dim not in array.dims:
        raise ValueError(f"Axis {dim} not found in array")

    coord = array.coords[dim]
    condition = (coord < start) | (coord > end)

    if callable(mask_value):
        mask_value = mask_value(array)

    return array.where(condition, other=mask_value)


class TimeMaskAugmentationConfig(AugmentationConfig):
    max_perc: float = 0.05
    max_masks: int = 3


def mask_time(
    example: xr.Dataset,
    max_perc: float = 0.05,
    max_mask: int = 3,
) -> xr.Dataset:
    """Mask a random section of the time axis."""
    num_masks = np.random.randint(1, max_mask + 1)
    start_time, end_time = arrays.get_dim_range(example, "time")  # type: ignore

    spectrogram = example["spectrogram"]
    for _ in range(num_masks):
        mask_size = np.random.uniform(0, max_perc) * (end_time - start_time)
        start = np.random.uniform(start_time, end_time - mask_size)
        end = start + mask_size
        spectrogram = mask_axis(spectrogram, "time", start, end)

    return example.assign(spectrogram=spectrogram)


class FrequencyMaskAugmentationConfig(AugmentationConfig):
    max_perc: float = 0.10
    max_masks: int = 3


def mask_frequency(
    example: xr.Dataset,
    max_perc: float = 0.10,
    max_masks: int = 3,
) -> xr.Dataset:
    """Mask a random section of the frequency axis."""
    num_masks = np.random.randint(1, max_masks + 1)
    min_freq, max_freq = arrays.get_dim_range(example, "frequency")  # type: ignore

    spectrogram = example["spectrogram"]
    for _ in range(num_masks):
        mask_size = np.random.uniform(0, max_perc) * (max_freq - min_freq)
        start = np.random.uniform(min_freq, max_freq - mask_size)
        end = start + mask_size
        spectrogram = mask_axis(spectrogram, "frequency", start, end)

    return example.assign(spectrogram=spectrogram)


class AugmentationsConfig(BaseConfig):
    subclip: SubclipConfig = Field(default_factory=SubclipConfig)
    mix: MixAugmentationConfig = Field(default_factory=MixAugmentationConfig)
    echo: EchoAugmentationConfig = Field(
        default_factory=EchoAugmentationConfig
    )
    volume: VolumeAugmentationConfig = Field(
        default_factory=VolumeAugmentationConfig
    )
    warp: WarpAugmentationConfig = Field(
        default_factory=WarpAugmentationConfig
    )
    time_mask: TimeMaskAugmentationConfig = Field(
        default_factory=TimeMaskAugmentationConfig
    )
    frequency_mask: FrequencyMaskAugmentationConfig = Field(
        default_factory=FrequencyMaskAugmentationConfig
    )


def should_apply(config: AugmentationConfig) -> bool:
    if not config.enable:
        return False

    return np.random.uniform() < config.probability


def augment_example(
    example: xr.Dataset,
    config: AugmentationsConfig,
    preprocessing_config: Optional[PreprocessingConfig] = None,
    others: Optional[Callable[[], xr.Dataset]] = None,
) -> xr.Dataset:
    if config.subclip.enable:
        example = select_random_subclip(
            example,
            duration=config.subclip.duration,
        )

    if should_apply(config.mix) and others is not None:
        other = others()

        if config.subclip.enable:
            other = select_random_subclip(
                other,
                duration=config.subclip.duration,
            )

        example = mix_examples(
            example,
            other,
            min_weight=config.mix.min_weight,
            max_weight=config.mix.max_weight,
            config=preprocessing_config,
        )

    if should_apply(config.echo):
        example = add_echo(
            example,
            max_delay=config.echo.max_delay,
            min_weight=config.echo.min_weight,
            max_weight=config.echo.max_weight,
            config=preprocessing_config,
        )

    if should_apply(config.volume):
        example = scale_volume(
            example,
            max_scaling=config.volume.max_scaling,
            min_scaling=config.volume.min_scaling,
        )

    if should_apply(config.warp):
        example = warp_spectrogram(
            example,
            delta=config.warp.delta,
        )

    if should_apply(config.time_mask):
        example = mask_time(
            example,
            max_perc=config.time_mask.max_perc,
            max_mask=config.time_mask.max_masks,
        )

    if should_apply(config.frequency_mask):
        example = mask_frequency(
            example,
            max_perc=config.frequency_mask.max_perc,
            max_masks=config.frequency_mask.max_masks,
        )

    return example
