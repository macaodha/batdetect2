"""Generate heatmap training targets for BatDetect2 models.

This module is responsible for creating the target labels used for training
BatDetect2 models. It converts sound event annotations for an audio clip into
the specific multi-channel heatmap formats required by the neural network.
"""

from functools import partial

import numpy as np
import torch
from loguru import logger
from soundevent import data

from batdetect2.core.configs import BaseConfig
from batdetect2.preprocess import MAX_FREQ, MIN_FREQ
from batdetect2.targets import (
    build_roi_mapping,
    build_targets,
    iterate_encoded_sound_events,
)
from batdetect2.targets.types import ROIMapperProtocol, TargetProtocol
from batdetect2.train.types import ClipLabeller, Heatmaps

__all__ = [
    "LabelConfig",
    "build_clip_labeler",
    "generate_heatmaps",
]


SIZE_DIMENSION = "dimension"
"""Dimension name for the size heatmap."""


class LabelConfig(BaseConfig):
    """Configuration parameters for heatmap generation.

    Attributes
    ----------
    sigma : float, default=3.0
    """

    sigma: float = 2.0


def build_clip_labeler(
    targets: TargetProtocol | None = None,
    roi_mapper: ROIMapperProtocol | None = None,
    min_freq: float = MIN_FREQ,
    max_freq: float = MAX_FREQ,
    config: LabelConfig | None = None,
) -> ClipLabeller:
    """Construct the final clip labelling function."""
    config = config or LabelConfig()
    logger.opt(lazy=True).debug(
        "Building clip labeler with config: \n{}",
        lambda: config.to_yaml_string(),
    )

    targets = targets or build_targets()
    roi_mapper = roi_mapper or build_roi_mapping()

    return partial(
        generate_heatmaps,
        targets=targets,
        roi_mapper=roi_mapper,
        min_freq=min_freq,
        max_freq=max_freq,
        target_sigma=config.sigma,
    )


def map_to_pixels(x, size, min_val, max_val) -> int:
    return int(np.interp(x, [min_val, max_val], [0, size]))


def generate_heatmaps(
    clip_annotation: data.ClipAnnotation,
    spec: torch.Tensor,
    targets: TargetProtocol,
    roi_mapper: ROIMapperProtocol,
    min_freq: float,
    max_freq: float,
    target_sigma: float = 3.0,
    dtype=torch.float32,
) -> Heatmaps:
    """Generate training heatmaps for a single annotated clip."""
    logger.debug(
        "Will generate heatmaps for clip annotation "
        "{uuid} with {num} annotated sound events",
        uuid=clip_annotation.uuid,
        num=len(clip_annotation.sound_events),
    )

    height = spec.shape[-2]
    width = spec.shape[-1]
    num_classes = len(targets.class_names)
    num_dims = len(roi_mapper.dimension_names)
    clip = clip_annotation.clip

    # Initialize heatmaps
    detection_heatmap = torch.zeros([1, height, width], dtype=dtype)
    class_heatmap = torch.zeros([num_classes, height, width], dtype=dtype)
    size_heatmap = torch.zeros([num_dims, height, width], dtype=dtype)

    freqs, times = torch.meshgrid(
        torch.arange(height, dtype=dtype),
        torch.arange(width, dtype=dtype),
        indexing="ij",
    )

    freqs = freqs.to(spec.device)
    times = times.to(spec.device)

    for class_name, (time, frequency), size in iterate_encoded_sound_events(
        clip_annotation.sound_events,
        targets,
        roi_mapper,
    ):
        time_index = map_to_pixels(time, width, clip.start_time, clip.end_time)
        freq_index = map_to_pixels(frequency, height, min_freq, max_freq)

        if (
            time_index < 0
            or time_index >= width
            or freq_index < 0
            or freq_index >= height
        ):
            logger.debug(
                "Skipping annotation: position outside spectrogram. Pos: %s",
                (time, frequency),
            )
            continue

        distance = (times - time_index) ** 2 + (freqs - freq_index) ** 2
        gaussian_blob = torch.exp(-distance / (2 * target_sigma**2))

        detection_heatmap[0] = torch.maximum(
            detection_heatmap[0],
            gaussian_blob,
        )
        size_heatmap[:, freq_index, time_index] = torch.tensor(size[:])

        # If the label is None skip the sound event
        if class_name is None:
            continue

        class_index = targets.class_names.index(class_name)
        class_heatmap[class_index] = torch.maximum(
            class_heatmap[class_index],
            gaussian_blob,
        )

    return Heatmaps(
        detection=detection_heatmap,
        classes=class_heatmap,
        size=size_heatmap,
    )
