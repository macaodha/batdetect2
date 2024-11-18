"""Module for preprocessing data for training."""

import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Callable, Optional, Sequence, Union

import xarray as xr
from pydantic import Field
from soundevent import data
from tqdm.auto import tqdm

from batdetect2.configs import BaseConfig
from batdetect2.preprocess import (
    PreprocessingConfig,
    compute_spectrogram,
    load_clip_audio,
)
from batdetect2.train.labels import HeatmapsConfig, generate_heatmaps
from batdetect2.train.targets import (
    TargetConfig,
    build_class_mapper,
    build_sound_event_filter,
)

PathLike = Union[Path, str, os.PathLike]
FilenameFn = Callable[[data.ClipAnnotation], str]

__all__ = [
    "preprocess_annotations",
]


class TrainPreprocessingConfig(BaseConfig):
    preprocessing: PreprocessingConfig = Field(
        default_factory=PreprocessingConfig
    )
    target: TargetConfig = Field(default_factory=TargetConfig)
    heatmaps: HeatmapsConfig = Field(default_factory=HeatmapsConfig)


def generate_train_example(
    clip_annotation: data.ClipAnnotation,
    config: Optional[TrainPreprocessingConfig] = None,
) -> xr.Dataset:
    """Generate a training example."""
    config = config or TrainPreprocessingConfig()

    wave = load_clip_audio(
        clip_annotation.clip,
        config=config.preprocessing.audio,
    )

    spectrogram = compute_spectrogram(
        wave,
        config=config.preprocessing.spectrogram,
    )

    filter_fn = build_sound_event_filter(
        include=config.target.include,
        exclude=config.target.exclude,
    )
    selected_events = [
        event for event in clip_annotation.sound_events if filter_fn(event)
    ]
    class_mapper = build_class_mapper(config.target.classes)
    detection_heatmap, class_heatmap, size_heatmap = generate_heatmaps(
        selected_events,
        spectrogram,
        class_mapper,
        target_sigma=config.heatmaps.sigma,
        position=config.heatmaps.position,
        time_scale=config.heatmaps.time_scale,
        frequency_scale=config.heatmaps.frequency_scale,
    )

    dataset = xr.Dataset(
        {
            # NOTE: Need to rename the time dimension to avoid conflicts with
            # the spectrogram time dimension, otherwise xarray will interpolate
            # the spectrogram and the heatmaps to the same temporal resolution
            # as the waveform.
            "audio": wave.rename({"time": "audio_time"}),
            "spectrogram": spectrogram,
            "detection": detection_heatmap,
            "class": class_heatmap,
            "size": size_heatmap,
        }
    )

    return dataset.assign_attrs(
        title=f"Training example for {clip_annotation.uuid}",
        config=config.model_dump_json(),
        clip_annotation=clip_annotation.model_dump_json(),
    )


def save_to_file(
    dataset: xr.Dataset,
    path: PathLike,
) -> None:
    dataset.to_netcdf(
        path,
        encoding={
            "spectrogram": {"zlib": True},
            "size": {"zlib": True},
            "class": {"zlib": True},
            "detection": {"zlib": True},
        },
    )


def _get_filename(clip_annotation: data.ClipAnnotation) -> str:
    return f"{clip_annotation.uuid}.nc"


def preprocess_annotations(
    clip_annotations: Sequence[data.ClipAnnotation],
    output_dir: PathLike,
    filename_fn: FilenameFn = _get_filename,
    replace: bool = False,
    config: Optional[TrainPreprocessingConfig] = None,
    max_workers: Optional[int] = None,
) -> None:
    """Preprocess annotations and save to disk."""
    output_dir = Path(output_dir)

    config = config or TrainPreprocessingConfig()

    if not output_dir.is_dir():
        output_dir.mkdir(parents=True)

    with Pool(max_workers) as pool:
        list(
            tqdm(
                pool.imap_unordered(
                    partial(
                        preprocess_single_annotation,
                        output_dir=output_dir,
                        config=config,
                        filename_fn=filename_fn,
                        replace=replace,
                    ),
                    clip_annotations,
                ),
                total=len(clip_annotations),
            )
        )


def preprocess_single_annotation(
    clip_annotation: data.ClipAnnotation,
    output_dir: PathLike,
    config: TrainPreprocessingConfig,
    filename_fn: FilenameFn = _get_filename,
    replace: bool = False,
) -> None:
    output_dir = Path(output_dir)

    filename = filename_fn(clip_annotation)
    path = output_dir / filename

    if path.is_file() and not replace:
        return

    if path.is_file() and replace:
        path.unlink()

    sample = generate_train_example(clip_annotation, config=config)
    save_to_file(sample, path)
