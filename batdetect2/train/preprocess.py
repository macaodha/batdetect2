"""Module for preprocessing data for training."""

import os
import warnings
from functools import partial
from pathlib import Path
from typing import Callable, Optional, Sequence, Union
from tqdm.auto import tqdm
from multiprocessing import Pool

import xarray as xr
from soundevent import data

from batdetect2.data.labels import TARGET_SIGMA, ClassMapper, generate_heatmaps
from batdetect2.data.preprocessing import (
    preprocess_audio_clip,
    PreprocessingConfig,
)

PathLike = Union[Path, str, os.PathLike]
FilenameFn = Callable[[data.ClipAnnotation], str]

__all__ = [
    "preprocess_annotations",
]



def generate_train_example(
    clip_annotation: data.ClipAnnotation,
    class_mapper: ClassMapper,
    preprocessing_config: PreprocessingConfig = PreprocessingConfig(),
    target_sigma: float = TARGET_SIGMA,
) -> xr.Dataset:
    """Generate a training example."""
    spectrogram = preprocess_audio_clip(
        clip_annotation.clip,
        config=preprocessing_config,
    )

    detection_heatmap, class_heatmap, size_heatmap = generate_heatmaps(
        clip_annotation,
        spectrogram,
        class_mapper,
        target_sigma=target_sigma,
    )

    dataset = xr.Dataset(
        {
            "spectrogram": spectrogram,
            "detection": detection_heatmap,
            "class": class_heatmap,
            "size": size_heatmap,
        }
    )

    return dataset.assign_attrs(
        title=f"Training example for {clip_annotation.uuid}",
        preprocessing_configuration=preprocessing_config.model_dump_json(),
        target_sigma=target_sigma,
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


def load_config(path: PathLike, **kwargs) -> PreprocessingConfig:
    """Load configuration from file."""

    path = Path(path)

    if not path.is_file():
        warnings.warn(f"Config file not found: {path}. Using default config.")
        return PreprocessingConfig(**kwargs)

    try:
        return PreprocessingConfig.model_validate_json(path.read_text())
    except ValueError as e:
        warnings.warn(
            f"Failed to load config file: {e}. Using default config."
        )
        return PreprocessingConfig(**kwargs)


def _get_filename(clip_annotation: data.ClipAnnotation) -> str:
    return f"{clip_annotation.uuid}.nc"


def preprocess_single_annotation(
    clip_annotation: data.ClipAnnotation,
    output_dir: PathLike,
    config: PreprocessingConfig,
    class_mapper: ClassMapper,
    filename_fn: FilenameFn = _get_filename,
    replace: bool = False,
    target_sigma: float = TARGET_SIGMA,
) -> None:
    output_dir = Path(output_dir)

    filename = filename_fn(clip_annotation)
    path = output_dir / filename

    if path.is_file() and not replace:
        return

    if path.is_file() and replace:
        path.unlink()

    sample = generate_train_example(
        clip_annotation,
        class_mapper,
        preprocessing_config=config,
        target_sigma=target_sigma,
    )

    save_to_file(sample, path)


def preprocess_annotations(
    clip_annotations: Sequence[data.ClipAnnotation],
    output_dir: PathLike,
    class_mapper: ClassMapper,
    target_sigma: float = TARGET_SIGMA,
    filename_fn: FilenameFn = _get_filename,
    replace: bool = False,
    config: Optional[PreprocessingConfig] = None,
    max_workers: Optional[int] = None,
) -> None:
    """Preprocess annotations and save to disk."""
    output_dir = Path(output_dir)

    if config is None:
        config = PreprocessingConfig()

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
                        class_mapper=class_mapper,
                        filename_fn=filename_fn,
                        replace=replace,
                        target_sigma=target_sigma,
                    ),
                    clip_annotations,
                ),
                total=len(clip_annotations),
            )
        )
