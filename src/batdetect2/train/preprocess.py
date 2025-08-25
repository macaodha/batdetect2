"""Preprocesses datasets for BatDetect2 model training.

This module provides functions to take a collection of annotated audio clips
(`soundevent.data.ClipAnnotation`) and process them into the final format
required for training a BatDetect2 model. This typically involves:

1.  Loading the relevant audio segment for each annotation using a configured
    `PreprocessorProtocol`.
2.  Generating the corresponding input spectrogram using the
    `PreprocessorProtocol`.
3.  Generating the target heatmaps (detection, classification, size) using a
    configured `ClipLabeller` (which encapsulates the `TargetProtocol` logic).
4.  Packaging the input spectrogram, target heatmaps, and potentially the
    processed audio waveform into an `xarray.Dataset`.
5.  Saving each processed `xarray.Dataset` to a separate file (typically NetCDF)
    in an output directory.

This offline preprocessing is often preferred for large datasets as it avoids
computationally intensive steps during the actual training loop. The module
includes utilities for parallel processing using `multiprocessing`.
"""

from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import torch
import xarray as xr
from loguru import logger
from pydantic import Field
from soundevent import data
from tqdm.auto import tqdm

from batdetect2.configs import BaseConfig, load_config
from batdetect2.data.datasets import Dataset
from batdetect2.preprocess import PreprocessingConfig, build_preprocessor
from batdetect2.preprocess.audio import build_audio_loader
from batdetect2.targets import TargetConfig, build_targets
from batdetect2.train.labels import LabelConfig, build_clip_labeler
from batdetect2.typing import ClipLabeller, PreprocessorProtocol
from batdetect2.typing.preprocess import AudioLoader
from batdetect2.utils.arrays import audio_to_xarray

__all__ = [
    "preprocess_annotations",
    "preprocess_single_annotation",
    "generate_train_example",
    "preprocess_dataset",
    "TrainPreprocessConfig",
    "load_train_preprocessing_config",
]

FilenameFn = Callable[[data.ClipAnnotation], str]
"""Type alias for a function that generates an output filename."""


class TrainPreprocessConfig(BaseConfig):
    preprocess: PreprocessingConfig = Field(
        default_factory=PreprocessingConfig
    )
    targets: TargetConfig = Field(default_factory=TargetConfig)
    labels: LabelConfig = Field(default_factory=LabelConfig)


def load_train_preprocessing_config(
    path: data.PathLike,
    field: Optional[str] = None,
) -> TrainPreprocessConfig:
    return load_config(path=path, schema=TrainPreprocessConfig, field=field)


def preprocess_dataset(
    dataset: Dataset,
    config: TrainPreprocessConfig,
    output: Path,
    force: bool = False,
    max_workers: Optional[int] = None,
) -> None:
    targets = build_targets(config=config.targets)
    preprocessor = build_preprocessor(config=config.preprocess)
    labeller = build_clip_labeler(targets, config=config.labels)
    audio_loader = build_audio_loader(config=config.preprocess.audio)

    if not output.exists():
        logger.debug("Creating directory {directory}", directory=output)
        output.mkdir(parents=True)

    preprocess_annotations(
        dataset,
        output_dir=output,
        audio_loader=audio_loader,
        preprocessor=preprocessor,
        labeller=labeller,
        replace=force,
        max_workers=max_workers,
    )


def generate_train_example(
    clip_annotation: data.ClipAnnotation,
    audio_loader: AudioLoader,
    preprocessor: PreprocessorProtocol,
    labeller: ClipLabeller,
) -> xr.Dataset:
    """Generate a complete training example for one annotation.

    This function takes a single `ClipAnnotation`, applies the configured
    preprocessing (`PreprocessorProtocol`) to get the processed waveform and
    input spectrogram, applies the configured target generation
    (`ClipLabeller`) to get the target heatmaps, and packages them all into a
    single `xr.Dataset`.

    Parameters
    ----------
    clip_annotation : data.ClipAnnotation
        The annotated clip to process. Contains the reference to the `Clip`
        (audio segment) and the associated `SoundEventAnnotation` objects.
    preprocessor : PreprocessorProtocol
        An initialized preprocessor object responsible for loading/processing
        audio and computing the input spectrogram.
    labeller : ClipLabeller
        An initialized clip labeller function responsible for generating the
        target heatmaps (detection, class, size) from the `clip_annotation`
        and the computed spectrogram.

    Returns
    -------
    xr.Dataset
        An xarray Dataset containing the following data variables:
        - `audio`: The preprocessed audio waveform (dims: 'audio_time').
        - `spectrogram`: The computed input spectrogram
          (dims: 'time', 'frequency').
        - `detection`: The target detection heatmap
          (dims: 'time', 'frequency').
        - `class`: The target class heatmap
          (dims: 'category', 'time', 'frequency').
        - `size`: The target size heatmap
          (dims: 'dimension', 'time', 'frequency').
        The Dataset also includes metadata in its attributes.

    Notes
    -----
    - The 'time' dimension of the 'audio' DataArray is renamed to 'audio_time'
      within the output Dataset to avoid coordinate conflicts with the
      spectrogram's 'time' dimension when stored together.
    - The original `ClipAnnotation` metadata is stored as a JSON string in the
      Dataset's attributes for provenance.
    """
    wave = audio_loader.load_clip(clip_annotation.clip)

    spectrogram = _spec_to_xr(
        preprocessor(torch.tensor(wave)),
        start_time=clip_annotation.clip.start_time,
        end_time=clip_annotation.clip.end_time,
        min_freq=preprocessor.min_freq,
        max_freq=preprocessor.max_freq,
    )

    heatmaps = labeller(clip_annotation, spectrogram)

    dataset = xr.Dataset(
        {
            # NOTE: Need to rename the time dimension to avoid conflicts with
            # the spectrogram time dimension, otherwise xarray will interpolate
            # the spectrogram and the heatmaps to the same temporal resolution
            # as the waveform.
            "audio": audio_to_xarray(
                wave,
                start_time=clip_annotation.clip.start_time,
                end_time=clip_annotation.clip.end_time,
                time_axis="audio_time",
            ),
            "spectrogram": spectrogram,
            "detection": heatmaps.detection,
            "class": heatmaps.classes,
            "size": heatmaps.size,
        }
    )

    return dataset.assign_attrs(
        title=f"Training example for {clip_annotation.uuid}",
        clip_annotation=clip_annotation.model_dump_json(
            exclude_none=True,
            exclude_defaults=True,
            exclude_unset=True,
        ),
    )


def _spec_to_xr(
    spec: torch.Tensor,
    start_time: float,
    end_time: float,
    min_freq: float,
    max_freq: float,
) -> xr.DataArray:
    data = spec.numpy()[0, 0]

    height, width = data.shape

    return xr.DataArray(
        data=data,
        dims=[
            "frequency",
            "time",
        ],
        coords={
            "frequency": np.linspace(
                min_freq, max_freq, height, endpoint=False
            ),
            "time": np.linspace(start_time, end_time, width, endpoint=False),
        },
    )


def _save_xr_dataset_to_file(
    dataset: xr.Dataset,
    path: data.PathLike,
) -> None:
    """Save an xarray Dataset to a NetCDF file with compression.

    Internal helper function used by `preprocess_single_annotation`.

    Parameters
    ----------
    dataset : xr.Dataset
        The training example dataset to save.
    path : PathLike
        The output file path (e.g., 'output/uuid.nc').
    """
    dataset.to_netcdf(
        path,
        encoding={
            "audio": {"zlib": True},
            "spectrogram": {"zlib": True},
            "size": {"zlib": True},
            "class": {"zlib": True},
            "detection": {"zlib": True},
        },
    )


def _get_filename(clip_annotation: data.ClipAnnotation) -> str:
    """Generate a default output filename based on the annotation UUID."""
    return f"{clip_annotation.uuid}.nc"


def preprocess_annotations(
    clip_annotations: Sequence[data.ClipAnnotation],
    output_dir: data.PathLike,
    preprocessor: PreprocessorProtocol,
    audio_loader: AudioLoader,
    labeller: ClipLabeller,
    filename_fn: FilenameFn = _get_filename,
    replace: bool = False,
    max_workers: Optional[int] = None,
) -> None:
    """Preprocess a sequence of ClipAnnotations and save results to disk.

    Generates the full training example (spectrogram, heatmaps, etc.) for each
    `ClipAnnotation` in the input sequence using the provided `preprocessor`
    and `labeller`. Saves each example as a separate NetCDF file in the
    `output_dir`. Utilizes multiprocessing for potentially faster processing.

    Parameters
    ----------
    clip_annotations : Sequence[data.ClipAnnotation]
        A sequence (e.g., list) of the clip annotations to preprocess.
    output_dir : PathLike
        Path to the directory where the processed NetCDF files will be saved.
        Will be created if it doesn't exist.
    preprocessor : PreprocessorProtocol
        Initialized preprocessor object to generate spectrograms.
    labeller : ClipLabeller
        Initialized labeller function to generate target heatmaps.
    filename_fn : FilenameFn, optional
        Function to generate the output filename (without extension) for each
        `ClipAnnotation`. Defaults to using the annotation UUID via
        `_get_filename`.
    replace : bool, default=False
        If True, existing files in `output_dir` with the same generated name
        will be overwritten. If False (default), existing files are skipped.
    max_workers : int, optional
        Maximum number of worker processes to use for parallel processing.
        If None (default), uses the number of CPUs available (`os.cpu_count()`).

    Returns
    -------
    None
        This function does not return anything; its side effect is creating
        files in the `output_dir`.

    Raises
    ------
    RuntimeError
        If processing fails for any individual annotation when using
        multiprocessing. The original exception will be attached as the cause.
    """
    output_dir = Path(output_dir)

    if not output_dir.is_dir():
        logger.info(
            "Creating output directory: {output_dir}", output_dir=output_dir
        )
        output_dir.mkdir(parents=True)

    logger.info(
        "Starting preprocessing of {num_annotations} annotations with {max_workers} workers.",
        num_annotations=len(clip_annotations),
        max_workers=max_workers or "all available",
    )

    with Pool(max_workers) as pool:
        list(
            tqdm(
                pool.imap_unordered(
                    partial(
                        preprocess_single_annotation,
                        output_dir=output_dir,
                        filename_fn=filename_fn,
                        replace=replace,
                        audio_loader=audio_loader,
                        preprocessor=preprocessor,
                        labeller=labeller,
                    ),
                    clip_annotations,
                ),
                total=len(clip_annotations),
                desc="Preprocessing annotations",
            )
        )
    logger.info("Finished preprocessing.")


def preprocess_single_annotation(
    clip_annotation: data.ClipAnnotation,
    output_dir: data.PathLike,
    audio_loader: AudioLoader,
    preprocessor: PreprocessorProtocol,
    labeller: ClipLabeller,
    filename_fn: FilenameFn = _get_filename,
    replace: bool = False,
) -> None:
    """Process a single ClipAnnotation and save the result to a file.

    Internal function designed to be called by `preprocess_annotations`, often
    in parallel worker processes. It generates the training example using
    `generate_train_example` and saves it using `save_to_file`. Handles
    file existence checks based on the `replace` flag.

    Parameters
    ----------
    clip_annotation : data.ClipAnnotation
        The single annotation to process.
    output_dir : Path
        The directory where the output NetCDF file should be saved.
    preprocessor : PreprocessorProtocol
        Initialized preprocessor object.
    labeller : ClipLabeller
        Initialized labeller function.
    filename_fn : FilenameFn, default=_get_filename
        Function to determine the output filename.
    replace : bool, default=False
        Whether to overwrite existing output files.
    """
    output_dir = Path(output_dir)

    filename = filename_fn(clip_annotation)
    path = output_dir / filename

    if path.is_file() and not replace:
        logger.debug("Skipping existing file: {path}", path=path)
        return

    if path.is_file() and replace:
        logger.debug("Removing existing file: {path}", path=path)
        path.unlink()

    logger.debug("Processing annotation {uuid}", uuid=clip_annotation.uuid)

    try:
        sample = generate_train_example(
            clip_annotation,
            audio_loader=audio_loader,
            preprocessor=preprocessor,
            labeller=labeller,
        )
    except Exception as error:
        logger.error(
            "Failed to process annotation {uuid} to {path}. Error: {error}",
            uuid=clip_annotation.uuid,
            path=path,
            error=error,
        )
        return

    _save_xr_dataset_to_file(sample, path)
