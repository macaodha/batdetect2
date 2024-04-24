"""Module for preprocessing data for training."""

import os
import warnings
from functools import partial
from pathlib import Path
from typing import Callable, Optional, Sequence, Union
from tqdm.auto import tqdm
from multiprocessing import Pool

import xarray as xr
from pydantic import BaseModel, Field
from soundevent import data

from batdetect2.data.labels import TARGET_SIGMA, LabelFn, generate_heatmaps
from batdetect2.data.preprocessing import (
    DENOISE_SPEC_AVG,
    FFT_OVERLAP,
    FFT_WIN_LENGTH_S,
    MAX_FREQ_HZ,
    MAX_SCALE_SPEC,
    MIN_FREQ_HZ,
    SCALE_RAW_AUDIO,
    SPEC_SCALE,
    TARGET_SAMPLERATE_HZ,
    preprocess_audio_clip,
)

PathLike = Union[Path, str, os.PathLike]
FilenameFn = Callable[[data.ClipAnnotation], str]

__all__ = [
    "preprocess_annotations",
]


class PreprocessingConfig(BaseModel):
    """Configuration for preprocessing data."""

    target_samplerate: int = Field(default=TARGET_SAMPLERATE_HZ, gt=0)

    scale_audio: bool = Field(default=SCALE_RAW_AUDIO)

    fft_win_length: float = Field(default=FFT_WIN_LENGTH_S, gt=0)

    fft_overlap: float = Field(default=FFT_OVERLAP, ge=0, lt=1)

    max_freq: int = Field(default=MAX_FREQ_HZ, gt=0)

    min_freq: int = Field(default=MIN_FREQ_HZ, gt=0)

    spec_scale: str = Field(default=SPEC_SCALE)

    denoise_spec_avg: bool = DENOISE_SPEC_AVG

    max_scale_spec: bool = MAX_SCALE_SPEC

    target_sigma: float = Field(default=TARGET_SIGMA, gt=0)

    class_labels: Sequence[str] = ["bat"]


def generate_train_example(
    clip_annotation: data.ClipAnnotation,
    label_fn: LabelFn = lambda _: None,
    config: Optional[PreprocessingConfig] = None,
) -> xr.Dataset:
    """Generate a training example."""
    if config is None:
        config = PreprocessingConfig()

    spectrogram = preprocess_audio_clip(
        clip_annotation.clip,
        target_sampling_rate=config.target_samplerate,
        scale_audio=config.scale_audio,
        fft_win_length=config.fft_win_length,
        fft_overlap=config.fft_overlap,
        max_freq=config.max_freq,
        min_freq=config.min_freq,
        spec_scale=config.spec_scale,
        denoise_spec_avg=config.denoise_spec_avg,
        max_scale_spec=config.max_scale_spec,
    )

    detection_heatmap, class_heatmap, size_heatmap = generate_heatmaps(
        clip_annotation,
        spectrogram,
        target_sigma=config.target_sigma,
        num_classes=len(config.class_labels),
        class_labels=list(config.class_labels),
        label_fn=label_fn,
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
        configuration=config.model_dump_json(),
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
    filename_fn: FilenameFn = _get_filename,
    replace: bool = False,
    label_fn: LabelFn = lambda _: None,
) -> None:
    output_dir = Path(output_dir)

    filename = filename_fn(clip_annotation)
    path = output_dir / filename

    if path.is_file() and not replace:
        return

    sample = generate_train_example(
        clip_annotation,
        label_fn=label_fn,
        config=config,
    )

    save_to_file(sample, path)


def preprocess_annotations(
    clip_annotations: Sequence[data.ClipAnnotation],
    output_dir: PathLike,
    filename_fn: FilenameFn = _get_filename,
    replace: bool = False,
    config_file: Optional[PathLike] = None,
    label_fn: LabelFn = lambda _: None,
    max_workers: Optional[int] = None,
    **kwargs,
) -> None:
    """Preprocess annotations and save to disk."""
    output_dir = Path(output_dir)

    if not output_dir.is_dir():
        output_dir.mkdir(parents=True)

    if config_file is not None:
        config = load_config(config_file, **kwargs)
    else:
        config = PreprocessingConfig(**kwargs)

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
                        label_fn=label_fn,
                    ),
                    clip_annotations,
                ),
                total=len(clip_annotations),
            )
        )
