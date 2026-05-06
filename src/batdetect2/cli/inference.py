from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING

import click
from loguru import logger

from batdetect2.cli.base import cli

if TYPE_CHECKING:
    from batdetect2.api_v2 import BatDetect2API
    from batdetect2.audio import AudioConfig
    from batdetect2.inference import InferenceConfig
    from batdetect2.outputs import OutputsConfig

__all__ = ["process"]


@cli.group(name="process", short_help="Run processing workflows.")
def process() -> None:
    """Run model inference on audio.

    Choose a subcommand based on how you want to provide input audio.
    """


def common_predict_options(func):
    """Attach options shared by all ``process`` subcommands."""

    @click.option(
        "--model",
        "model_path",
        type=str,
        help=(
            "Path to a checkpoint, checkpoint alias, or a Hugging Face "
            "URI to fine-tune from. Defaults to uk_same"
        ),
    )
    @click.option(
        "--audio-config",
        type=click.Path(exists=True),
        help=(
            "Path to an audio config file. Use this to override audio "
            "loading settings."
        ),
    )
    @click.option(
        "--inference-config",
        type=click.Path(exists=True),
        help=(
            "Path to an inference config file. Use this to override "
            "prediction settings."
        ),
    )
    @click.option(
        "--outputs-config",
        type=click.Path(exists=True),
        help=(
            "Path to an outputs config file. Use this to control the "
            "saved output format and fields."
        ),
    )
    @click.option(
        "--logging-config",
        type=click.Path(exists=True),
        help=("Path to a logging config file. Use this to change log output."),
    )
    @click.option(
        "--batch-size",
        type=int,
        help=(
            "Batch size for inference. If omitted, the config value is used."
        ),
    )
    @click.option(
        "--workers",
        "num_workers",
        type=int,
        default=0,
        show_default=True,
        help="Number of worker processes for audio loading.",
    )
    @click.option(
        "--format",
        "format_name",
        type=str,
        help=(
            "Output format name used by the prediction writer. If omitted, "
            "the config default is used."
        ),
    )
    @click.option(
        "--detection-threshold",
        type=click.FloatRange(min=0.0, max=1.0),
        default=None,
        help=(
            "Optional detection score threshold override. If omitted, "
            "the configured threshold is used."
        ),
    )
    @wraps(func)
    def wrapped(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapped


def _build_api(
    model_path: str | None,
    audio_config: Path | None,
    inference_config: Path | None,
    outputs_config: Path | None,
    logging_config: Path | None,
) -> "tuple[BatDetect2API, AudioConfig | None, InferenceConfig | None, OutputsConfig | None]":
    from batdetect2.api_v2 import BatDetect2API
    from batdetect2.audio import AudioConfig
    from batdetect2.inference import InferenceConfig
    from batdetect2.logging import AppLoggingConfig
    from batdetect2.outputs import OutputsConfig

    audio_conf = (
        AudioConfig.load(audio_config) if audio_config is not None else None
    )
    inference_conf = (
        InferenceConfig.load(inference_config)
        if inference_config is not None
        else None
    )
    outputs_conf = (
        OutputsConfig.load(outputs_config)
        if outputs_config is not None
        else None
    )
    logging_conf = (
        AppLoggingConfig.load(logging_config)
        if logging_config is not None
        else None
    )

    api = BatDetect2API.from_checkpoint(
        path=model_path,
        audio_config=audio_conf,
        inference_config=inference_conf,
        outputs_config=outputs_conf,
        logging_config=logging_conf,
    )
    return api, audio_conf, inference_conf, outputs_conf


def _run_prediction(
    model_path: str | None,
    audio_files: list[Path],
    output_path: Path,
    audio_config: Path | None,
    inference_config: Path | None,
    outputs_config: Path | None,
    logging_config: Path | None,
    batch_size: int | None,
    num_workers: int,
    format_name: str | None,
    detection_threshold: float | None,
) -> None:
    logger.info("Initiating prediction process...")

    api, audio_conf, inference_conf, outputs_conf = _build_api(
        model_path,
        audio_config,
        inference_config,
        outputs_config,
        logging_config,
    )

    logger.info("Found {num_files} audio files", num_files=len(audio_files))

    predictions = api.process_files(
        audio_files,
        batch_size=batch_size,
        num_workers=num_workers,
        audio_config=audio_conf,
        inference_config=inference_conf,
        output_config=outputs_conf,
        detection_threshold=detection_threshold,
    )

    common_path = audio_files[0].parent if audio_files else None
    api.save_predictions(
        predictions,
        path=output_path,
        audio_dir=common_path,
        format=format_name,
    )

    logger.info(
        "Prediction complete. Results saved to {path}", path=output_path
    )


@process.command(
    name="directory",
    short_help="Process audio files in a directory.",
)
@click.argument("audio_dir", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@common_predict_options
def predict_directory_command(
    model_path: str | None,
    audio_dir: Path,
    output_path: Path,
    audio_config: Path | None,
    inference_config: Path | None,
    outputs_config: Path | None,
    logging_config: Path | None,
    batch_size: int | None,
    num_workers: int,
    format_name: str | None,
    detection_threshold: float | None,
) -> None:
    """Run processing on all supported audio files in a directory.

    This command scans ``audio_dir`` for audio files, runs processing, and
    saves the results to ``output_path``.
    """
    from soundevent.audio.files import get_audio_files

    audio_files = list(get_audio_files(audio_dir))
    _run_prediction(
        model_path=model_path,
        audio_files=audio_files,
        output_path=output_path,
        audio_config=audio_config,
        inference_config=inference_config,
        outputs_config=outputs_config,
        logging_config=logging_config,
        batch_size=batch_size,
        num_workers=num_workers,
        format_name=format_name,
        detection_threshold=detection_threshold,
    )


@process.command(
    name="file_list",
    short_help="Process paths listed in a text file.",
)
@click.argument("file_list", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@common_predict_options
def predict_file_list_command(
    file_list: Path,
    output_path: Path,
    model_path: str | None,
    audio_config: Path | None,
    inference_config: Path | None,
    outputs_config: Path | None,
    logging_config: Path | None,
    batch_size: int | None,
    num_workers: int,
    format_name: str | None,
    detection_threshold: float | None,
) -> None:
    """Run processing on audio files listed in a text file.

    The text file should contain one audio path per line. Empty lines are
    ignored.
    """
    file_list = Path(file_list)
    audio_files = [
        Path(line.strip())
        for line in file_list.read_text().splitlines()
        if line.strip()
    ]

    _run_prediction(
        model_path=model_path,
        audio_files=audio_files,
        output_path=output_path,
        audio_config=audio_config,
        inference_config=inference_config,
        outputs_config=outputs_config,
        logging_config=logging_config,
        batch_size=batch_size,
        num_workers=num_workers,
        format_name=format_name,
        detection_threshold=detection_threshold,
    )


@process.command(
    name="dataset",
    short_help="Process recordings from a dataset config.",
)
@click.argument("dataset_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@common_predict_options
def predict_dataset_command(
    dataset_path: Path,
    output_path: Path,
    model_path: str | None,
    audio_config: Path | None,
    inference_config: Path | None,
    outputs_config: Path | None,
    logging_config: Path | None,
    batch_size: int | None,
    num_workers: int,
    format_name: str | None,
    detection_threshold: float | None,
) -> None:
    """Run processing on recordings referenced in a dataset file.

    Recording paths are read from the dataset and each recording is processed
    once.
    """
    from soundevent import io

    dataset_path = Path(dataset_path)
    dataset = io.load(dataset_path, type="annotation_set")
    audio_files = sorted(
        {
            Path(clip_annotation.clip.recording.path)
            for clip_annotation in dataset.clip_annotations
        }
    )

    _run_prediction(
        model_path=model_path,
        audio_files=audio_files,
        output_path=output_path,
        audio_config=audio_config,
        inference_config=inference_config,
        outputs_config=outputs_config,
        logging_config=logging_config,
        batch_size=batch_size,
        num_workers=num_workers,
        format_name=format_name,
        detection_threshold=detection_threshold,
    )
