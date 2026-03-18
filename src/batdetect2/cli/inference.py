from pathlib import Path

import click
from loguru import logger
from soundevent import io
from soundevent.audio.files import get_audio_files

from batdetect2.cli.base import cli

__all__ = ["predict"]


@cli.group(name="predict")
def predict() -> None:
    """Run prediction with BatDetect2 API v2."""


def _build_api(
    model_path: Path,
    audio_config: Path | None,
    inference_config: Path | None,
    outputs_config: Path | None,
    logging_config: Path | None,
):
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
        model_path,
        audio_config=audio_conf,
        inference_config=inference_conf,
        outputs_config=outputs_conf,
        logging_config=logging_conf,
    )
    return api, audio_conf, inference_conf, outputs_conf


def _run_inference(
    model_path: Path,
    audio_files: list[Path],
    output_path: Path,
    audio_config: Path | None,
    inference_config: Path | None,
    outputs_config: Path | None,
    logging_config: Path | None,
    batch_size: int | None,
    num_workers: int,
    format_name: str | None,
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
    )

    common_path = audio_files[0].parent if audio_files else None
    api.save_predictions(
        predictions,
        path=output_path,
        audio_dir=common_path,
        format=format_name,
    )

    logger.info(
        "Inference complete. Results saved to {path}", path=output_path
    )


@predict.command(name="directory")
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("audio_dir", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option("--audio-config", type=click.Path(exists=True))
@click.option("--inference-config", type=click.Path(exists=True))
@click.option("--outputs-config", type=click.Path(exists=True))
@click.option("--logging-config", type=click.Path(exists=True))
@click.option("--batch-size", type=int)
@click.option("--workers", "num_workers", type=int, default=0)
@click.option("--format", "format_name", type=str)
def inference_directory_command(
    model_path: Path,
    audio_dir: Path,
    output_path: Path,
    audio_config: Path | None,
    inference_config: Path | None,
    outputs_config: Path | None,
    logging_config: Path | None,
    batch_size: int | None,
    num_workers: int,
    format_name: str | None,
) -> None:
    audio_files = list(get_audio_files(audio_dir))
    _run_inference(
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
    )


@predict.command(name="file_list")
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("file_list", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option("--audio-config", type=click.Path(exists=True))
@click.option("--inference-config", type=click.Path(exists=True))
@click.option("--outputs-config", type=click.Path(exists=True))
@click.option("--logging-config", type=click.Path(exists=True))
@click.option("--batch-size", type=int)
@click.option("--workers", "num_workers", type=int, default=0)
@click.option("--format", "format_name", type=str)
def inference_file_list_command(
    model_path: Path,
    file_list: Path,
    output_path: Path,
    audio_config: Path | None,
    inference_config: Path | None,
    outputs_config: Path | None,
    logging_config: Path | None,
    batch_size: int | None,
    num_workers: int,
    format_name: str | None,
) -> None:
    file_list = Path(file_list)
    audio_files = [
        Path(line.strip())
        for line in file_list.read_text().splitlines()
        if line.strip()
    ]

    _run_inference(
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
    )


@predict.command(name="dataset")
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("dataset_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option("--audio-config", type=click.Path(exists=True))
@click.option("--inference-config", type=click.Path(exists=True))
@click.option("--outputs-config", type=click.Path(exists=True))
@click.option("--logging-config", type=click.Path(exists=True))
@click.option("--batch-size", type=int)
@click.option("--workers", "num_workers", type=int, default=0)
@click.option("--format", "format_name", type=str)
def inference_dataset_command(
    model_path: Path,
    dataset_path: Path,
    output_path: Path,
    audio_config: Path | None,
    inference_config: Path | None,
    outputs_config: Path | None,
    logging_config: Path | None,
    batch_size: int | None,
    num_workers: int,
    format_name: str | None,
) -> None:
    dataset_path = Path(dataset_path)
    dataset = io.load(dataset_path, type="annotation_set")
    audio_files = sorted(
        {
            Path(clip_annotation.clip.recording.path)
            for clip_annotation in dataset.clip_annotations
        }
    )

    _run_inference(
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
    )
