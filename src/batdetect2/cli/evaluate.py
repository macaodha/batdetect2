from pathlib import Path

import click
from loguru import logger

from batdetect2.cli.base import cli

__all__ = ["evaluate_command"]


DEFAULT_OUTPUT_DIR = Path("outputs") / "evaluation"


@cli.command(name="evaluate")
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("test_dataset", type=click.Path(exists=True))
@click.option("--targets", "targets_config", type=click.Path(exists=True))
@click.option("--audio-config", type=click.Path(exists=True))
@click.option("--evaluation-config", type=click.Path(exists=True))
@click.option("--inference-config", type=click.Path(exists=True))
@click.option("--outputs-config", type=click.Path(exists=True))
@click.option("--logging-config", type=click.Path(exists=True))
@click.option("--base-dir", type=click.Path(), default=Path.cwd())
@click.option("--output-dir", type=click.Path(), default=DEFAULT_OUTPUT_DIR)
@click.option("--experiment-name", type=str)
@click.option("--run-name", type=str)
@click.option("--workers", "num_workers", type=int)
def evaluate_command(
    model_path: Path,
    test_dataset: Path,
    base_dir: Path,
    targets_config: Path | None,
    audio_config: Path | None,
    evaluation_config: Path | None,
    inference_config: Path | None,
    outputs_config: Path | None,
    logging_config: Path | None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    num_workers: int = 0,
    experiment_name: str | None = None,
    run_name: str | None = None,
):
    from batdetect2.api_v2 import BatDetect2API
    from batdetect2.audio import AudioConfig
    from batdetect2.data import load_dataset_from_config
    from batdetect2.evaluate import EvaluationConfig
    from batdetect2.inference import InferenceConfig
    from batdetect2.logging import AppLoggingConfig
    from batdetect2.outputs import OutputsConfig
    from batdetect2.targets import TargetConfig

    logger.info("Initiating evaluation process...")

    test_annotations = load_dataset_from_config(
        test_dataset,
        base_dir=base_dir,
    )

    logger.debug(
        "Loaded {num_annotations} test examples",
        num_annotations=len(test_annotations),
    )

    target_conf = (
        TargetConfig.load(targets_config)
        if targets_config is not None
        else None
    )
    audio_conf = (
        AudioConfig.load(audio_config) if audio_config is not None else None
    )
    eval_conf = (
        EvaluationConfig.load(evaluation_config)
        if evaluation_config is not None
        else None
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
        targets_config=target_conf,
        audio_config=audio_conf,
        evaluation_config=eval_conf,
        inference_config=inference_conf,
        outputs_config=outputs_conf,
        logging_config=logging_conf,
    )

    api.evaluate(
        test_annotations,
        num_workers=num_workers,
        output_dir=output_dir,
        experiment_name=experiment_name,
        run_name=run_name,
    )
