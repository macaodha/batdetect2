from pathlib import Path

import click
from loguru import logger

from batdetect2.cli.base import cli

__all__ = ["evaluate_command"]


DEFAULT_OUTPUT_DIR = Path("outputs") / "evaluation"


@cli.command(name="evaluate", short_help="Evaluate a model checkpoint.")
@click.argument("test_dataset", type=click.Path(exists=True))
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
    help="Path to an audio config file.",
)
@click.option(
    "--evaluation-config",
    type=click.Path(exists=True),
    help="Path to an evaluation config file.",
)
@click.option(
    "--inference-config",
    type=click.Path(exists=True),
    help="Path to an inference config file.",
)
@click.option(
    "--outputs-config",
    type=click.Path(exists=True),
    help="Path to an outputs config file.",
)
@click.option(
    "--logging-config",
    type=click.Path(exists=True),
    help="Path to a logging config file.",
)
@click.option(
    "--base-dir",
    type=click.Path(),
    default=Path.cwd(),
    show_default=True,
    help=(
        "Base directory used to resolve relative paths in the dataset "
        "configuration."
    ),
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default=DEFAULT_OUTPUT_DIR,
    show_default=True,
    help="Directory where evaluation outputs are written.",
)
@click.option(
    "--experiment-name",
    type=str,
    help="Experiment name used for logging backends.",
)
@click.option(
    "--run-name",
    type=str,
    help="Run name used for logging backends.",
)
@click.option(
    "--workers",
    "num_workers",
    type=int,
    help="Number of worker processes for dataset loading.",
    default=0,
)
def evaluate_command(
    test_dataset: Path,
    model_path: str | None = None,
    base_dir: Path | None = None,
    audio_config: Path | None = None,
    evaluation_config: Path | None = None,
    inference_config: Path | None = None,
    outputs_config: Path | None = None,
    logging_config: Path | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    num_workers: int = 0,
    experiment_name: str | None = None,
    run_name: str | None = None,
):
    """Evaluate a checkpoint on a labelled test dataset.

    This command loads a checkpoint, runs evaluation on ``test_dataset``, and
    writes metrics to ``output_dir``.
    """
    from batdetect2.api_v2 import BatDetect2API
    from batdetect2.audio import AudioConfig
    from batdetect2.data import load_dataset_from_config
    from batdetect2.evaluate import EvaluationConfig
    from batdetect2.inference import InferenceConfig
    from batdetect2.logging import AppLoggingConfig
    from batdetect2.outputs import OutputsConfig

    logger.info("Initiating evaluation process...")

    test_annotations = load_dataset_from_config(
        test_dataset,
        base_dir=base_dir,
    )

    logger.debug(
        "Loaded {num_annotations} test examples",
        num_annotations=len(test_annotations),
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
