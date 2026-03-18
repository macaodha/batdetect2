from pathlib import Path

import click
from loguru import logger

from batdetect2.cli.base import cli

__all__ = ["evaluate_command"]


DEFAULT_OUTPUT_DIR = Path("outputs") / "evaluation"


@cli.command(name="evaluate")
@click.argument("model-path", type=click.Path(exists=True))
@click.argument("test_dataset", type=click.Path(exists=True))
@click.option("--config", "config_path", type=click.Path())
@click.option("--base-dir", type=click.Path(), default=Path.cwd())
@click.option("--output-dir", type=click.Path(), default=DEFAULT_OUTPUT_DIR)
@click.option("--experiment-name", type=str)
@click.option("--run-name", type=str)
@click.option("--workers", "num_workers", type=int)
def evaluate_command(
    model_path: Path,
    test_dataset: Path,
    base_dir: Path,
    config_path: Path | None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    num_workers: int = 0,
    experiment_name: str | None = None,
    run_name: str | None = None,
):
    from batdetect2.api_v2 import BatDetect2API
    from batdetect2.config import load_full_config
    from batdetect2.data import load_dataset_from_config

    logger.info("Initiating evaluation process...")

    test_annotations = load_dataset_from_config(
        test_dataset,
        base_dir=base_dir,
    )

    logger.debug(
        "Loaded {num_annotations} test examples",
        num_annotations=len(test_annotations),
    )

    config = None
    if config_path is not None:
        config = load_full_config(config_path)

    api = BatDetect2API.from_checkpoint(model_path, config=config)

    api.evaluate(
        test_annotations,
        num_workers=num_workers,
        output_dir=output_dir,
        experiment_name=experiment_name,
        run_name=run_name,
    )
