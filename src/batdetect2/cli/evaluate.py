import sys
from pathlib import Path
from typing import Optional

import click
from loguru import logger

from batdetect2.cli.base import cli

__all__ = ["evaluate_command"]


DEFAULT_OUTPUT_DIR = Path("outputs") / "evaluation"


@cli.command(name="evaluate")
@click.argument("model-path", type=click.Path(exists=True))
@click.argument("test_dataset", type=click.Path(exists=True))
@click.option("--config", "config_path", type=click.Path())
@click.option("--output-dir", type=click.Path(), default=DEFAULT_OUTPUT_DIR)
@click.option("--experiment-name", type=str)
@click.option("--run-name", type=str)
@click.option("--workers", "num_workers", type=int)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Increase verbosity. -v for INFO, -vv for DEBUG.",
)
def evaluate_command(
    model_path: Path,
    test_dataset: Path,
    config_path: Optional[Path],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    num_workers: Optional[int] = None,
    experiment_name: Optional[str] = None,
    run_name: Optional[str] = None,
    verbose: int = 0,
):
    from batdetect2.api.base import BatDetect2API
    from batdetect2.config import load_full_config
    from batdetect2.data import load_dataset_from_config

    logger.remove()
    if verbose == 0:
        log_level = "WARNING"
    elif verbose == 1:
        log_level = "INFO"
    else:
        log_level = "DEBUG"
    logger.add(sys.stderr, level=log_level)

    logger.info("Initiating evaluation process...")

    test_annotations = load_dataset_from_config(test_dataset)
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
