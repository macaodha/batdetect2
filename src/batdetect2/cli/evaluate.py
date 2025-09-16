import sys
from pathlib import Path
from typing import Optional

import click
from loguru import logger

from batdetect2.cli.base import cli

__all__ = ["evaluate_command"]


@cli.command(name="evaluate")
@click.argument("model-path", type=click.Path(exists=True))
@click.argument("test_dataset", type=click.Path(exists=True))
@click.option("--output-dir", type=click.Path())
@click.option("--workers", type=int)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Increase verbosity. -v for INFO, -vv for DEBUG.",
)
def evaluate_command(
    model_path: Path,
    test_dataset: Path,
    output_dir: Optional[Path] = None,
    workers: Optional[int] = None,
    verbose: int = 0,
):
    from batdetect2.data import load_dataset_from_config
    from batdetect2.evaluate.evaluate import evaluate
    from batdetect2.train.lightning import load_model_from_checkpoint

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

    model, train_config = load_model_from_checkpoint(model_path)

    df, results = evaluate(
        model,
        test_annotations,
        config=train_config,
        num_workers=workers,
    )

    print(results)

    if output_dir:
        df.to_csv(output_dir / "results.csv")
