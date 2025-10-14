"""BatDetect2 command line interface."""

import sys

import click
from loguru import logger

# from batdetect2.cli.ascii import BATDETECT_ASCII_ART

__all__ = [
    "cli",
]


INFO_STR = """
BatDetect2 - Detection and Classification
    Assumes audio files are mono, not stereo.
    Spaces in the input paths will throw an error. Wrap in quotes.
    Input files should be short in duration e.g. < 30 seconds.
"""


@click.group()
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Increase verbosity. -v for INFO, -vv for DEBUG.",
)
def cli(
    verbose: int = 0,
):
    """BatDetect2 - Bat Call Detection and Classification."""
    click.echo(INFO_STR)

    logger.remove()

    if verbose == 0:
        log_level = "WARNING"
    elif verbose == 1:
        log_level = "INFO"
    else:
        log_level = "DEBUG"

    logger.add(sys.stderr, level=log_level)

    logger.enable("batdetect2")
    # click.echo(BATDETECT_ASCII_ART)
