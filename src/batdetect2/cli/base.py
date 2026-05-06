"""BatDetect2 command line interface."""

import click

__all__ = [
    "cli",
]


INFO_STR = """
BatDetect2
    Input audio should be mono.
    Wrap paths that contain spaces in quotes.
    For long recordings, split audio into shorter files before running
    prediction.
"""


@click.group()
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Increase verbosity. -v for INFO, -vv for DEBUG.",
)
def cli(verbose: int = 0):
    """Run the BatDetect2 CLI.

    Use subcommands to run prediction, training, evaluation, and dataset
    utilities.
    """
    click.echo(INFO_STR)

    from batdetect2.logging import enable_logging

    enable_logging(verbose)
    # click.echo(BATDETECT_ASCII_ART)
