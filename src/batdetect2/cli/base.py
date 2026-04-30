"""BatDetect2 command line interface."""

import click

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
def cli(verbose: int = 0):
    """Run the BatDetect2 CLI.

    This command initializes logging and exposes subcommands for prediction,
    training, evaluation, and dataset utilities.
    """
    click.echo(INFO_STR)

    from batdetect2.logging import enable_logging

    enable_logging(verbose)
    # click.echo(BATDETECT_ASCII_ART)
