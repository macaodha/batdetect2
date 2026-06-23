"""BatDetect2 command line interface."""

from pathlib import Path

import click

from batdetect2.cli.ascii import BATDETECT_ASCII_ART

__all__ = [
    "cli",
]


INFO_STR = """
BatDetect2
    Wrap paths that contain spaces in quotes.
"""


@click.group(invoke_without_command=True)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Increase verbosity. -v for INFO, -vv for DEBUG.",
)
@click.option(
    "--log-file",
    type=click.Path(path_type=Path),
    help="Write CLI logs to a file in addition to the terminal.",
)
@click.pass_context
def cli(
    ctx: click.Context,
    verbose: int = 0,
    log_file: Path | None = None,
) -> None:
    """Run the BatDetect2 CLI.

    Use subcommands to run processing, training, evaluation, and dataset
    utilities.
    """

    if ctx.invoked_subcommand is None:
        click.echo(BATDETECT_ASCII_ART)
        click.echo(ctx.get_help())
        ctx.exit()

    from batdetect2.logging import enable_logging

    enable_logging(verbose, log_file=log_file)
