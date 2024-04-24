"""BatDetect2 command line interface."""

import os

import click


__all__ = [
    "cli",
]


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


INFO_STR = """
BatDetect2 - Detection and Classification
    Assumes audio files are mono, not stereo.
    Spaces in the input paths will throw an error. Wrap in quotes.
    Input files should be short in duration e.g. < 30 seconds.
"""


@click.group()
def cli():
    """BatDetect2 - Bat Call Detection and Classification."""
    click.echo(INFO_STR)
