from batdetect2.cli.base import cli
from batdetect2.cli.compat import detect
from batdetect2.cli.data import data
from batdetect2.cli.preprocess import preprocess
from batdetect2.cli.train import train_detector

__all__ = [
    "cli",
    "detect",
    "data",
    "train_detector",
    "preprocess",
]


if __name__ == "__main__":
    cli()
