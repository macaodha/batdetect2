from batdetect2.cli.base import cli
from batdetect2.cli.compat import detect
from batdetect2.cli.data import data
from batdetect2.cli.evaluate import evaluate_command
from batdetect2.cli.inference import predict
from batdetect2.cli.train import train_command

__all__ = [
    "cli",
    "detect",
    "data",
    "train_command",
    "evaluate_command",
    "predict",
]


if __name__ == "__main__":
    cli()
