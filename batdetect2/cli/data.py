from pathlib import Path
from typing import Optional

import click

from batdetect2.cli.base import cli
from batdetect2.data import load_dataset_from_config

__all__ = ["data"]


@cli.group()
def data(): ...


@data.command()
@click.argument(
    "dataset_config",
    type=click.Path(exists=True),
)
@click.option(
    "--field",
    type=str,
    help="If the dataset info is in a nested field please specify here.",
)
@click.option(
    "--base-dir",
    type=click.Path(exists=True),
    help="The base directory to which all recording and annotations paths are relative to.",
)
def summary(
    dataset_config: Path,
    field: Optional[str] = None,
    base_dir: Optional[Path] = None,
):
    base_dir = base_dir or Path.cwd()
    dataset = load_dataset_from_config(
        dataset_config, field=field, base_dir=base_dir
    )
    print(f"Number of annotated clips: {len(dataset.clip_annotations)}")
