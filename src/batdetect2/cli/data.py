from pathlib import Path
from typing import Optional

import click

from batdetect2.cli.base import cli

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
    from batdetect2.data import load_dataset_from_config

    base_dir = base_dir or Path.cwd()

    dataset = load_dataset_from_config(
        dataset_config,
        field=field,
        base_dir=base_dir,
    )

    print(f"Number of annotated clips: {len(dataset)}")


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
    "--output",
    type=click.Path(exists=False),
    default="annotations.json",
)
@click.option(
    "--base-dir",
    type=click.Path(exists=True),
    help="The base directory to which all recording and annotations paths are relative to.",
)
def convert(
    dataset_config: Path,
    field: Optional[str] = None,
    output: Path = Path("annotations.json"),
    base_dir: Optional[Path] = None,
):
    """Convert a dataset config file to soundevent format."""
    from soundevent import data, io

    from batdetect2.data import load_dataset, load_dataset_config

    base_dir = base_dir or Path.cwd()

    config = load_dataset_config(
        dataset_config,
        field=field,
    )

    dataset = load_dataset(
        config,
        base_dir=base_dir,
    )

    annotation_set = data.AnnotationSet(
        clip_annotations=list(dataset),
        name=config.name,
        description=config.description,
    )

    io.save(annotation_set, output)
