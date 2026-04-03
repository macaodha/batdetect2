from pathlib import Path

import click

from batdetect2.cli.base import cli

__all__ = ["data"]


@cli.group(short_help="Inspect and convert datasets.")
def data():
    """Inspect and convert dataset configuration files."""


@data.command(short_help="Print dataset summary information.")
@click.argument(
    "dataset_config",
    type=click.Path(exists=True),
)
@click.option(
    "--field",
    type=str,
    help=(
        "Nested field name that contains dataset configuration. "
        "Use this when the config is wrapped in a larger file."
    ),
)
@click.option(
    "--targets",
    "targets_path",
    type=click.Path(exists=True),
    help=(
        "Path to targets config file. If provided, a per-class summary "
        "table is printed."
    ),
)
@click.option(
    "--base-dir",
    type=click.Path(exists=True),
    help=(
        "Base directory used to resolve relative recording and annotation "
        "paths in the dataset config."
    ),
)
def summary(
    dataset_config: Path,
    field: str | None = None,
    targets_path: Path | None = None,
    base_dir: Path | None = None,
):
    """Show dataset size and optional class summary.

    Prints the number of annotated clips. If `--targets` is provided, it also
    prints a per-class summary table based on the configured targets.
    """
    from batdetect2.data import compute_class_summary, load_dataset_from_config
    from batdetect2.targets import load_targets

    base_dir = base_dir or Path.cwd()

    dataset = load_dataset_from_config(
        dataset_config,
        field=field,
        base_dir=base_dir,
    )

    print(f"Number of annotated clips: {len(dataset)}")

    if targets_path is None:
        return

    targets = load_targets(targets_path)

    summary = compute_class_summary(dataset, targets)

    print(summary.sort_values("class_name").to_markdown())


@data.command(short_help="Convert dataset config to annotation set.")
@click.argument(
    "dataset_config",
    type=click.Path(exists=True),
)
@click.option(
    "--field",
    type=str,
    help=(
        "Nested field name that contains dataset configuration. "
        "Use this when the config is wrapped in a larger file."
    ),
)
@click.option(
    "--output",
    type=click.Path(exists=False),
    default="annotations.json",
)
@click.option(
    "--base-dir",
    type=click.Path(exists=True),
    help=(
        "Base directory used to resolve relative recording and annotation "
        "paths in the dataset config."
    ),
)
@click.option(
    "--audio-dir",
    type=click.Path(exists=True),
    help=(
        "Directory containing audio files. Output annotation paths are "
        "made relative to this directory."
    ),
)
@click.option(
    "--add-source-tag",
    is_flag=True,
    help=(
        "Add a source tag to each clip annotation. This is useful for "
        "downstream tools that need to know which source the annotations "
        "came from."
    ),
)
@click.option(
    "--include-sources",
    type=str,
    multiple=True,
    help=(
        "Only include sources with the specified names. If provided, only "
        "sources with matching names will be included in the output."
    ),
)
@click.option(
    "--exclude-sources",
    type=str,
    multiple=True,
    help=(
        "Exclude sources with the specified names. If provided, sources with "
        "matching names will be excluded from the output."
    ),
)
@click.option(
    "--apply-transforms/--no-apply-transforms",
    default=True,
    help=(
        "Apply any configured sound event transforms to the annotations. "
        "Defaults to True."
    ),
)
@click.option(
    "--apply-filters/--no-apply-filters",
    default=True,
    help=(
        "Apply any configured sound event filters to the annotations. "
        "Defaults to True."
    ),
)
def convert(
    dataset_config: Path,
    field: str | None = None,
    output: Path = Path("annotations.json"),
    base_dir: Path | None = None,
    audio_dir: Path | None = None,
    add_source_tag: bool = True,
    include_sources: list[str] | None = None,
    exclude_sources: list[str] | None = None,
    apply_transforms: bool = True,
    apply_filters: bool = True,
):
    """Convert a dataset config into soundevent annotation-set format.

    Writes a single annotation-set file that can be used by downstream tools.
    Use `--audio-dir` to control relative audio path handling in the output.
    """
    from soundevent import data, io

    from batdetect2.data import load_dataset, load_dataset_config

    base_dir = base_dir or Path.cwd()

    config = load_dataset_config(dataset_config, field=field)

    dataset = load_dataset(
        config,
        base_dir=base_dir,
        add_source_tag=add_source_tag,
        include_sources=include_sources,
        exclude_sources=exclude_sources,
        apply_transforms=apply_transforms,
        apply_filters=apply_filters,
    )

    annotation_set = data.AnnotationSet(
        clip_annotations=list(dataset),
        name=config.name,
        description=config.description,
    )

    if audio_dir:
        audio_dir = Path(audio_dir)

        if not audio_dir.is_absolute():
            audio_dir = audio_dir.resolve()

        print(f"Using audio directory: {audio_dir}")

    io.save(annotation_set, output, audio_dir=audio_dir)
