from typing import Annotated

from pydantic import Field
from soundevent.data import PathLike

from batdetect2.outputs.formats.base import (
    OutputFormatterProtocol,
    output_formatters,
)
from batdetect2.outputs.formats.batdetect2 import BatDetect2OutputConfig
from batdetect2.outputs.formats.parquet import ParquetOutputConfig
from batdetect2.outputs.formats.raw import RawOutputConfig
from batdetect2.outputs.formats.soundevent import SoundEventOutputConfig
from batdetect2.typing import TargetProtocol

__all__ = [
    "BatDetect2OutputConfig",
    "OutputFormatConfig",
    "ParquetOutputConfig",
    "RawOutputConfig",
    "SoundEventOutputConfig",
    "build_output_formatter",
    "get_output_formatter",
    "load_predictions",
]


OutputFormatConfig = Annotated[
    BatDetect2OutputConfig
    | ParquetOutputConfig
    | SoundEventOutputConfig
    | RawOutputConfig,
    Field(discriminator="name"),
]


def build_output_formatter(
    targets: TargetProtocol | None = None,
    config: OutputFormatConfig | None = None,
) -> OutputFormatterProtocol:
    """Construct the final output formatter."""
    from batdetect2.targets import build_targets

    config = config or RawOutputConfig()

    targets = targets or build_targets()
    return output_formatters.build(config, targets)


def get_output_formatter(
    name: str | None = None,
    targets: TargetProtocol | None = None,
    config: OutputFormatConfig | None = None,
) -> OutputFormatterProtocol:
    """Get the output formatter by name."""

    if config is None:
        if name is None:
            raise ValueError("Either config or name must be provided.")

        config_class = output_formatters.get_config_type(name)
        config = config_class()  # type: ignore

    if config.name != name:  # type: ignore
        raise ValueError(
            f"Config name {config.name} does not match formatter name {name}"  # type: ignore
        )

    return build_output_formatter(targets, config)


def load_predictions(
    path: PathLike,
    format: str | None = "raw",
    config: OutputFormatConfig | None = None,
    targets: TargetProtocol | None = None,
):
    """Load predictions from a file."""
    from batdetect2.targets import build_targets

    targets = targets or build_targets()
    formatter = get_output_formatter(format, targets, config)
    return formatter.load(path)
