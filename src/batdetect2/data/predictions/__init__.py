from typing import Annotated, Optional, Union

from pydantic import Field

from batdetect2.data.predictions.base import (
    OutputFormatterProtocol,
    prediction_formatters,
)
from batdetect2.data.predictions.batdetect2 import BatDetect2OutputConfig
from batdetect2.data.predictions.raw import RawOutputConfig
from batdetect2.data.predictions.soundevent import SoundEventOutputConfig
from batdetect2.typing import TargetProtocol

__all__ = [
    "build_output_formatter",
    "get_output_formatter",
    "BatDetect2OutputConfig",
    "RawOutputConfig",
    "SoundEventOutputConfig",
]


OutputFormatConfig = Annotated[
    Union[BatDetect2OutputConfig, SoundEventOutputConfig, RawOutputConfig],
    Field(discriminator="name"),
]


def build_output_formatter(
    targets: Optional[TargetProtocol] = None,
    config: Optional[OutputFormatConfig] = None,
) -> OutputFormatterProtocol:
    """Construct the final output formatter."""
    from batdetect2.targets import build_targets

    config = config or RawOutputConfig()

    targets = targets or build_targets()
    return prediction_formatters.build(config, targets)


def get_output_formatter(
    name: str,
    targets: Optional[TargetProtocol] = None,
    config: Optional[OutputFormatConfig] = None,
) -> OutputFormatterProtocol:
    """Get the output formatter by name."""

    if config is None:
        config_class = prediction_formatters.get_config_type(name)
        config = config_class()  # type: ignore

    if config.name != name:  # type: ignore
        raise ValueError(
            f"Config name {config.name} does not match formatter name {name}"  # type: ignore
        )

    return build_output_formatter(targets, config)
