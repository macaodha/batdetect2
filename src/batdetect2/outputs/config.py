from pydantic import Field

from batdetect2.core.configs import BaseConfig
from batdetect2.outputs.formats import OutputFormatConfig
from batdetect2.outputs.formats.raw import RawOutputConfig
from batdetect2.outputs.transforms import OutputTransformConfig

__all__ = ["OutputsConfig"]


class OutputsConfig(BaseConfig):
    format: OutputFormatConfig = Field(default_factory=RawOutputConfig)
    transform: OutputTransformConfig = Field(
        default_factory=OutputTransformConfig
    )
