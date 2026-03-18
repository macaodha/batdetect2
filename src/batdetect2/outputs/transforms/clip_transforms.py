from typing import Annotated, Literal

from pydantic import Field

from batdetect2.core.registries import (
    ImportConfig,
    Registry,
    add_import_config,
)
from batdetect2.outputs.types import ClipDetectionsTransform

__all__ = [
    "ClipDetectionsTransformConfig",
    "clip_transforms",
]


clip_transforms: Registry[ClipDetectionsTransform, []] = Registry(
    "clip_detection_transform"
)


@add_import_config(clip_transforms)
class ClipDetectionsTransformImportConfig(ImportConfig):
    name: Literal["import"] = "import"


ClipDetectionsTransformConfig = Annotated[
    ClipDetectionsTransformImportConfig,
    Field(discriminator="name"),
]
