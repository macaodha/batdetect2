from typing import Literal

from batdetect2.core import ImportConfig, Registry, add_import_config
from batdetect2.data.annotations.types import AnnotationLoader

__all__ = [
    "AnnotationFormatImportConfig",
    "annotation_format_registry",
]

annotation_format_registry: Registry[AnnotationLoader, []] = Registry(
    "annotation_format",
    discriminator="format",
)


@add_import_config(annotation_format_registry)
class AnnotationFormatImportConfig(ImportConfig):
    """Import escape hatch for the annotation format registry.

    Use this config to dynamically instantiate any callable as an
    annotation loader without registering it in
    ``annotation_format_registry`` ahead of time.

    Parameters
    ----------
    format : Literal["import"]
        Discriminator value; must always be ``"import"``.
    target : str
        Fully-qualified dotted path to the callable to instantiate.
    arguments : dict[str, Any]
        Keyword arguments forwarded to the callable.
    """

    format: Literal["import"] = "import"
