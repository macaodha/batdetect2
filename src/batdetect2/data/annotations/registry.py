from batdetect2.core import Registry
from batdetect2.data.annotations.types import AnnotationLoader

__all__ = [
    "annotation_format_registry",
]

annotation_format_registry: Registry[AnnotationLoader, []] = Registry(
    "annotation_format",
    discriminator="format",
)
