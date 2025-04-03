from typing import Annotated, List

from pydantic import Field

from batdetect2.configs import BaseConfig
from batdetect2.data.annotations import AnnotationFormats


class Dataset(BaseConfig):
    """Represents a collection of one or more DatasetSources.

    In the context of batdetect2, a Dataset aggregates multiple `DatasetSource`
    instances. It serves as the primary unit for defining data splits,
    typically used for model training, validation, or testing phases.

    Attributes:
        name: A descriptive name for the overall dataset
            (e.g., "UK Training Set").
        description: A detailed explanation of the dataset's purpose,
            composition, how it was assembled, or any specific characteristics.
        sources: A list containing the `DatasetSource` objects included in this
            dataset.
    """

    name: str
    description: str
    sources: List[
        Annotated[AnnotationFormats, Field(..., discriminator="format")]
    ]
