"""Manages the vocabulary for defining training targets.

This module provides the necessary tools to declare, register, and manage the
set of `soundevent.data.Term` objects used throughout the `batdetect2.targets`
sub-package. It establishes a consistent vocabulary for filtering,
transforming, and classifying sound events based on their annotations (Tags).

Terms can be pre-defined, loaded from the `soundevent.terms` library or defined
programmatically.
"""

from pydantic import BaseModel
from soundevent import data, terms

__all__ = [
    "call_type",
    "individual",
    "data_source",
    "get_tag_from_info",
    "TagInfo",
]

# The default key used to reference the 'generic_class' term.
# Often used implicitly when defining classification targets.
GENERIC_CLASS_KEY = "class"


data_source = data.Term(
    name="soundevent:data_source",
    label="Data Source",
    definition=(
        "A unique identifier for the source of the data, typically "
        "representing the project, site, or deployment context."
    ),
)

call_type = data.Term(
    name="soundevent:call_type",
    label="Call Type",
    definition=(
        "A broad categorization of animal vocalizations based on their "
        "intended function or purpose (e.g., social, distress, mating, "
        "territorial, echolocation)."
    ),
)
"""Term representing the broad functional category of a vocalization."""

individual = data.Term(
    name="soundevent:individual",
    label="Individual",
    definition=(
        "An id for an individual animal. In the context of bioacoustic "
        "annotation, this term is used to label vocalizations that are "
        "attributed to a specific individual."
    ),
)
"""Term used for tags identifying a specific individual animal."""

generic_class = data.Term(
    name="soundevent:class",
    label="Class",
    definition=(
        "A generic term representing the name of a class within a "
        "classification model. Its specific meaning is determined by "
        "the model's application."
    ),
)
"""Generic term representing a classification model's output class label."""

terms.register_term_set(
    terms.TermSet(
        terms=[
            generic_class,
            individual,
            call_type,
            data_source,
        ],
        aliases={
            "class": generic_class.name,
            "individual": individual.name,
            "event": call_type.name,
            "source": data_source.name,
            "call_type": call_type.name,
        },
    ),
    override_existing=True,
)


class TagInfo(BaseModel):
    """Represents information needed to define a specific Tag.

    This model is typically used in configuration files (e.g., YAML) to
    specify tags used for filtering, target class definition, or associating
    tags with output classes. It links a tag value to a term definition
    via the term's registry key.

    Attributes
    ----------
    value : str
        The value of the tag (e.g., "Myotis myotis", "Echolocation").
    key : str, default="class"
        The key (alias) of the term associated with this tag. Defaults to
        "class", implying it represents a classification target label by
        default.
    """

    value: str
    key: str = GENERIC_CLASS_KEY


def get_tag_from_info(tag_info: TagInfo) -> data.Tag:
    """Creates a soundevent.data.Tag object from TagInfo data.

    Looks up the term using the key in the provided `tag_info` and constructs a
    Tag object.

    Parameters
    ----------
    tag_info : TagInfo
        The TagInfo object containing the value and term key.

    Returns
    -------
    soundevent.data.Tag
        A soundevent.data.Tag object corresponding to the input info.

    Raises
    ------
    KeyError
        If the term key specified in `tag_info.key` is not found.
    """
    term = terms.get_term(tag_info.key)

    if not term:
        raise KeyError(f"Key {tag_info.key} not found")

    return data.Tag(term=term, value=tag_info.value)
