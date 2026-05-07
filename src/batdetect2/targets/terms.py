"""Manages the vocabulary for defining training targets."""

from soundevent import data, terms

__all__ = [
    "call_type",
    "individual",
    "data_source",
    "generic_class",
]

# The default key used to reference the 'generic_class' term.
# Often used implicitly when defining classification targets.
GENERIC_CLASS_KEY = "class"


data_source = data.Term(
    name="dcterms:source",
    label="Source",
    uri="http://purl.org/dc/terms/source",
    definition=(
        "A related resource from which the described resource is derived."
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

dataset_split = data.Term(
    name="batdetect2:split",
    label="Dataset Split",
    definition=(
        "Identifies the specific data partition (e.g., 'train', 'test') "
        "that the item belongs to within an experimental setup. "
        "The expected value is a literal text string."
    ),
)
"""Custom metadata term defining the machine learning partition of an item."""

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
