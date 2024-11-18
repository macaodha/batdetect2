from inspect import getmembers
from typing import Optional

from pydantic import BaseModel
from soundevent import data, terms

__all__ = [
    "call_type",
    "individual",
    "get_term_from_info",
    "get_tag_from_info",
    "TermInfo",
    "TagInfo",
]


class TermInfo(BaseModel):
    label: Optional[str]
    name: Optional[str]
    uri: Optional[str]


class TagInfo(BaseModel):
    value: str
    term: Optional[TermInfo] = None
    key: Optional[str] = None
    label: Optional[str] = None


call_type = data.Term(
    name="soundevent:call_type",
    label="Call Type",
    definition="A broad categorization of animal vocalizations based on their intended function or purpose (e.g., social, distress, mating, territorial, echolocation).",
)

individual = data.Term(
    name="soundevent:individual",
    label="Individual",
    definition="An id for an individual animal. In the context of bioacoustic annotation, this term is used to label vocalizations that are attributed to a specific individual.",
)


ALL_TERMS = [
    *getmembers(terms, lambda x: isinstance(x, data.Term)),
    call_type,
    individual,
]


def get_term_from_info(term_info: TermInfo) -> data.Term:
    for term in ALL_TERMS:
        if term_info.name and term_info.name == term.name:
            return term

        if term_info.label and term_info.label == term.label:
            return term

        if term_info.uri and term_info.uri == term.uri:
            return term

    if term_info.name is None:
        if term_info.label is None:
            raise ValueError("At least one of name or label must be provided.")

        term_info.name = (
            f"soundevent:{term_info.label.lower().replace(' ', '_')}"
        )

    if term_info.label is None:
        term_info.label = term_info.name

    return data.Term(
        name=term_info.name,
        label=term_info.label,
        uri=term_info.uri,
        definition="Unknown",
    )


def get_tag_from_info(tag_info: TagInfo) -> data.Tag:
    if tag_info.term:
        term = get_term_from_info(tag_info.term)
    elif tag_info.key:
        term = data.term_from_key(tag_info.key)
    else:
        raise ValueError("Either term or key must be provided in tag info.")

    return data.Tag(term=term, value=tag_info.value)
