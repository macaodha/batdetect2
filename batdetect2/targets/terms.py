"""Manages the vocabulary (Terms and Tags) for defining training targets.

This module provides the necessary tools to declare, register, and manage the
set of `soundevent.data.Term` objects used throughout the `batdetect2.targets`
sub-package. It establishes a consistent vocabulary for filtering,
transforming, and classifying sound events based on their annotations (Tags).

The core component is the `TermRegistry`, which maps unique string keys
(aliases) to specific `Term` definitions. This allows users to refer to complex
terms using simple, consistent keys in configuration files and code.

Terms can be pre-defined, loaded from the `soundevent.terms` library, defined
programmatically, or loaded from external configuration files (e.g., YAML).
"""

from collections.abc import Mapping
from inspect import getmembers
from typing import Dict, List, Optional

from pydantic import BaseModel, Field
from soundevent import data, terms

from batdetect2.configs import load_config

__all__ = [
    "call_type",
    "individual",
    "get_tag_from_info",
    "TermInfo",
    "TagInfo",
]

# The default key used to reference the 'generic_class' term.
# Often used implicitly when defining classification targets.
GENERIC_CLASS_KEY = "class"


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


class TermRegistry(Mapping[str, data.Term]):
    """Manages a registry mapping unique keys to Term definitions.

    This class acts as the central repository for the vocabulary of terms
    used within the target definition process. It allows registering terms
    with simple string keys and retrieving them consistently.
    """

    def __init__(self, terms: Optional[Dict[str, data.Term]] = None):
        """Initializes the TermRegistry.

        Parameters
        ----------
        terms : dict[str, soundevent.data.Term], optional
            An optional dictionary of initial key-to-Term mappings
            to populate the registry with. Defaults to an empty registry.
        """
        self._terms: Dict[str, data.Term] = terms or {}

    def __getitem__(self, key: str) -> data.Term:
        return self._terms[key]

    def __len__(self) -> int:
        return len(self._terms)

    def __iter__(self):
        return iter(self._terms)

    def add_term(self, key: str, term: data.Term) -> None:
        """Adds a Term object to the registry with the specified key.

        Parameters
        ----------
        key : str
            The unique string key to associate with the term.
        term : soundevent.data.Term
            The soundevent.data.Term object to register.

        Raises
        ------
        KeyError
            If a term with the provided key already exists in the
            registry.
        """
        if key in self._terms:
            raise KeyError("A term with the provided key already exists.")

        self._terms[key] = term

    def get_term(self, key: str) -> data.Term:
        """Retrieves a registered term by its unique key.

        Parameters
        ----------
        key : str
            The unique string key of the term to retrieve.

        Returns
        -------
        soundevent.data.Term
            The corresponding soundevent.data.Term object.

        Raises
        ------
        KeyError
            If no term with the specified key is found, with a
            helpful message suggesting listing available keys.
        """
        try:
            return self._terms[key]
        except KeyError as err:
            raise KeyError(
                "No term found for key "
                f"'{key}'. Ensure it is registered or loaded. "
                "Use `get_term_keys()` to list available terms."
            ) from err

    def add_custom_term(
        self,
        key: str,
        name: Optional[str] = None,
        uri: Optional[str] = None,
        label: Optional[str] = None,
        definition: Optional[str] = None,
    ) -> data.Term:
        """Creates a new Term from attributes and adds it to the registry.

        This is useful for defining terms directly in code or when loading
        from configuration files where only attributes are provided.

        If optional fields (`name`, `label`, `definition`) are not provided,
        reasonable defaults are used (`key` for name/label, "Unknown" for
        definition).

        Parameters
        ----------
        key : str
            The unique string key for the new term.
        name : str, optional
            The name for the new term (defaults to `key`).
        uri : str, optional
            The URI for the new term (optional).
        label : str, optional
            The display label for the new term (defaults to `key`).
        definition : str, optional
            The definition for the new term (defaults to "Unknown").

        Returns
        -------
        soundevent.data.Term
            The newly created and registered soundevent.data.Term object.

        Raises
        ------
        KeyError
            If a term with the provided key already exists.
        """
        term = data.Term(
            name=name or key,
            label=label or key,
            uri=uri,
            definition=definition or "Unknown",
        )
        self.add_term(key, term)
        return term

    def get_keys(self) -> List[str]:
        """Returns a list of all keys currently registered.

        Returns
        -------
        list[str]
            A list of strings representing the keys of all registered terms.
        """
        return list(self._terms.keys())

    def get_terms(self) -> List[data.Term]:
        """Returns a list of all registered terms.

        Returns
        -------
        list[soundevent.data.Term]
            A list containing all registered Term objects.
        """
        return list(self._terms.values())


registry = TermRegistry(
    terms=dict(
        [
            *getmembers(terms, lambda x: isinstance(x, data.Term)),
            ("call_type", call_type),
            ("individual", individual),
            (GENERIC_CLASS_KEY, generic_class),
        ]
    )
)
"""The default, globally accessible TermRegistry instance.

It is pre-populated with standard terms from `soundevent.terms` and common
terms defined in this module (`call_type`, `individual`, `generic_class`).
Functions in this module use this registry by default unless another instance
is explicitly passed.
"""


def get_term_from_key(
    key: str,
    term_registry: TermRegistry = registry,
) -> data.Term:
    """Convenience function to retrieve a term by key from a registry.

    Uses the global default registry unless a specific `term_registry`
    instance is provided.

    Parameters
    ----------
    key : str
        The unique key of the term to retrieve.
    term_registry : TermRegistry, optional
        The TermRegistry instance to search in. Defaults to the global
        `registry`.

    Returns
    -------
    soundevent.data.Term
        The corresponding soundevent.data.Term object.

    Raises
    ------
    KeyError
        If the key is not found in the specified registry.
    """
    return term_registry.get_term(key)


def get_term_keys(term_registry: TermRegistry = registry) -> List[str]:
    """Convenience function to get all registered keys from a registry.

    Uses the global default registry unless a specific `term_registry`
    instance is provided.

    Parameters
    ----------
    term_registry : TermRegistry, optional
        The TermRegistry instance to query. Defaults to the global `registry`.

    Returns
    -------
    list[str]
        A list of strings representing the keys of all registered terms.
    """
    return term_registry.get_keys()


def get_terms(term_registry: TermRegistry = registry) -> List[data.Term]:
    """Convenience function to get all registered terms from a registry.

    Uses the global default registry unless a specific `term_registry`
    instance is provided.

    Parameters
    ----------
    term_registry : TermRegistry, optional
        The TermRegistry instance to query. Defaults to the global `registry`.

    Returns
    -------
    list[soundevent.data.Term]
        A list containing all registered Term objects.
    """
    return term_registry.get_terms()


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
        The key (alias) of the term associated with this tag, as
        registered in the TermRegistry. Defaults to "class", implying
        it represents a classification target label by default.
    """

    value: str
    key: str = GENERIC_CLASS_KEY


def get_tag_from_info(
    tag_info: TagInfo,
    term_registry: TermRegistry = registry,
) -> data.Tag:
    """Creates a soundevent.data.Tag object from TagInfo data.

    Looks up the term using the key in the provided `tag_info` from the
    specified registry and constructs a Tag object.

    Parameters
    ----------
    tag_info : TagInfo
        The TagInfo object containing the value and term key.
    term_registry : TermRegistry, optional
        The TermRegistry instance to use for term lookup. Defaults to the
        global `registry`.

    Returns
    -------
    soundevent.data.Tag
        A soundevent.data.Tag object corresponding to the input info.

    Raises
    ------
    KeyError
        If the term key specified in `tag_info.key` is not found
        in the registry.
    """
    term = get_term_from_key(tag_info.key, term_registry=term_registry)
    return data.Tag(term=term, value=tag_info.value)


class TermInfo(BaseModel):
    """Represents the definition of a Term within a configuration file.

    This model allows users to define custom terms directly in configuration
    files (e.g., YAML) which can then be loaded into the TermRegistry.
    It mirrors the parameters of `TermRegistry.add_custom_term`.

    Attributes
    ----------
    key : str
        The unique key (alias) that will be used to register and
        reference this term.
    label : str, optional
        The optional display label for the term. Defaults to `key`
        if not provided during registration.
    name : str, optional
        The optional formal name for the term. Defaults to `key`
        if not provided during registration.
    uri : str, optional
        The optional URI identifying the term (e.g., from a standard
        vocabulary).
    definition : str, optional
        The optional textual definition of the term. Defaults to
        "Unknown" if not provided during registration.
    """

    key: str
    label: Optional[str] = None
    name: Optional[str] = None
    uri: Optional[str] = None
    definition: Optional[str] = None


class TermConfig(BaseModel):
    """Pydantic schema for loading a list of term definitions from config.

    This model typically corresponds to a section in a configuration file
    (e.g., YAML) containing a list of term definitions to be registered.

    Attributes
    ----------
    terms : list[TermInfo]
        A list of TermInfo objects, each defining a term to be
        registered. Defaults to an empty list.

    Examples
    --------
    Example YAML structure:

    ```yaml
    terms:
      - key: species
        uri: dwc:scientificName
        label: Scientific Name
      - key: my_custom_term
        name: My Custom Term
        definition: Describes a specific project attribute.
      # ... more TermInfo definitions
    ```
    """

    terms: List[TermInfo] = Field(default_factory=list)


def load_terms_from_config(
    path: data.PathLike,
    field: Optional[str] = None,
    term_registry: TermRegistry = registry,
) -> Dict[str, data.Term]:
    """Loads term definitions from a configuration file and registers them.

    Parses a configuration file (e.g., YAML) using the TermConfig schema,
    extracts the list of TermInfo definitions, and adds each one as a
    custom term to the specified TermRegistry instance.

    Parameters
    ----------
    path : data.PathLike
        The path to the configuration file.
    field : str, optional
        Optional key indicating a specific section within the config
        file where the 'terms' list is located. If None, expects the
        list directly at the top level or within a structure matching
        TermConfig schema.
    term_registry : TermRegistry, optional
        The TermRegistry instance to add the loaded terms to. Defaults to
        the global `registry`.

    Returns
    -------
    dict[str, soundevent.data.Term]
        A dictionary mapping the keys of the newly added terms to their
        corresponding Term objects.

    Raises
    ------
    FileNotFoundError
        If the config file path does not exist.
    pydantic.ValidationError
        If the config file structure does not match the TermConfig schema.
    KeyError
        If a term key loaded from the config conflicts with a key
        already present in the registry.
    """
    data = load_config(path, schema=TermConfig, field=field)
    return {
        info.key: term_registry.add_custom_term(
            info.key,
            name=info.name,
            uri=info.uri,
            label=info.label,
            definition=info.definition,
        )
        for info in data.terms
    }
