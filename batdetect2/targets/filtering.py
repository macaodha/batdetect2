import logging
from functools import partial
from typing import Callable, List, Literal, Optional, Set

from pydantic import Field
from soundevent import data

from batdetect2.configs import BaseConfig, load_config
from batdetect2.targets.terms import TagInfo, get_tag_from_info

__all__ = [
    "build_filter_from_config",
    "SoundEventFilter",
]


SoundEventFilter = Callable[[data.SoundEventAnnotation], bool]
"""Type alias for a filter function.

A filter function accepts a soundevent.data.SoundEventAnnotation object
and returns True if the annotation should be kept based on the filter's
criteria, or False if it should be discarded.
"""

logger = logging.getLogger(__name__)


class FilterRule(BaseConfig):
    """Defines a single rule for filtering sound event annotations.

    Based on the `match_type`, this rule checks if the tags associated with a
    sound event annotation meet certain criteria relative to the `tags` list
    defined in this rule.

    Attributes
    ----------
    match_type : Literal["any", "all", "exclude", "equal"]
        Determines how the `tags` list is used:
        - "any": Pass if the annotation has at least one tag from the list.
        - "all": Pass if the annotation has all tags from the list (it can
                 have others too).
        - "exclude": Pass if the annotation has none of the tags from the list.
        - "equal": Pass if the annotation's tags are exactly the same set as
                   provided in the list.
    tags : List[TagInfo]
        A list of tags (defined using TagInfo for configuration) that this
        rule operates on.
    """

    match_type: Literal["any", "all", "exclude", "equal"]
    tags: List[TagInfo]


def has_any_tag(
    sound_event_annotation: data.SoundEventAnnotation,
    tags: Set[data.Tag],
) -> bool:
    """Check if the annotation has at least one of the specified tags.

    Parameters
    ----------
    sound_event_annotation : data.SoundEventAnnotation
        The annotation to check.
    tags : Set[data.Tag]
        The set of tags to look for.

    Returns
    -------
    bool
        True if the annotation has one or more tags from the specified set,
        False otherwise.
    """
    sound_event_tags = set(sound_event_annotation.tags)
    return bool(tags & sound_event_tags)


def contains_tags(
    sound_event_annotation: data.SoundEventAnnotation,
    tags: Set[data.Tag],
) -> bool:
    """Check if the annotation contains all of the specified tags.

    The annotation may have additional tags beyond those specified.

    Parameters
    ----------
    sound_event_annotation : data.SoundEventAnnotation
        The annotation to check.
    tags : Set[data.Tag]
        The set of tags that must all be present in the annotation.

    Returns
    -------
    bool
        True if the annotation's tags are a superset of the specified tags,
        False otherwise.
    """
    sound_event_tags = set(sound_event_annotation.tags)
    return tags < sound_event_tags


def does_not_have_tags(
    sound_event_annotation: data.SoundEventAnnotation,
    tags: Set[data.Tag],
):
    """Check if the annotation has none of the specified tags.

    Parameters
    ----------
    sound_event_annotation : data.SoundEventAnnotation
        The annotation to check.
    tags : Set[data.Tag]
        The set of tags that must *not* be present in the annotation.

    Returns
    -------
    bool
        True if the annotation has zero tags in common with the specified set,
        False otherwise.
    """
    return not has_any_tag(sound_event_annotation, tags)


def equal_tags(
    sound_event_annotation: data.SoundEventAnnotation,
    tags: Set[data.Tag],
) -> bool:
    """Check if the annotation's tags are exactly equal to the specified set.

    Parameters
    ----------
    sound_event_annotation : data.SoundEventAnnotation
        The annotation to check.
    tags : Set[data.Tag]
        The exact set of tags the annotation must have.

    Returns
    -------
    bool
        True if the annotation's tags set is identical to the specified set,
        False otherwise.
    """
    sound_event_tags = set(sound_event_annotation.tags)
    return tags == sound_event_tags


def build_filter_from_rule(rule: FilterRule) -> SoundEventFilter:
    """Creates a callable filter function from a single FilterRule.

    Parameters
    ----------
    rule : FilterRule
        The filter rule configuration object.

    Returns
    -------
    SoundEventFilter
        A function that takes a SoundEventAnnotation and returns True if it
        passes the rule, False otherwise.

    Raises
    ------
    ValueError
        If the rule contains an invalid `match_type`.
    """
    tag_set = {get_tag_from_info(tag_info) for tag_info in rule.tags}

    if rule.match_type == "any":
        return partial(has_any_tag, tags=tag_set)

    if rule.match_type == "all":
        return partial(contains_tags, tags=tag_set)

    if rule.match_type == "exclude":
        return partial(does_not_have_tags, tags=tag_set)

    if rule.match_type == "equal":
        return partial(equal_tags, tags=tag_set)

    raise ValueError(
        f"Invalid match type {rule.match_type}. Valid types "
        "are: 'any', 'all', 'exclude' and 'equal'"
    )


def merge_filters(*filters: SoundEventFilter) -> SoundEventFilter:
    """Combines multiple filter functions into a single filter function.

    The resulting filter function applies AND logic: an annotation must pass
    *all* the input filters to pass the merged filter.

    Parameters
    ----------
    *filters_with_rules : Tuple[FilterRule, SoundEventFilter]
        Variable number of tuples, each containing the original FilterRule
        and its corresponding filter function (SoundEventFilter).

    Returns
    -------
    SoundEventFilter
        A single function that returns True only if the annotation passes
        all the input filters.
    """

    def merged_filter(
        sound_event_annotation: data.SoundEventAnnotation,
    ) -> bool:
        for filter_fn in filters:
            if not filter_fn(sound_event_annotation):
                logging.debug(
                    f"Sound event annotation {sound_event_annotation.uuid} "
                    f"excluded due to rule {filter_fn}",
                )
                return False

        return True

    return merged_filter


class FilterConfig(BaseConfig):
    """Configuration model for defining a list of filter rules.

    Attributes
    ----------
    rules : List[FilterRule]
        A list of FilterRule objects. An annotation must pass all rules in
        this list to be considered valid by the filter built from this config.
    """

    rules: List[FilterRule] = Field(default_factory=list)


def build_filter_from_config(config: FilterConfig) -> SoundEventFilter:
    """Builds a merged filter function from a FilterConfig object.

    Creates individual filter functions for each rule in the configuration
    and merges them using AND logic.

    Parameters
    ----------
    config : FilterConfig
        The configuration object containing the list of filter rules.

    Returns
    -------
    SoundEventFilter
        A single callable filter function that applies all defined rules.
    """
    filters = [build_filter_from_rule(rule) for rule in config.rules]
    return merge_filters(*filters)


def load_filter_config(
    path: data.PathLike, field: Optional[str] = None
) -> FilterConfig:
    """Loads the filter configuration from a file.

    Parameters
    ----------
    path : data.PathLike
        Path to the configuration file (YAML).
    field : Optional[str], optional
        If the filter configuration is nested under a specific key in the
        file, specify the key here. Defaults to None.

    Returns
    -------
    FilterConfig
        The loaded and validated filter configuration object.
    """
    return load_config(path, schema=FilterConfig, field=field)


def load_filter_from_config(
    path: data.PathLike, field: Optional[str] = None
) -> SoundEventFilter:
    """Loads filter configuration from a file and builds the filter function.

    This is a convenience function that combines loading the configuration
    and building the final callable filter function.

    Parameters
    ----------
    path : data.PathLike
        Path to the configuration file (YAML).
    field : Optional[str], optional
        If the filter configuration is nested under a specific key in the
        file, specify the key here. Defaults to None.

    Returns
    -------
    SoundEventFilter
        The final merged filter function ready to be used.
    """
    config = load_filter_config(path=path, field=field)
    return build_filter_from_config(config)
