from pathlib import Path
from typing import Callable, List, Set

import pytest
from soundevent import data

from batdetect2.targets.filtering import (
    FilterConfig,
    FilterRule,
    build_filter_from_rule,
    build_sound_event_filter,
    contains_tags,
    does_not_have_tags,
    equal_tags,
    has_any_tag,
    load_filter_config,
    load_filter_from_config,
)
from batdetect2.targets.terms import TagInfo, generic_class


@pytest.fixture
def create_annotation(
    sound_event: data.SoundEvent,
) -> Callable[[List[str]], data.SoundEventAnnotation]:
    """Helper function to create a SoundEventAnnotation with given tags."""

    def factory(tags: List[str]) -> data.SoundEventAnnotation:
        return data.SoundEventAnnotation(
            sound_event=sound_event,
            tags=[
                data.Tag(
                    term=generic_class,
                    value=tag,
                )
                for tag in tags
            ],
        )

    return factory


def create_tag_set(tags: List[str]) -> Set[data.Tag]:
    """Helper function to create a set of data.Tag objects from a list of strings."""
    return {
        data.Tag(
            term=generic_class,
            value=tag,
        )
        for tag in tags
    }


def test_has_any_tag(create_annotation):
    annotation = create_annotation(["tag1", "tag2"])
    tags = create_tag_set(["tag1", "tag3"])
    assert has_any_tag(annotation, tags) is True

    annotation = create_annotation(["tag2", "tag4"])
    tags = create_tag_set(["tag1", "tag3"])
    assert has_any_tag(annotation, tags) is False


def test_contains_tags(create_annotation):
    annotation = create_annotation(["tag1", "tag2", "tag3"])
    tags = create_tag_set(["tag1", "tag2"])
    assert contains_tags(annotation, tags) is True

    annotation = create_annotation(["tag1", "tag2"])
    tags = create_tag_set(["tag1", "tag2", "tag3"])
    assert contains_tags(annotation, tags) is False


def test_does_not_have_tags(create_annotation):
    annotation = create_annotation(["tag1", "tag2"])
    tags = create_tag_set(["tag3", "tag4"])
    assert does_not_have_tags(annotation, tags) is True

    annotation = create_annotation(["tag1", "tag2"])
    tags = create_tag_set(["tag1", "tag3"])
    assert does_not_have_tags(annotation, tags) is False


def test_equal_tags(create_annotation):
    annotation = create_annotation(["tag1", "tag2"])
    tags = create_tag_set(["tag1", "tag2"])
    assert equal_tags(annotation, tags) is True

    annotation = create_annotation(["tag1", "tag2", "tag3"])
    tags = create_tag_set(["tag1", "tag2"])
    assert equal_tags(annotation, tags) is False


def test_build_filter_from_rule():
    rule_any = FilterRule(match_type="any", tags=[TagInfo(value="tag1")])
    build_filter_from_rule(rule_any)

    rule_all = FilterRule(match_type="all", tags=[TagInfo(value="tag1")])
    build_filter_from_rule(rule_all)

    rule_exclude = FilterRule(
        match_type="exclude", tags=[TagInfo(value="tag1")]
    )
    build_filter_from_rule(rule_exclude)

    rule_equal = FilterRule(match_type="equal", tags=[TagInfo(value="tag1")])
    build_filter_from_rule(rule_equal)

    with pytest.raises(ValueError):
        FilterRule(match_type="invalid", tags=[TagInfo(value="tag1")])  # type: ignore
        build_filter_from_rule(
            FilterRule(match_type="invalid", tags=[TagInfo(value="tag1")])  # type: ignore
        )


def test_build_filter_from_config(create_annotation):
    config = FilterConfig(
        rules=[
            FilterRule(match_type="any", tags=[TagInfo(value="tag1")]),
            FilterRule(match_type="any", tags=[TagInfo(value="tag2")]),
        ]
    )
    filter_from_config = build_sound_event_filter(config)

    annotation_pass = create_annotation(["tag1", "tag2"])
    assert filter_from_config(annotation_pass)

    annotation_fail = create_annotation(["tag1"])
    assert not filter_from_config(annotation_fail)


def test_load_filter_config(tmp_path: Path):
    test_config_path = tmp_path / "filtering.yaml"
    test_config_path.write_text(
        """
rules:
    - match_type: any
      tags:
        - value: tag1
    """
    )
    config = load_filter_config(test_config_path)
    assert isinstance(config, FilterConfig)
    assert len(config.rules) == 1
    rule = config.rules[0]
    assert rule.match_type == "any"
    assert len(rule.tags) == 1
    assert rule.tags[0].value == "tag1"


def test_load_filter_from_config(tmp_path: Path, create_annotation):
    test_config_path = tmp_path / "filtering.yaml"
    test_config_path.write_text(
        """
rules:
    - match_type: any
      tags:
        - value: tag1
    """
    )

    filter_result = load_filter_from_config(test_config_path)
    annotation = create_annotation(["tag1", "tag3"])
    assert filter_result(annotation)

    test_config_path = tmp_path / "filtering.yaml"
    test_config_path.write_text(
        """
rules:
    - match_type: any
      tags:
        - value: tag2
    """
    )

    filter_result = load_filter_from_config(test_config_path)
    annotation = create_annotation(["tag1", "tag3"])
    assert filter_result(annotation) is False
