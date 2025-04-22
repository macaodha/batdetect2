from pathlib import Path

import pytest
from soundevent import data

from batdetect2.targets import (
    DeriveTagRule,
    MapValueRule,
    ReplaceRule,
    TagInfo,
    TransformConfig,
    build_transformation_from_config,
)
from batdetect2.targets.terms import TermRegistry
from batdetect2.targets.transform import (
    DerivationRegistry,
    build_transform_from_rule,
)


@pytest.fixture
def term_registry():
    return TermRegistry()


@pytest.fixture
def derivation_registry():
    return DerivationRegistry()


@pytest.fixture
def term1(term_registry: TermRegistry) -> data.Term:
    return term_registry.add_custom_term(key="term1")


@pytest.fixture
def term2(term_registry: TermRegistry) -> data.Term:
    return term_registry.add_custom_term(key="term2")


@pytest.fixture
def annotation(
    sound_event: data.SoundEvent,
    term1: data.Term,
) -> data.SoundEventAnnotation:
    return data.SoundEventAnnotation(
        sound_event=sound_event, tags=[data.Tag(term=term1, value="value1")]
    )


def test_map_value_rule(
    annotation: data.SoundEventAnnotation,
    term_registry: TermRegistry,
):
    rule = MapValueRule(
        rule_type="map_value",
        source_term_key="term1",
        value_mapping={"value1": "value2"},
    )
    transform_fn = build_transform_from_rule(rule, term_registry=term_registry)
    transformed_annotation = transform_fn(annotation)
    assert transformed_annotation.tags[0].value == "value2"


def test_map_value_rule_no_match(
    annotation: data.SoundEventAnnotation,
    term_registry: TermRegistry,
):
    rule = MapValueRule(
        rule_type="map_value",
        source_term_key="term1",
        value_mapping={"other_value": "value2"},
    )
    transform_fn = build_transform_from_rule(rule, term_registry=term_registry)
    transformed_annotation = transform_fn(annotation)
    assert transformed_annotation.tags[0].value == "value1"


def test_replace_rule(
    annotation: data.SoundEventAnnotation,
    term2: data.Term,
    term_registry: TermRegistry,
):
    rule = ReplaceRule(
        rule_type="replace",
        original=TagInfo(key="term1", value="value1"),
        replacement=TagInfo(key="term2", value="value2"),
    )
    transform_fn = build_transform_from_rule(rule, term_registry=term_registry)
    transformed_annotation = transform_fn(annotation)
    assert transformed_annotation.tags[0].term == term2
    assert transformed_annotation.tags[0].value == "value2"


def test_replace_rule_no_match(
    annotation: data.SoundEventAnnotation,
    term_registry: TermRegistry,
    term2: data.Term,
):
    rule = ReplaceRule(
        rule_type="replace",
        original=TagInfo(key="term1", value="wrong_value"),
        replacement=TagInfo(key="term2", value="value2"),
    )
    transform_fn = build_transform_from_rule(rule, term_registry=term_registry)
    transformed_annotation = transform_fn(annotation)
    assert transformed_annotation.tags[0].key == "term1"
    assert transformed_annotation.tags[0].term != term2
    assert transformed_annotation.tags[0].value == "value1"


def test_build_transformation_from_config(
    annotation: data.SoundEventAnnotation,
    term_registry: TermRegistry,
):
    config = TransformConfig(
        rules=[
            MapValueRule(
                rule_type="map_value",
                source_term_key="term1",
                value_mapping={"value1": "value2"},
            ),
            ReplaceRule(
                rule_type="replace",
                original=TagInfo(key="term2", value="value2"),
                replacement=TagInfo(key="term3", value="value3"),
            ),
        ]
    )
    term_registry.add_custom_term("term2")
    term_registry.add_custom_term("term3")
    transform = build_transformation_from_config(
        config,
        term_registry=term_registry,
    )
    transformed_annotation = transform(annotation)
    assert transformed_annotation.tags[0].key == "term1"
    assert transformed_annotation.tags[0].value == "value2"


def test_derive_tag_rule(
    annotation: data.SoundEventAnnotation,
    term_registry: TermRegistry,
    derivation_registry: DerivationRegistry,
    term1: data.Term,
):
    def derivation_func(x: str) -> str:
        return x + "_derived"

    derivation_registry.register("my_derivation", derivation_func)

    rule = DeriveTagRule(
        rule_type="derive_tag",
        source_term_key="term1",
        derivation_function="my_derivation",
    )
    transform_fn = build_transform_from_rule(
        rule,
        term_registry=term_registry,
        derivation_registry=derivation_registry,
    )
    transformed_annotation = transform_fn(annotation)

    assert len(transformed_annotation.tags) == 2
    assert transformed_annotation.tags[0].term == term1
    assert transformed_annotation.tags[0].value == "value1"
    assert transformed_annotation.tags[1].term == term1
    assert transformed_annotation.tags[1].value == "value1_derived"


def test_derive_tag_rule_keep_source_false(
    annotation: data.SoundEventAnnotation,
    term_registry: TermRegistry,
    derivation_registry: DerivationRegistry,
    term1: data.Term,
):
    def derivation_func(x: str) -> str:
        return x + "_derived"

    derivation_registry.register("my_derivation", derivation_func)

    rule = DeriveTagRule(
        rule_type="derive_tag",
        source_term_key="term1",
        derivation_function="my_derivation",
        keep_source=False,
    )
    transform_fn = build_transform_from_rule(
        rule,
        term_registry=term_registry,
        derivation_registry=derivation_registry,
    )
    transformed_annotation = transform_fn(annotation)

    assert len(transformed_annotation.tags) == 1
    assert transformed_annotation.tags[0].term == term1
    assert transformed_annotation.tags[0].value == "value1_derived"


def test_derive_tag_rule_target_term(
    annotation: data.SoundEventAnnotation,
    term_registry: TermRegistry,
    derivation_registry: DerivationRegistry,
    term1: data.Term,
    term2: data.Term,
):
    def derivation_func(x: str) -> str:
        return x + "_derived"

    derivation_registry.register("my_derivation", derivation_func)

    rule = DeriveTagRule(
        rule_type="derive_tag",
        source_term_key="term1",
        derivation_function="my_derivation",
        target_term_key="term2",
    )
    transform_fn = build_transform_from_rule(
        rule,
        term_registry=term_registry,
        derivation_registry=derivation_registry,
    )
    transformed_annotation = transform_fn(annotation)

    assert len(transformed_annotation.tags) == 2
    assert transformed_annotation.tags[0].term == term1
    assert transformed_annotation.tags[0].value == "value1"
    assert transformed_annotation.tags[1].term == term2
    assert transformed_annotation.tags[1].value == "value1_derived"


def test_derive_tag_rule_import_derivation(
    annotation: data.SoundEventAnnotation,
    term_registry: TermRegistry,
    term1: data.Term,
    tmp_path: Path,
):
    # Create a dummy derivation function in a temporary file
    derivation_module_path = (
        tmp_path / "temp_derivation.py"
    )  # Changed to /tmp since /home/santiago is not writable
    derivation_module_path.write_text(
        """
def my_imported_derivation(x: str) -> str:
    return x + "_imported"
"""
    )
    # Ensure the temporary file is importable by adding its directory to sys.path
    import sys

    sys.path.insert(0, str(tmp_path))

    rule = DeriveTagRule(
        rule_type="derive_tag",
        source_term_key="term1",
        derivation_function="temp_derivation.my_imported_derivation",
        import_derivation=True,
    )
    transform_fn = build_transform_from_rule(rule, term_registry=term_registry)
    transformed_annotation = transform_fn(annotation)

    assert len(transformed_annotation.tags) == 2
    assert transformed_annotation.tags[0].term == term1
    assert transformed_annotation.tags[0].value == "value1"
    assert transformed_annotation.tags[1].term == term1
    assert transformed_annotation.tags[1].value == "value1_imported"

    # Clean up the temporary file and sys.path
    sys.path.remove(str(tmp_path))


def test_derive_tag_rule_invalid_derivation(term_registry: TermRegistry):
    rule = DeriveTagRule(
        rule_type="derive_tag",
        source_term_key="term1",
        derivation_function="nonexistent_derivation",
    )
    with pytest.raises(KeyError):
        build_transform_from_rule(rule, term_registry=term_registry)


def test_build_transform_from_rule_invalid_rule_type():
    class InvalidRule:
        rule_type = "invalid"

    rule = InvalidRule()  # type: ignore

    with pytest.raises(ValueError):
        build_transform_from_rule(rule)  # type: ignore


def test_map_value_rule_target_term(
    annotation: data.SoundEventAnnotation,
    term_registry: TermRegistry,
    term2: data.Term,
):
    rule = MapValueRule(
        rule_type="map_value",
        source_term_key="term1",
        value_mapping={"value1": "value2"},
        target_term_key="term2",
    )
    transform_fn = build_transform_from_rule(rule, term_registry=term_registry)
    transformed_annotation = transform_fn(annotation)
    assert transformed_annotation.tags[0].term == term2
    assert transformed_annotation.tags[0].value == "value2"


def test_map_value_rule_target_term_none(
    annotation: data.SoundEventAnnotation,
    term_registry: TermRegistry,
    term1: data.Term,
):
    rule = MapValueRule(
        rule_type="map_value",
        source_term_key="term1",
        value_mapping={"value1": "value2"},
        target_term_key=None,
    )
    transform_fn = build_transform_from_rule(rule, term_registry=term_registry)
    transformed_annotation = transform_fn(annotation)
    assert transformed_annotation.tags[0].term == term1
    assert transformed_annotation.tags[0].value == "value2"


def test_derive_tag_rule_target_term_none(
    annotation: data.SoundEventAnnotation,
    term_registry: TermRegistry,
    derivation_registry: DerivationRegistry,
    term1: data.Term,
):
    def derivation_func(x: str) -> str:
        return x + "_derived"

    derivation_registry.register("my_derivation", derivation_func)

    rule = DeriveTagRule(
        rule_type="derive_tag",
        source_term_key="term1",
        derivation_function="my_derivation",
        target_term_key=None,
    )
    transform_fn = build_transform_from_rule(
        rule,
        term_registry=term_registry,
        derivation_registry=derivation_registry,
    )
    transformed_annotation = transform_fn(annotation)

    assert len(transformed_annotation.tags) == 2
    assert transformed_annotation.tags[0].term == term1
    assert transformed_annotation.tags[0].value == "value1"
    assert transformed_annotation.tags[1].term == term1
    assert transformed_annotation.tags[1].value == "value1_derived"


def test_build_transformation_from_config_empty(
    annotation: data.SoundEventAnnotation,
):
    config = TransformConfig(rules=[])
    transform = build_transformation_from_config(config)
    transformed_annotation = transform(annotation)
    assert transformed_annotation == annotation
