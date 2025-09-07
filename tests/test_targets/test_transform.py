from pathlib import Path

import pytest
from soundevent import data, terms

from batdetect2.targets import (
    DeriveTagRule,
    MapValueRule,
    ReplaceRule,
    TagInfo,
    TransformConfig,
    build_transformation_from_config,
)
from batdetect2.targets.transform import (
    DerivationRegistry,
    build_transform_from_rule,
)


@pytest.fixture
def derivation_registry():
    return DerivationRegistry()


@pytest.fixture
def term1() -> data.Term:
    term = data.Term(label="Term 1", definition="unknown", name="test:term1")
    terms.add_term(term, key="term1", force=True)
    return term


@pytest.fixture
def term2() -> data.Term:
    term = data.Term(label="Term 2", definition="unknown", name="test:term2")
    terms.add_term(term, key="term2", force=True)
    return term


@pytest.fixture
def term3() -> data.Term:
    term = data.Term(label="Term 3", definition="unknown", name="test:term3")
    terms.add_term(term, key="term3", force=True)
    return term


@pytest.fixture
def annotation(
    sound_event: data.SoundEvent,
    term1: data.Term,
) -> data.SoundEventAnnotation:
    return data.SoundEventAnnotation(
        sound_event=sound_event, tags=[data.Tag(term=term1, value="value1")]
    )

@pytest.fixture
def annotation2(
    sound_event: data.SoundEvent,
    term2: data.Term,
) -> data.SoundEventAnnotation:
    return data.SoundEventAnnotation(
        sound_event=sound_event, tags=[data.Tag(term=term2, value="value2")]
    )


def test_map_value_rule(annotation: data.SoundEventAnnotation):
    rule = MapValueRule(
        rule_type="map_value",
        source_term_key="term1",
        value_mapping={"value1": "value2"},
    )
    transform_fn = build_transform_from_rule(rule)
    transformed_annotation = transform_fn(annotation)
    assert transformed_annotation.tags[0].value == "value2"


def test_map_value_rule_no_match(annotation: data.SoundEventAnnotation):
    rule = MapValueRule(
        rule_type="map_value",
        source_term_key="term1",
        value_mapping={"other_value": "value2"},
    )
    transform_fn = build_transform_from_rule(rule)
    transformed_annotation = transform_fn(annotation)
    assert transformed_annotation.tags[0].value == "value1"


def test_replace_rule(annotation: data.SoundEventAnnotation, term2: data.Term):
    rule = ReplaceRule(
        rule_type="replace",
        original=TagInfo(key="term1", value="value1"),
        replacement=TagInfo(key="term2", value="value2"),
    )
    transform_fn = build_transform_from_rule(rule)
    transformed_annotation = transform_fn(annotation)
    assert transformed_annotation.tags[0].term == term2
    assert transformed_annotation.tags[0].value == "value2"


def test_replace_rule_no_match(
    annotation: data.SoundEventAnnotation,
    term1: data.Term,
    term2: data.Term,
):
    rule = ReplaceRule(
        rule_type="replace",
        original=TagInfo(key="term1", value="wrong_value"),
        replacement=TagInfo(key="term2", value="value2"),
    )
    transform_fn = build_transform_from_rule(rule)
    transformed_annotation = transform_fn(annotation)
    assert transformed_annotation.tags[0].term == term1
    assert transformed_annotation.tags[0].term != term2
    assert transformed_annotation.tags[0].value == "value1"


def test_build_transformation_from_config(
    annotation: data.SoundEventAnnotation,
    annotation2: data.SoundEventAnnotation,
    term1: data.Term,
    term2: data.Term,
    term3: data.Term,
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
    transform = build_transformation_from_config(config)

    transformed_annotation = transform(annotation)
    assert transformed_annotation.tags[0].term == term1
    assert transformed_annotation.tags[0].term != term2
    assert transformed_annotation.tags[0].value == "value2"

    transformed_annotation = transform(annotation2)
    assert transformed_annotation.tags[0].term == term3
    assert transformed_annotation.tags[0].value == "value3"


def test_derive_tag_rule(
    annotation: data.SoundEventAnnotation,
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
        derivation_registry=derivation_registry,
    )
    transformed_annotation = transform_fn(annotation)

    assert len(transformed_annotation.tags) == 1
    assert transformed_annotation.tags[0].term == term1
    assert transformed_annotation.tags[0].value == "value1_derived"


def test_derive_tag_rule_target_term(
    annotation: data.SoundEventAnnotation,
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
    transform_fn = build_transform_from_rule(rule)
    transformed_annotation = transform_fn(annotation)

    assert len(transformed_annotation.tags) == 2
    assert transformed_annotation.tags[0].term == term1
    assert transformed_annotation.tags[0].value == "value1"
    assert transformed_annotation.tags[1].term == term1
    assert transformed_annotation.tags[1].value == "value1_imported"

    # Clean up the temporary file and sys.path
    sys.path.remove(str(tmp_path))


def test_derive_tag_rule_invalid_derivation():
    rule = DeriveTagRule(
        rule_type="derive_tag",
        source_term_key="term1",
        derivation_function="nonexistent_derivation",
    )
    with pytest.raises(KeyError):
        build_transform_from_rule(rule)


def test_build_transform_from_rule_invalid_rule_type():
    class InvalidRule:
        rule_type = "invalid"

    rule = InvalidRule()  # type: ignore

    with pytest.raises(ValueError):
        build_transform_from_rule(rule)  # type: ignore


def test_map_value_rule_target_term(
    annotation: data.SoundEventAnnotation,
    term2: data.Term,
):
    rule = MapValueRule(
        rule_type="map_value",
        source_term_key="term1",
        value_mapping={"value1": "value2"},
        target_term_key="term2",
    )
    transform_fn = build_transform_from_rule(rule)
    transformed_annotation = transform_fn(annotation)
    assert transformed_annotation.tags[0].term == term2
    assert transformed_annotation.tags[0].value == "value2"


def test_map_value_rule_target_term_none(
    annotation: data.SoundEventAnnotation,
    term1: data.Term,
):
    rule = MapValueRule(
        rule_type="map_value",
        source_term_key="term1",
        value_mapping={"value1": "value2"},
        target_term_key=None,
    )
    transform_fn = build_transform_from_rule(rule)
    transformed_annotation = transform_fn(annotation)
    assert transformed_annotation.tags[0].term == term1
    assert transformed_annotation.tags[0].value == "value2"


def test_derive_tag_rule_target_term_none(
    annotation: data.SoundEventAnnotation,
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
