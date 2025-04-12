import pytest
import yaml
from soundevent import data

from batdetect2.targets import terms
from batdetect2.targets.terms import (
    TagInfo,
    TermRegistry,
    load_terms_from_config,
)


def test_term_registry_initialization():
    registry = TermRegistry()
    assert registry._terms == {}

    initial_terms = {
        "test_term": data.Term(name="test", label="Test", definition="test")
    }
    registry = TermRegistry(terms=initial_terms)
    assert registry._terms == initial_terms


def test_term_registry_add_term():
    registry = TermRegistry()
    term = data.Term(name="test", label="Test", definition="test")
    registry.add_term("test_key", term)
    assert registry._terms["test_key"] == term


def test_term_registry_get_term():
    registry = TermRegistry()
    term = data.Term(name="test", label="Test", definition="test")
    registry.add_term("test_key", term)
    retrieved_term = registry.get_term("test_key")
    assert retrieved_term == term


def test_term_registry_add_custom_term():
    registry = TermRegistry()
    term = registry.add_custom_term(
        "custom_key", name="custom", label="Custom", definition="A custom term"
    )
    assert registry._terms["custom_key"] == term
    assert term.name == "custom"
    assert term.label == "Custom"
    assert term.definition == "A custom term"


def test_term_registry_add_duplicate_term():
    registry = TermRegistry()
    term = data.Term(name="test", label="Test", definition="test")
    registry.add_term("test_key", term)
    with pytest.raises(KeyError):
        registry.add_term("test_key", term)


def test_term_registry_get_term_not_found():
    registry = TermRegistry()
    with pytest.raises(KeyError):
        registry.get_term("non_existent_key")


def test_term_registry_get_keys():
    registry = TermRegistry()
    term1 = data.Term(name="test1", label="Test1", definition="test")
    term2 = data.Term(name="test2", label="Test2", definition="test")
    registry.add_term("key1", term1)
    registry.add_term("key2", term2)
    keys = registry.get_keys()
    assert set(keys) == {"key1", "key2"}


def test_get_term_from_key():
    term = terms.get_term_from_key("call_type")
    assert term == terms.call_type

    custom_registry = TermRegistry()
    custom_term = data.Term(name="custom", label="Custom", definition="test")
    custom_registry.add_term("custom_key", custom_term)
    term = terms.get_term_from_key("custom_key", term_registry=custom_registry)
    assert term == custom_term


def test_get_term_keys():
    keys = terms.get_term_keys()
    assert "call_type" in keys
    assert "individual" in keys
    assert terms.GENERIC_CLASS_KEY in keys

    custom_registry = TermRegistry()
    custom_term = data.Term(name="custom", label="Custom", definition="test")
    custom_registry.add_term("custom_key", custom_term)
    keys = terms.get_term_keys(term_registry=custom_registry)
    assert "custom_key" in keys


def test_tag_info_and_get_tag_from_info():
    tag_info = TagInfo(value="Myotis myotis", key="call_type")
    tag = terms.get_tag_from_info(tag_info)
    assert tag.value == "Myotis myotis"
    assert tag.term == terms.call_type


def test_get_tag_from_info_key_not_found():
    tag_info = TagInfo(value="test", key="non_existent_key")
    with pytest.raises(KeyError):
        terms.get_tag_from_info(tag_info)


def test_load_terms_from_config(tmp_path):
    config_data = {
        "terms": [
            {
                "key": "species",
                "name": "dwc:scientificName",
                "label": "Scientific Name",
            },
            {
                "key": "my_custom_term",
                "name": "soundevent:custom_term",
                "definition": "Describes a specific project attribute",
            },
        ]
    }
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    loaded_terms = load_terms_from_config(config_file)
    assert "species" in loaded_terms
    assert "my_custom_term" in loaded_terms
    assert loaded_terms["species"].name == "dwc:scientificName"
    assert loaded_terms["my_custom_term"].name == "soundevent:custom_term"


def test_load_terms_from_config_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_terms_from_config("non_existent_file.yaml")


def test_load_terms_from_config_validation_error(tmp_path):
    config_data = {
        "terms": [
            {
                "key": "species",
                "uri": "dwc:scientificName",
                "label": 123,
            },  # Invalid label type
        ]
    }
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    with pytest.raises(ValueError):
        load_terms_from_config(config_file)


def test_load_terms_from_config_key_already_exists(tmp_path):
    config_data = {
        "terms": [
            {
                "key": "call_type",
                "uri": "dwc:scientificName",
                "label": "Scientific Name",
            },  # Duplicate key
        ]
    }
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    with pytest.raises(KeyError):
        load_terms_from_config(config_file)
