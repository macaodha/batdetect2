from pathlib import Path
from typing import Callable
from uuid import uuid4

import pytest
from pydantic import ValidationError
from soundevent import data

from batdetect2.targets.classes import (
    DEFAULT_SPECIES_LIST,
    ClassesConfig,
    TargetClass,
    _get_default_class_name,
    _get_default_classes,
    _is_target_class,
    build_generic_class_tags,
    build_sound_event_decoder,
    build_sound_event_encoder,
    get_class_names_from_config,
    load_classes_config,
    load_decoder_from_config,
    load_encoder_from_config,
)
from batdetect2.targets.terms import TagInfo, TermRegistry


@pytest.fixture
def sample_annotation(
    sound_event: data.SoundEvent,
    sample_term_registry: TermRegistry,
) -> data.SoundEventAnnotation:
    """Fixture for a sample SoundEventAnnotation."""
    return data.SoundEventAnnotation(
        sound_event=sound_event,
        tags=[
            data.Tag(
                term=sample_term_registry.get_term("species"),
                value="Pipistrellus pipistrellus",
            ),
            data.Tag(
                term=sample_term_registry.get_term("quality"),
                value="Good",
            ),
        ],
    )


@pytest.fixture
def create_temp_yaml(tmp_path: Path) -> Callable[[str], Path]:
    """Create a temporary YAML file with the given content."""

    def factory(content: str) -> Path:
        temp_file = tmp_path / f"{uuid4()}.yaml"
        temp_file.write_text(content)
        return temp_file

    return factory


def test_target_class_creation():
    target_class = TargetClass(
        name="pippip",
        tags=[TagInfo(key="species", value="Pipistrellus pipistrellus")],
    )
    assert target_class.name == "pippip"
    assert target_class.tags[0].key == "species"
    assert target_class.tags[0].value == "Pipistrellus pipistrellus"
    assert target_class.match_type == "all"


def test_classes_config_creation():
    target_class = TargetClass(
        name="pippip",
        tags=[TagInfo(key="species", value="Pipistrellus pipistrellus")],
    )
    config = ClassesConfig(classes=[target_class])
    assert len(config.classes) == 1
    assert config.classes[0].name == "pippip"


def test_classes_config_unique_names():
    target_class1 = TargetClass(
        name="pippip",
        tags=[TagInfo(key="species", value="Pipistrellus pipistrellus")],
    )
    target_class2 = TargetClass(
        name="myodau",
        tags=[TagInfo(key="species", value="Myotis daubentonii")],
    )
    ClassesConfig(classes=[target_class1, target_class2])  # No error


def test_classes_config_non_unique_names():
    target_class1 = TargetClass(
        name="pippip",
        tags=[TagInfo(key="species", value="Pipistrellus pipistrellus")],
    )
    target_class2 = TargetClass(
        name="pippip",
        tags=[TagInfo(key="species", value="Myotis daubentonii")],
    )
    with pytest.raises(ValidationError):
        ClassesConfig(classes=[target_class1, target_class2])


def test_load_classes_config_valid(create_temp_yaml: Callable[[str], Path]):
    yaml_content = """
    classes:
      - name: pippip
        tags:
          - key: species
            value: Pipistrellus pipistrellus
    """
    temp_yaml_path = create_temp_yaml(yaml_content)
    config = load_classes_config(temp_yaml_path)
    assert len(config.classes) == 1
    assert config.classes[0].name == "pippip"


def test_load_classes_config_invalid(create_temp_yaml: Callable[[str], Path]):
    yaml_content = """
    classes:
      - name: pippip
        tags:
          - key: species
            value: Pipistrellus pipistrellus
      - name: pippip
        tags:
          - key: species
            value: Myotis daubentonii
    """
    temp_yaml_path = create_temp_yaml(yaml_content)
    with pytest.raises(ValidationError):
        load_classes_config(temp_yaml_path)


def test_is_target_class_match_all(
    sample_annotation: data.SoundEventAnnotation,
    sample_term_registry: TermRegistry,
):
    tags = {
        data.Tag(
            term=sample_term_registry["species"],
            value="Pipistrellus pipistrellus",
        ),
        data.Tag(term=sample_term_registry["quality"], value="Good"),
    }
    assert _is_target_class(sample_annotation, tags, match_all=True) is True

    tags = {
        data.Tag(
            term=sample_term_registry["species"],
            value="Pipistrellus pipistrellus",
        )
    }
    assert _is_target_class(sample_annotation, tags, match_all=True) is True

    tags = {
        data.Tag(
            term=sample_term_registry["species"], value="Myotis daubentonii"
        )
    }
    assert _is_target_class(sample_annotation, tags, match_all=True) is False


def test_is_target_class_match_any(
    sample_annotation: data.SoundEventAnnotation,
    sample_term_registry: TermRegistry,
):
    tags = {
        data.Tag(
            term=sample_term_registry["species"],
            value="Pipistrellus pipistrellus",
        ),
        data.Tag(term=sample_term_registry["quality"], value="Good"),
    }
    assert _is_target_class(sample_annotation, tags, match_all=False) is True

    tags = {
        data.Tag(
            term=sample_term_registry["species"],
            value="Pipistrellus pipistrellus",
        )
    }
    assert _is_target_class(sample_annotation, tags, match_all=False) is True

    tags = {
        data.Tag(
            term=sample_term_registry["species"], value="Myotis daubentonii"
        )
    }
    assert _is_target_class(sample_annotation, tags, match_all=False) is False


def test_get_class_names_from_config():
    target_class1 = TargetClass(
        name="pippip",
        tags=[TagInfo(key="species", value="Pipistrellus pipistrellus")],
    )
    target_class2 = TargetClass(
        name="myodau",
        tags=[TagInfo(key="species", value="Myotis daubentonii")],
    )
    config = ClassesConfig(classes=[target_class1, target_class2])
    names = get_class_names_from_config(config)
    assert names == ["pippip", "myodau"]


def test_build_encoder_from_config(
    sample_annotation: data.SoundEventAnnotation,
    sample_term_registry: TermRegistry,
):
    config = ClassesConfig(
        classes=[
            TargetClass(
                name="pippip",
                tags=[
                    TagInfo(key="species", value="Pipistrellus pipistrellus")
                ],
            )
        ]
    )
    encoder = build_sound_event_encoder(
        config,
        term_registry=sample_term_registry,
    )
    result = encoder(sample_annotation)
    assert result == "pippip"

    config = ClassesConfig(classes=[])
    encoder = build_sound_event_encoder(
        config,
        term_registry=sample_term_registry,
    )
    result = encoder(sample_annotation)
    assert result is None


def test_load_encoder_from_config_valid(
    sample_annotation: data.SoundEventAnnotation,
    sample_term_registry: TermRegistry,
    create_temp_yaml: Callable[[str], Path],
):
    yaml_content = """
    classes:
      - name: pippip
        tags:
          - key: species
            value: Pipistrellus pipistrellus
    """
    temp_yaml_path = create_temp_yaml(yaml_content)
    encoder = load_encoder_from_config(
        temp_yaml_path,
        term_registry=sample_term_registry,
    )
    # We cannot directly compare the function, so we test it.
    result = encoder(sample_annotation)  # type: ignore
    assert result == "pippip"


def test_load_encoder_from_config_invalid(
    create_temp_yaml: Callable[[str], Path],
    sample_term_registry: TermRegistry,
):
    yaml_content = """
    classes:
      - name: pippip
        tags:
          - key: invalid_key
            value: Pipistrellus pipistrellus
    """
    temp_yaml_path = create_temp_yaml(yaml_content)
    with pytest.raises(KeyError):
        load_encoder_from_config(
            temp_yaml_path,
            term_registry=sample_term_registry,
        )


def test_get_default_class_name():
    assert _get_default_class_name("Myotis daubentonii") == "myodau"


def test_get_default_classes():
    default_classes = _get_default_classes()
    assert len(default_classes) == len(DEFAULT_SPECIES_LIST)
    first_class = default_classes[0]
    assert isinstance(first_class, TargetClass)
    assert first_class.name == _get_default_class_name(DEFAULT_SPECIES_LIST[0])
    assert first_class.tags[0].key == "class"
    assert first_class.tags[0].value == DEFAULT_SPECIES_LIST[0]


def test_build_decoder_from_config(sample_term_registry: TermRegistry):
    config = ClassesConfig(
        classes=[
            TargetClass(
                name="pippip",
                tags=[
                    TagInfo(key="species", value="Pipistrellus pipistrellus")
                ],
                output_tags=[TagInfo(key="call_type", value="Echolocation")],
            )
        ],
        generic_class=[TagInfo(key="order", value="Chiroptera")],
    )
    decoder = build_sound_event_decoder(
        config, term_registry=sample_term_registry
    )
    tags = decoder("pippip")
    assert len(tags) == 1
    assert tags[0].term == sample_term_registry["call_type"]
    assert tags[0].value == "Echolocation"

    # Test when output_tags is None, should fall back to tags
    config = ClassesConfig(
        classes=[
            TargetClass(
                name="pippip",
                tags=[
                    TagInfo(key="species", value="Pipistrellus pipistrellus")
                ],
            )
        ],
        generic_class=[TagInfo(key="order", value="Chiroptera")],
    )
    decoder = build_sound_event_decoder(
        config, term_registry=sample_term_registry
    )
    tags = decoder("pippip")
    assert len(tags) == 1
    assert tags[0].term == sample_term_registry["species"]
    assert tags[0].value == "Pipistrellus pipistrellus"

    # Test raise_on_unmapped=True
    decoder = build_sound_event_decoder(
        config, term_registry=sample_term_registry, raise_on_unmapped=True
    )
    with pytest.raises(ValueError):
        decoder("unknown_class")

    # Test raise_on_unmapped=False
    decoder = build_sound_event_decoder(
        config, term_registry=sample_term_registry, raise_on_unmapped=False
    )
    tags = decoder("unknown_class")
    assert len(tags) == 0


def test_load_decoder_from_config_valid(
    create_temp_yaml: Callable[[str], Path],
    sample_term_registry: TermRegistry,
):
    yaml_content = """
    classes:
      - name: pippip
        tags:
          - key: species
            value: Pipistrellus pipistrellus
        output_tags:
          - key: call_type
            value: Echolocation
    generic_class:
      - key: order
        value: Chiroptera
    """
    temp_yaml_path = create_temp_yaml(yaml_content)
    decoder = load_decoder_from_config(
        temp_yaml_path, term_registry=sample_term_registry
    )
    tags = decoder("pippip")
    assert len(tags) == 1
    assert tags[0].term == sample_term_registry["call_type"]
    assert tags[0].value == "Echolocation"


def test_build_generic_class_tags_from_config(
    sample_term_registry: TermRegistry,
):
    config = ClassesConfig(
        classes=[
            TargetClass(
                name="pippip",
                tags=[
                    TagInfo(key="species", value="Pipistrellus pipistrellus")
                ],
            )
        ],
        generic_class=[
            TagInfo(key="order", value="Chiroptera"),
            TagInfo(key="call_type", value="Echolocation"),
        ],
    )
    generic_tags = build_generic_class_tags(
        config, term_registry=sample_term_registry
    )
    assert len(generic_tags) == 2
    assert generic_tags[0].term == sample_term_registry["order"]
    assert generic_tags[0].value == "Chiroptera"
    assert generic_tags[1].term == sample_term_registry["call_type"]
    assert generic_tags[1].value == "Echolocation"
